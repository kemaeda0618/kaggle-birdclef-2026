"""Generate exp027: NB4 ProtoSSM-only (W_PROTO=1.0, W_MLP=0.0) ablation.

NB4 baseline (ProtoSSM 0.5 + MLP 0.5 LB 0.924) から MLP 削除版。
目的: data 軸の base model を確立、MLP dampening が無い分 data 改善が直接 amplification 効く。

仮説: exp024 ProtoSSM v5 upgrade が effect 0 だった = MLP 50% が ProtoSSM 改善を吸収してた。
W_PROTO=1.0 で MLP 抜けば ProtoSSM の改善が直接 LB に反映される。

期待 LB 分布:
- ≥ 0.948: MLP は dampening 確定、ProtoSSM only が優位
- 0.946-0.947: 50:50 と同等、MLP 効果限定
- 0.943-0.945: MLP やや貢献あり、許容範囲
- ≤ 0.942: MLP 明確に貢献、ProtoSSM only は risky

Pipeline (NB4 と同じ):
1. Load pre-computed embeddings
2. Train ProtoSSM (5 seeds) on labeled soundscapes
   (MLP Head は学習しても W_MLP=0 で使われない、ただし code 維持)
3. Run Perch ONNX on test_soundscapes (CPU)
4. Inference: TTA x seeds -> ProtoSSM logits only -> retrieval -> FCS -> submission.csv

Usage: python _gen_nb_protossm_only.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src,
    }


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": src,
    }


cells = []

# ── Header ──
cells.append(md_cell("hdr",
    "# exp027: NB4 ProtoSSM-only (W_PROTO=1.0, W_MLP=0.0) ablation\n"
    "\n"
    "CPU submission notebook (90 min limit). MLP stream は学習するが W_MLP=0 で使われない:\n"
    "- ProtoSSM (NB3 v29 LB 0.921 base): 12-window SSM seq + cross-attn + prototypes  ★ W_PROTO=1.0\n"
    "- MLP Head (NB2 v10 LB 0.918 base): per-window MLP + Gated Fusion  ★ W_MLP=0.0 (削除)\n"
    "\n"
    "目的: data 軸の base 確立。exp024 ProtoSSM v5 upgrade が effect 0 だったのは MLP 50% absorption 仮説。\n"
    "MLP 抜きで ProtoSSM 改善が直接 LB に反映される構造へ。\n"
    "\n"
    "Same shared infra: Prior Tables, multi-seed=5, TTA, SC retrieval, file_conf_scale.\n"
    "**v6 addition**: Train Audio KNN Retrieval (265k focal recordings as secondary pool,\n"
    "LAMBDA_TA=0.05). Expands retrieval pool from 792 labeled SC windows to 265k TA windows.\n"
    "Blend: w_proto * logit(P_proto) + w_mlp * logit(P_mlp) + LAMBDA_SC*sc_ret + LAMBDA_TA*ta_ret"))

# ── Install ──
INSTALL = (
    "import subprocess, sys, os, time\n"
    "\n"
    "START = time.time()\n"
    "\n"
    "# Find perch-onnx dataset\n"
    "ONNX_DS = None\n"
    "for _c in [\n"
    '    "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026",\n'
    '    "/kaggle/input/perch-onnx-for-birdclef-2026",\n'
    '    "/kaggle/input/perch-onnx-for-birdclef2026",\n'
    "]:\n"
    "    if os.path.isdir(_c):\n"
    "        ONNX_DS = _c\n"
    "        break\n"
    "\n"
    "if ONNX_DS is None:\n"
    '    print("Available /kaggle/input/:")\n'
    '    for d in sorted(os.listdir("/kaggle/input/")):\n'
    '        print(f"  {d}")\n'
    '        sub = os.path.join("/kaggle/input", d)\n'
    "        if os.path.isdir(sub):\n"
    "            for f in sorted(os.listdir(sub))[:5]:\n"
    '                print(f"    {f}")\n'
    '    raise FileNotFoundError("perch-onnx dataset not found")\n'
    "\n"
    'print(f"ONNX dataset: {ONNX_DS}")\n'
    'print(f"Files: {os.listdir(ONNX_DS)}")\n'
    "\n"
    "# Install onnxruntime from dataset wheel\n"
    'whls = [f for f in os.listdir(ONNX_DS) if f.endswith(".whl")]\n'
    "if whls:\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                           os.path.join(ONNX_DS, whls[0])])\n"
    '    print(f"Installed: {whls[0]}")\n'
    "else:\n"
    '    print("No wheel found, using pre-installed onnxruntime")'
)
cells.append(code_cell("install", INSTALL))

# ── Imports ──
IMPORTS = """\
import gc, re, warnings, glob, random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel

warnings.filterwarnings("ignore")
DEVICE = "cpu"

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)

print(f"onnxruntime {ort.__version__}, torch {torch.__version__}, seed={SEED}")"""
cells.append(code_cell("imports", IMPORTS))

# ── Config ──
CONFIG = """\
# CONFIG (NB3 v29 + NB2 v10 統合)
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12

BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")

EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")
ANURA_DIR = Path("/kaggle/input/datasets/maekeso/birdclef2026-perch-embed-anura")
INAT_DIR  = Path("/kaggle/input/datasets/maekeso/birdclef2026-perch-embed-inat-nonbird")
TEST_DIR = BASE / "test_soundscapes"
TRAIN_SC_DIR = BASE / "train_soundscapes"
TAXONOMY_CSV = BASE / "taxonomy.csv"
SC_LABELS_CSV = BASE / "train_soundscapes_labels.csv"

# ONNX_DS is set in install cell
LABELS_CSV = os.path.join(ONNX_DS, "labels.csv")
ONNX_MODEL = os.path.join(ONNX_DS, "perch_v2.onnx")

# ProtoSSM config (NB3 v29)
D_MODEL = 128
D_STATE = 16
N_SSM_LAYERS = 2

# MLP Head config (NB2 v10)
MLP_HIDDEN = 256

# Common training hyperparams
DROPOUT = 0.1
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
BATCH_ONNX = 48

# Metadata embedding (site/hour parsed from filename)
N_SITES = 32
META_DIM = 8

# Multi-seed ensemble
SEEDS = [42, 123, 777, 2024, 9999]

# v4 = v2 base (SWA on) + Knowledge Distillation.
# v3 (Focal γ=2.0 単独) LB 0.920 → revert. Focal は pos_weight と過剰圧力で悪化。
# SWA は v2 で ±0.000 だったが v2 を baseline と決めたので維持。
USE_SWA = True
SWA_START_FRAC = 0.65

# Knowledge Distillation (v4): Perch の sigmoid output を soft target として
# loss = BCE(out, hard) + LAMBDA_KD * BCE(out, sigmoid(perch_logit))
# 66 files の partial label noise を Perch baseline で正則化する目的。
LAMBDA_KD = 0.15

# Aggregation across (TTA x seeds)
AGG_MODE = "mean"
BLEND_ALPHA = 0.5

# TTA shifts
TTA_SHIFTS = [-1, 0, 1]

# Prior tables (site/hour co-occurrence)
LAMBDA_PRIOR = 0.3
PRIOR_STRENGTH_SITE = 8.0
PRIOR_STRENGTH_HOUR = 8.0
PRIOR_STRENGTH_SH = 4.0

# Hierarchical Site-Conditioned KNN retrieval (inference-only)
RETRIEVAL_K = 10
RETRIEVAL_TAU = 0.05
RETRIEVAL_ALPHA_SITE = 1.5
RETRIEVAL_ALPHA_HOUR = 1.2
LAMBDA_RETRIEVAL = 0.10
RETRIEVAL_EPS = 1e-4

# ★ exp027 ablation: ProtoSSM only (NB4 baseline W_PROTO=0.5, W_MLP=0.5)
# 仮説: MLP 50% が ProtoSSM の data 改善 (exp024 ProtoSSM v5 等) を absorption
# 検証: W_MLP=0 で MLP stream を切り離し、ProtoSSM 単独で LB 計測
W_PROTO = 1.0
W_MLP = 0.0

# file_confidence_scale
FCS_TOP_K = 2
FCS_POWER = 0.4

# v11: E19 — file-level species consistency boost (logit space)
# EDA: 隣接 window Jaccard 0.918 = 同一ファイル内 species 構成は静的
# boosted = (1 - BETA) * window_logit + BETA * file_signal
USE_E19 = True
E19_AGG  = "max"   # "max" / "mean" / "median"
E19_BETA = 0.2

# Train audio retrieval pool (v6): expand soundscape pool (792) → TA pool (265k)
USE_TA_RETRIEVAL = True
RETRIEVAL_TA_K = 20
# v10: class-specific LAMBDA — Aves keeps 0.05, non-Aves boosted 3x to 0.15
# (external non-Aves pool {AnuraSet, iNat} added in v9 had ±0.000 effect with uniform 0.05)
LAMBDA_RETRIEVAL_TA = 0.05         # legacy scalar (kept for fallback / Aves)
LAMBDA_TA_AVES     = 0.05
LAMBDA_TA_NONAVES  = 0.15

META_PAT = re.compile(r"_S(\\d{2})_(\\d{8})_(\\d{2})\\d{4}")

print(f"BASE: {BASE}")
print(f"EMB_DIR: {EMB_DIR}")
print(f"Files: {sorted(os.listdir(EMB_DIR))}")"""
cells.append(code_cell("config", CONFIG))

# ── Taxonomy & Label Mapping ──
TAXONOMY = """\
# TAXONOMY & PERCH LABEL MAPPING
taxonomy = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

bc_labels = (
    pd.read_csv(LABELS_CSV)
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
NO_LABEL_INDEX = len(bc_labels)

taxonomy_m = taxonomy.copy()
taxonomy_m["scientific_name_lookup"] = taxonomy_m["scientific_name"]
bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

mapping = taxonomy_m.merge(
    bc_lookup[["scientific_name_lookup", "bc_index"]],
    on="scientific_name_lookup", how="left",
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)
label_to_bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES = np.array([int(label_to_bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC = BC_INDICES[MAPPED_MASK].astype(np.int32)

proxy_map = {}
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sono = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
]
for _, row in unmapped_non_sono.iterrows():
    genus = str(row["scientific_name"]).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(
            rf"^{re.escape(genus)}\\s", na=False
        )
    ]
    if len(hits) > 0:
        proxy_map[label_to_idx[row["primary_label"]]] = (
            hits["bc_index"].astype(int).values
        )

print(f"Species: {N_CLASSES}, Mapped: {MAPPED_MASK.sum()}, Proxies: {len(proxy_map)}")

# v10: class-specific LAMBDA_TA vector — non-Aves species get 3x boost
class_name_arr = np.array([
    taxonomy.set_index("primary_label").loc[lbl, "class_name"]
    for lbl in PRIMARY_LABELS
])
NON_AVES_MASK = (class_name_arr != "Aves")
LAMBDA_TA_VEC = np.full(N_CLASSES, LAMBDA_TA_AVES, dtype=np.float32)
LAMBDA_TA_VEC[NON_AVES_MASK] = LAMBDA_TA_NONAVES
print(f"LAMBDA_TA: Aves={LAMBDA_TA_AVES} ({(~NON_AVES_MASK).sum()} sp), "
      f"non-Aves={LAMBDA_TA_NONAVES} ({NON_AVES_MASK.sum()} sp)")"""
cells.append(code_cell("taxonomy", TAXONOMY))

# ── Load Embeddings + Labels + Prior Tables + Retrieval Pool ──
LOAD = """\
# LOAD SOUNDSCAPE EMBEDDINGS + LABELS
sc_data = np.load(EMB_DIR / "soundscape_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)
sc_scores = sc_data["scores"].astype(np.float32)
sc_meta = pd.read_parquet(EMB_DIR / "soundscape_meta.parquet")

print(f"Soundscapes: {sc_emb.shape[0]} windows, {sc_meta['filename'].nunique()} files")

sc_labels_df = pd.read_csv(SC_LABELS_CSV)
labeled_files = set(sc_labels_df["filename"].unique())
print(f"Labeled files: {len(labeled_files)}")

label_map = {}
for _, r in sc_labels_df.iterrows():
    fn = r["filename"]
    end_sec = int(pd.Timedelta(r["end"]).total_seconds())
    row_id = f"{Path(fn).stem}_{end_sec}"
    labels_str = str(r["primary_label"]).split(";")
    y = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in labels_str:
        lbl = lbl.strip()
        if lbl in label_to_idx:
            y[label_to_idx[lbl]] = 1.0
    label_map[row_id] = y

is_labeled = sc_meta["filename"].isin(labeled_files).values

def reshape_to_files(arr, meta):
    fnames = meta["filename"].values
    unique = list(dict.fromkeys(fnames))
    n_files = len(unique)
    D = arr.shape[1]
    out = np.zeros((n_files, N_WINDOWS, D), dtype=arr.dtype)
    file_to_idx = {f: i for i, f in enumerate(unique)}
    counters = np.zeros(n_files, dtype=int)
    for ri, fn in enumerate(fnames):
        fi = file_to_idx[fn]
        wi = counters[fi]
        if wi < N_WINDOWS:
            out[fi, wi] = arr[ri]
            counters[fi] += 1
    return out, unique

lab_meta = sc_meta[is_labeled].reset_index(drop=True)
lab_emb_flat = sc_emb[is_labeled]
lab_scores_flat = sc_scores[is_labeled]

lab_emb_files, lab_file_list = reshape_to_files(lab_emb_flat, lab_meta)
lab_scores_files, _ = reshape_to_files(lab_scores_flat, lab_meta)

def parse_meta(fname):
    m = META_PAT.search(fname)
    if m is None:
        return 0, 0
    return int(m.group(1)), int(m.group(3))

lab_site_ids = np.array([parse_meta(fn)[0] for fn in lab_file_list], dtype=np.int64)
lab_hours = np.array([parse_meta(fn)[1] for fn in lab_file_list], dtype=np.int64)
print(f"Labeled metadata: unique sites={np.unique(lab_site_ids).tolist()}, unique hours={np.unique(lab_hours).tolist()}")

lab_labels_files = np.zeros((len(lab_file_list), N_WINDOWS, N_CLASSES), dtype=np.float32)
for fi, fn in enumerate(lab_file_list):
    stem = Path(fn).stem
    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        rid = f"{stem}_{end_sec}"
        if rid in label_map:
            lab_labels_files[fi, wi] = label_map[rid]

print(f"Labeled: {lab_emb_files.shape[0]} files, {lab_emb_files.shape}")
print(f"Active classes: {int((lab_labels_files.sum(axis=(0,1)) > 0).sum())}")

# PRIOR TABLES
file_labels = (lab_labels_files.sum(axis=1) > 0).astype(np.float32)
global_p = file_labels.mean(axis=0).astype(np.float32)

prior_site_ids = sorted(set(int(s) for s in lab_site_ids))
site_to_pi = {s: i for i, s in enumerate(prior_site_ids)}
site_n = np.zeros(len(prior_site_ids), dtype=np.float32)
site_p = np.zeros((len(prior_site_ids), N_CLASSES), dtype=np.float32)
for s in prior_site_ids:
    m = (lab_site_ids == s)
    site_n[site_to_pi[s]] = m.sum()
    site_p[site_to_pi[s]] = file_labels[m].mean(axis=0)

prior_hours = sorted(set(int(h) for h in lab_hours))
hour_to_pi = {h: i for i, h in enumerate(prior_hours)}
hour_n = np.zeros(len(prior_hours), dtype=np.float32)
hour_p = np.zeros((len(prior_hours), N_CLASSES), dtype=np.float32)
for h in prior_hours:
    m = (lab_hours == h)
    hour_n[hour_to_pi[h]] = m.sum()
    hour_p[hour_to_pi[h]] = file_labels[m].mean(axis=0)

sh_to_pi = {}
sh_n_list, sh_p_list = [], []
for s in prior_site_ids:
    for h in prior_hours:
        m = (lab_site_ids == s) & (lab_hours == h)
        if m.sum() > 0:
            sh_to_pi[(s, h)] = len(sh_n_list)
            sh_n_list.append(float(m.sum()))
            sh_p_list.append(file_labels[m].mean(axis=0))
sh_n = np.array(sh_n_list, dtype=np.float32) if sh_n_list else np.zeros(0, dtype=np.float32)
sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, N_CLASSES), dtype=np.float32)

def compute_prior_logit(site_id, hour, eps=1e-4):
    p = global_p.astype(np.float32).copy()
    h_i = hour_to_pi.get(int(hour), -1)
    if h_i >= 0:
        nh = hour_n[h_i]
        wh = nh / (nh + PRIOR_STRENGTH_HOUR)
        p = wh * hour_p[h_i] + (1 - wh) * p
    s_i = site_to_pi.get(int(site_id), -1)
    if s_i >= 0:
        ns = site_n[s_i]
        ws = ns / (ns + PRIOR_STRENGTH_SITE)
        p = ws * site_p[s_i] + (1 - ws) * p
    sh_i = sh_to_pi.get((int(site_id), int(hour)), -1)
    if sh_i >= 0:
        nsh = sh_n[sh_i]
        wsh = nsh / (nsh + PRIOR_STRENGTH_SH)
        p = wsh * sh_p[sh_i] + (1 - wsh) * p
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)

lab_prior_files = np.stack(
    [compute_prior_logit(s, h) for s, h in zip(lab_site_ids, lab_hours)]
).astype(np.float32)
print(f"Prior tables: sites={len(prior_site_ids)}, hours={len(prior_hours)}, "
      f"sh={len(sh_to_pi)}; lab_prior shape={lab_prior_files.shape}")

# RETRIEVAL POOL
lab_pool_emb = lab_emb_flat.astype(np.float32)
_lab_pool_norm = np.linalg.norm(lab_pool_emb, axis=1, keepdims=True) + 1e-8
lab_pool_emb_norm = (lab_pool_emb / _lab_pool_norm).astype(np.float32)

lab_pool_labels = np.zeros((len(lab_meta), N_CLASSES), dtype=np.float32)
for _i, _rid in enumerate(lab_meta["row_id"].values):
    if _rid in label_map:
        lab_pool_labels[_i] = label_map[_rid]

lab_pool_site_ids = np.zeros(len(lab_meta), dtype=np.int64)
lab_pool_hours = np.zeros(len(lab_meta), dtype=np.int64)
for _i, _fn in enumerate(lab_meta["filename"].values):
    _s, _h = parse_meta(_fn)
    lab_pool_site_ids[_i] = _s
    lab_pool_hours[_i] = _h

print(f"Retrieval pool: {lab_pool_emb_norm.shape[0]} windows, "
      f"{int((lab_pool_labels.sum(axis=0) > 0).sum())} active classes, "
      f"sites={len(set(lab_pool_site_ids.tolist()))}, hours={len(set(lab_pool_hours.tolist()))}")

# TRAIN AUDIO RETRIEVAL POOL (v6): expand from 792 SC windows to ~265k TA windows
# Focal recordings provide cleaner per-class signal than partial SC labels.
_ta_npz = EMB_DIR / "trainaudio_embeddings.npz"
_ta_pq  = EMB_DIR / "trainaudio_meta.parquet"

if USE_TA_RETRIEVAL and _ta_npz.exists() and _ta_pq.exists():
    _ta_data = np.load(_ta_npz)
    _ta_meta = pd.read_parquet(_ta_pq)

    _ta_emb_raw = _ta_data["embeddings"].astype(np.float32)
    del _ta_data
    gc.collect()

    ta_pool_labels = np.zeros((len(_ta_meta), N_CLASSES), dtype=np.float32)
    for _i, _lbl in enumerate(_ta_meta["primary_label"].values):
        if _lbl in label_to_idx:
            ta_pool_labels[_i, label_to_idx[_lbl]] = 1.0

    _ta_norm = np.linalg.norm(_ta_emb_raw, axis=1, keepdims=True) + 1e-8
    ta_pool_emb_norm = (_ta_emb_raw / _ta_norm).astype(np.float32)
    del _ta_emb_raw, _ta_norm, _ta_meta
    gc.collect()

    print(f"TA pool (BC2026): {ta_pool_emb_norm.shape[0]} windows, "
          f"{int((ta_pool_labels.sum(0) > 0).sum())} active classes")

    # ── External non-Aves pools (v9): AnuraSet + iNat non-Aves ──
    _ext_emb_chunks = []
    _ext_lbl_chunks = []
    # AnuraSet (multi-label, primary_labels csv)
    _an_npz = ANURA_DIR / "anura_embeddings.npz"
    _an_pq  = ANURA_DIR / "anura_meta.parquet"
    if _an_npz.exists() and _an_pq.exists():
        _an_data = np.load(_an_npz)
        _an_meta = pd.read_parquet(_an_pq)
        _an_emb = _an_data["embeddings"].astype(np.float32)
        del _an_data; gc.collect()
        _an_n = _an_emb.shape[0]
        _an_lbl = np.zeros((_an_n, N_CLASSES), dtype=np.float32)
        for _i, _csv in enumerate(_an_meta["primary_labels"].values):
            for _lbl in str(_csv).split(","):
                _lbl = _lbl.strip()
                if _lbl in label_to_idx:
                    _an_lbl[_i, label_to_idx[_lbl]] = 1.0
        _an_norm = np.linalg.norm(_an_emb, axis=1, keepdims=True) + 1e-8
        _an_emb_norm = (_an_emb / _an_norm).astype(np.float32)
        _ext_emb_chunks.append(_an_emb_norm)
        _ext_lbl_chunks.append(_an_lbl)
        print(f"AnuraSet pool: {_an_n} windows, "
              f"{int((_an_lbl.sum(0) > 0).sum())} active classes")
        del _an_emb, _an_norm, _an_meta; gc.collect()
    else:
        print(f"WARN: AnuraSet pool not found at {ANURA_DIR}")

    # iNat non-Aves (single-label primary_label)
    _in_npz = INAT_DIR / "inat_nonaves_embeddings.npz"
    _in_pq  = INAT_DIR / "inat_nonaves_meta.parquet"
    if _in_npz.exists() and _in_pq.exists():
        _in_data = np.load(_in_npz)
        _in_meta = pd.read_parquet(_in_pq)
        _in_emb = _in_data["embeddings"].astype(np.float32)
        del _in_data; gc.collect()
        _in_n = _in_emb.shape[0]
        _in_lbl = np.zeros((_in_n, N_CLASSES), dtype=np.float32)
        for _i, _lbl in enumerate(_in_meta["primary_label"].values):
            if _lbl in label_to_idx:
                _in_lbl[_i, label_to_idx[_lbl]] = 1.0
        _in_norm = np.linalg.norm(_in_emb, axis=1, keepdims=True) + 1e-8
        _in_emb_norm = (_in_emb / _in_norm).astype(np.float32)
        _ext_emb_chunks.append(_in_emb_norm)
        _ext_lbl_chunks.append(_in_lbl)
        print(f"iNat non-Aves pool: {_in_n} windows, "
              f"{int((_in_lbl.sum(0) > 0).sum())} active classes")
        del _in_emb, _in_norm, _in_meta; gc.collect()
    else:
        print(f"WARN: iNat pool not found at {INAT_DIR}")

    if _ext_emb_chunks:
        _ext_emb = np.concatenate(_ext_emb_chunks, axis=0)
        _ext_lbl = np.concatenate(_ext_lbl_chunks, axis=0)
        ta_pool_emb_norm = np.concatenate([ta_pool_emb_norm, _ext_emb], axis=0)
        ta_pool_labels   = np.concatenate([ta_pool_labels, _ext_lbl], axis=0)
        del _ext_emb, _ext_lbl, _ext_emb_chunks, _ext_lbl_chunks; gc.collect()
        print(f"TA pool (after external concat): {ta_pool_emb_norm.shape[0]} windows, "
              f"{int((ta_pool_labels.sum(0) > 0).sum())} active classes")
else:
    print("WARNING: trainaudio_embeddings.npz not found -- TA retrieval disabled")
    USE_TA_RETRIEVAL = False
    ta_pool_emb_norm = None
    ta_pool_labels = None"""
cells.append(code_cell("load", LOAD))

# ── ProtoSSM Model + MLP Head Model ──
MODEL = """\
# === PROTOSSM (NB3 v29) ===
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_D = nn.Linear(d_model, d_model)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.log_A = nn.Parameter(torch.log(A))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B_sz, L, D = x.shape
        delta = F.softplus(self.proj_delta(x))
        B = self.proj_B(x)
        C = self.proj_C(x)
        D_param = self.proj_D(x)
        A = -torch.exp(self.log_A)
        h = torch.zeros(B_sz, self.d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            dt = delta[:, t].unsqueeze(-1)
            dA = torch.exp(A.unsqueeze(0) * dt)
            dB = dt * B[:, t].unsqueeze(1)
            h = h * dA + x[:, t].unsqueeze(-1) * dB
            y = (h * C[:, t].unsqueeze(1)).sum(-1) + D_param[:, t]
            outputs.append(y)
        return self.dropout(torch.stack(outputs, dim=1))


class BiSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.fwd_ssm = SelectiveSSM(d_model, d_state, dropout)
        self.bwd_ssm = SelectiveSSM(d_model, d_state, dropout)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fwd = self.fwd_ssm(x)
        bwd = self.bwd_ssm(x.flip(1)).flip(1)
        out = self.proj(torch.cat([fwd, bwd], dim=-1))
        return self.norm(x + out)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(a))


class ProtoSSM(nn.Module):
    def __init__(self, d_input, d_model, d_state, n_ssm_layers, n_classes,
                 n_windows, dropout=0.1, n_sites=32, meta_dim=8, n_attn_heads=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_layers = nn.ModuleList([
            BiSSMBlock(d_model, d_state, dropout) for _ in range(n_ssm_layers)
        ])
        self.attn_layers = nn.ModuleList([
            CrossAttnBlock(d_model, num_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_ssm_layers)
        ])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None, prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb) + self.pos_emb[:, :emb.shape[1]]
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        for ssm_layer, attn_layer in zip(self.ssm_layers, self.attn_layers):
            x = ssm_layer(x)
            x = attn_layer(x)
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("btd,cd->btc", x_norm, p_norm) * self.temperature + self.bias
        alpha = torch.sigmoid(self.alpha)
        out = alpha * sim + (1 - alpha) * logits
        if prior_logit is not None and lambda_prior > 0:
            out = out + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(out)

    def init_prototypes(self, emb_flat, labels_flat):
        with torch.no_grad():
            x = self.input_proj(emb_flat)
            for ci in range(self.prototypes.shape[0]):
                mask = labels_flat[:, ci] > 0.5
                if mask.sum() > 0:
                    self.prototypes[ci] = x[mask].mean(0)


# === MLP Head (NB2 v10) ===
class MLPHead(nn.Module):
    def __init__(self, d_input, d_hidden, n_classes, dropout=0.1,
                 n_sites=32, meta_dim=8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None,
                prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb)
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        h = self.mlp(x) * self.temperature
        alpha = torch.sigmoid(self.alpha)
        out = alpha * h + (1 - alpha) * logits
        if prior_logit is not None and lambda_prior > 0:
            out = out + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(out)


print("ProtoSSM + MLPHead defined.")"""
cells.append(code_cell("model", MODEL))

# ── Train Both Models (multi-seed) ──
TRAIN = """\
# TRAIN BOTH MODELS (multi-seed ensemble)
# Shared tensors
emb_flat = torch.tensor(lab_emb_flat, dtype=torch.float32)
lab_flat = torch.zeros(len(lab_meta), N_CLASSES, dtype=torch.float32)
for i, rid in enumerate(lab_meta["row_id"]):
    if rid in label_map:
        lab_flat[i] = torch.tensor(label_map[rid])

train_emb = torch.tensor(lab_emb_files, dtype=torch.float32)
train_logits = torch.tensor(lab_scores_files, dtype=torch.float32)
train_labels = torch.tensor(lab_labels_files, dtype=torch.float32)
train_site = torch.tensor(lab_site_ids, dtype=torch.long)
train_hour = torch.tensor(lab_hours, dtype=torch.long)
train_prior = torch.tensor(lab_prior_files, dtype=torch.float32)

pos_counts = train_labels.sum(dim=(0, 1)).clamp(min=1)
neg_counts = train_labels.shape[0] * train_labels.shape[1] - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(max=30.0)

# KD teacher: Perch's mapped logits → sigmoid → soft target
teacher_prob = torch.sigmoid(train_logits).clamp(1e-7, 1 - 1e-7)


def _train_one(model_cls_kwargs, model_factory, seed):
    seed_everything(seed)
    model = model_factory().to(DEVICE)
    if hasattr(model, "init_prototypes"):
        model.init_prototypes(emb_flat, lab_flat)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    epoch_done = 0

    swa_model = AveragedModel(model) if USE_SWA else None
    swa_start = int(N_EPOCHS * SWA_START_FRAC)
    swa_n = 0

    for epoch in range(N_EPOCHS):
        model.train()
        out = model(train_emb, train_logits, site_ids=train_site, hours=train_hour,
                    prior_logit=train_prior, lambda_prior=LAMBDA_PRIOR)
        # Main BCE loss (hard labels)
        loss_main = F.binary_cross_entropy(out, train_labels, reduction="none")
        loss_main = (loss_main * pos_weight.unsqueeze(0).unsqueeze(0)).mean()
        # KD loss (soft target from Perch)
        loss_kd = F.binary_cross_entropy(out, teacher_prob, reduction="mean")
        loss = loss_main + LAMBDA_KD * loss_kd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        epoch_done = epoch + 1
        if USE_SWA and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1
        # disable early stop while SWA is collecting
        if not USE_SWA and patience_counter >= PATIENCE:
            break

    if USE_SWA and swa_n >= 1:
        model.load_state_dict(swa_model.module.state_dict())
    else:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss, epoch_done, swa_n


# Train ProtoSSM x SEEDS
proto_models = []
t0_all = time.time()
for si, seed in enumerate(SEEDS):
    factory = lambda: ProtoSSM(
        d_input=1536, d_model=D_MODEL, d_state=D_STATE,
        n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
        n_windows=N_WINDOWS, dropout=DROPOUT,
        n_sites=N_SITES, meta_dim=META_DIM,
    )
    if si == 0:
        _tmp = factory()
        print(f"Training ProtoSSM (x{len(SEEDS)} seeds): {sum(p.numel() for p in _tmp.parameters())} params/model")
        del _tmp
    t0 = time.time()
    m, bl, ep, sn = _train_one(None, factory, seed)
    proto_models.append(m)
    print(f"  ProtoSSM[seed {seed}] done in {time.time()-t0:.1f}s, best_loss={bl:.4f}, epochs={ep}, swa_n={sn}")
print(f"ProtoSSM ensemble: {len(proto_models)} models in {time.time()-t0_all:.1f}s")

# Train MLPHead x SEEDS
mlp_models = []
t0_all = time.time()
for si, seed in enumerate(SEEDS):
    factory = lambda: MLPHead(
        d_input=1536, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
        dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM,
    )
    if si == 0:
        _tmp = factory()
        print(f"Training MLPHead (x{len(SEEDS)} seeds): {sum(p.numel() for p in _tmp.parameters())} params/model")
        del _tmp
    t0 = time.time()
    m, bl, ep, sn = _train_one(None, factory, seed)
    mlp_models.append(m)
    print(f"  MLPHead[seed {seed}] done in {time.time()-t0:.1f}s, best_loss={bl:.4f}, epochs={ep}, swa_n={sn}")
print(f"MLPHead ensemble: {len(mlp_models)} models in {time.time()-t0_all:.1f}s")"""
cells.append(code_cell("train", TRAIN))

# ── ONNX for Test (+ retrieval helper) ──
ONNX_TEST = """\
# PERCH ONNX FOR TEST INFERENCE (CPU)
print(f"Loading ONNX model: {ONNX_MODEL}")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=["CPUExecutionProvider"])


def read_audio(path, target_samples=None):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import torchaudio
        y = torch.from_numpy(y).unsqueeze(0)
        y = torchaudio.functional.resample(y, sr, SR).squeeze(0).numpy()
    if target_samples is not None:
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
    return y


def map_logits_to_scores(logits):
    scores = np.zeros((logits.shape[0], N_CLASSES), dtype=np.float32)
    scores[:, MAPPED_POS] = logits[:, MAPPED_BC]
    for pos, bc_idx_arr in proxy_map.items():
        scores[:, pos] = logits[:, bc_idx_arr].max(axis=1)
    return scores


def infer_perch_cpu(windows):
    all_emb, all_scores = [], []
    for i in range(0, len(windows), BATCH_ONNX):
        batch = windows[i:i + BATCH_ONNX]
        outputs = session.run(None, {"inputs": batch})
        out_dict = {o.name: v for o, v in zip(session.get_outputs(), outputs)}
        all_emb.append(out_dict["embedding"].astype(np.float32))
        all_scores.append(map_logits_to_scores(out_dict["label"].astype(np.float32)))
    return np.concatenate(all_emb), np.concatenate(all_scores)


def compute_retrieval_logit(test_emb, site_id, hour,
                            K=RETRIEVAL_K, tau=RETRIEVAL_TAU,
                            alpha_site=RETRIEVAL_ALPHA_SITE,
                            alpha_hour=RETRIEVAL_ALPHA_HOUR,
                            eps=RETRIEVAL_EPS):
    e_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    sim = e_norm @ lab_pool_emb_norm.T
    w = np.ones(lab_pool_emb_norm.shape[0], dtype=np.float32)
    w[lab_pool_site_ids == site_id] *= alpha_site
    w[lab_pool_hours == hour] *= alpha_hour
    w_sim = sim * w[None, :]
    K_eff = min(K, w_sim.shape[1])
    topk_idx = np.argpartition(w_sim, -K_eff, axis=1)[:, -K_eff:]
    topk_sim = np.take_along_axis(w_sim, topk_idx, axis=1)
    topk_label = lab_pool_labels[topk_idx]
    sm = topk_sim - topk_sim.max(axis=1, keepdims=True)
    sm = np.exp(sm / max(tau, 1e-6))
    sm = sm / (sm.sum(axis=1, keepdims=True) + 1e-8)
    p = (sm[..., None] * topk_label).sum(axis=1)
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def compute_retrieval_ta_logit(test_emb,
                               K=RETRIEVAL_TA_K, tau=RETRIEVAL_TAU,
                               eps=RETRIEVAL_EPS):
    # KNN retrieval on train_audio pool (~265k focal recordings, primary_label only)
    e_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    sim = e_norm @ ta_pool_emb_norm.T          # (N_windows, 265k)
    K_eff = min(K, sim.shape[1])
    topk_idx = np.argpartition(sim, -K_eff, axis=1)[:, -K_eff:]
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
    topk_label = ta_pool_labels[topk_idx]      # (N_windows, K, N_CLASSES)
    sm = topk_sim - topk_sim.max(axis=1, keepdims=True)
    sm = np.exp(sm / max(tau, 1e-6))
    sm = sm / (sm.sum(axis=1, keepdims=True) + 1e-8)
    p = (sm[..., None] * topk_label).sum(axis=1)
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


print("Perch ONNX (CPU) + retrieval helper ready.")"""
cells.append(code_cell("onnx-test", ONNX_TEST))

# ── Inference (Blend) ──
INFERENCE = """\
# INFERENCE ON TEST SOUNDSCAPES (Blend ProtoSSM + MLP in logit space)
test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))
if len(test_files) == 0:
    print("No test files, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))[:8]

print(f"Test files: {len(test_files)}")

all_row_ids = []
all_probs = []

t0 = time.time()
for _m in proto_models + mlp_models:
    _m.eval()

from concurrent.futures import ThreadPoolExecutor

def _load_windows(fp):
    y = read_audio(fp, target_samples=SR * 60)
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES)

PREFETCH = 4
executor = ThreadPoolExecutor(max_workers=4)
pending = {}
for _i in range(min(PREFETCH, len(test_files))):
    pending[_i] = executor.submit(_load_windows, test_files[_i])


def _ensemble_one(models_list, emb_t, scores_t, site_t, hour_t, prior_t):
    ens_probs = []
    for s in TTA_SHIFTS:
        if s == 0:
            e_shift, sc_shift = emb_t, scores_t
        else:
            e_shift = torch.roll(emb_t, shifts=s, dims=1)
            sc_shift = torch.roll(scores_t, shifts=s, dims=1)
        for m in models_list:
            p = m(e_shift, sc_shift, site_ids=site_t, hours=hour_t,
                  prior_logit=prior_t, lambda_prior=LAMBDA_PRIOR)
            if s != 0:
                p = torch.roll(p, shifts=-s, dims=1)
            ens_probs.append(p)
    stacked = torch.stack(ens_probs, dim=0)
    if AGG_MODE == "max":
        agg = stacked.max(dim=0).values
    elif AGG_MODE == "blend":
        mean_p = stacked.mean(dim=0)
        max_p = stacked.max(dim=0).values
        agg = BLEND_ALPHA * mean_p + (1 - BLEND_ALPHA) * max_p
    else:
        agg = stacked.mean(dim=0)
    return agg


for fi, fpath in enumerate(test_files):
    stem = Path(fpath).stem

    _ni = fi + PREFETCH
    if _ni < len(test_files):
        pending[_ni] = executor.submit(_load_windows, test_files[_ni])

    windows = pending.pop(fi).result()

    emb, scores = infer_perch_cpu(windows)

    _m = META_PAT.search(fpath)
    site_id = int(_m.group(1)) if _m else 0
    hour = int(_m.group(3)) if _m else 0
    site_t = torch.tensor([site_id], dtype=torch.long)
    hour_t = torch.tensor([hour], dtype=torch.long)
    prior_vec = compute_prior_logit(site_id, hour)
    prior_t = torch.tensor(prior_vec, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        scores_t = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)

        # ProtoSSM ensemble
        p_proto = _ensemble_one(proto_models, emb_t, scores_t, site_t, hour_t, prior_t)
        # MLP ensemble
        p_mlp = _ensemble_one(mlp_models, emb_t, scores_t, site_t, hour_t, prior_t)

        # Logit-space blend
        p_proto_c = p_proto.clamp(min=1e-7, max=1 - 1e-7)
        p_mlp_c = p_mlp.clamp(min=1e-7, max=1 - 1e-7)
        l_proto = torch.log(p_proto_c) - torch.log1p(-p_proto_c)
        l_mlp = torch.log(p_mlp_c) - torch.log1p(-p_mlp_c)
        l_blend = W_PROTO * l_proto + W_MLP * l_mlp

        # Add retrieval logit (shared, applied once after blend)
        if LAMBDA_RETRIEVAL > 0:
            ret_logit = compute_retrieval_logit(emb, site_id, hour)
            ret_logit_t = torch.tensor(ret_logit, dtype=torch.float32).unsqueeze(0)
            l_blend = l_blend + LAMBDA_RETRIEVAL * ret_logit_t

        # v6: TA retrieval (train_audio pool focal recordings)
        # v10: class-specific LAMBDA_TA (Aves=0.05, non-Aves=0.15)
        if USE_TA_RETRIEVAL and ta_pool_emb_norm is not None:
            ta_ret_logit = compute_retrieval_ta_logit(emb)
            ta_ret_logit_t = torch.tensor(ta_ret_logit, dtype=torch.float32).unsqueeze(0)
            lam_t = torch.tensor(LAMBDA_TA_VEC, dtype=torch.float32)  # (N_CLASSES,)
            l_blend = l_blend + lam_t * ta_ret_logit_t

        # v11: E19 — file-level species consistency boost (logit space, per file)
        if USE_E19 and E19_BETA > 0:
            if E19_AGG == "max":
                file_sig = l_blend.max(dim=1, keepdim=True).values   # (1, 1, C)
            elif E19_AGG == "mean":
                file_sig = l_blend.mean(dim=1, keepdim=True)
            elif E19_AGG == "median":
                file_sig = l_blend.median(dim=1, keepdim=True).values
            else:
                file_sig = None
            if file_sig is not None:
                l_blend = (1.0 - E19_BETA) * l_blend + E19_BETA * file_sig

        agg = torch.sigmoid(l_blend)
        probs = agg.squeeze(0).numpy()

    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        all_row_ids.append(f"{stem}_{end_sec}")
        all_probs.append(probs[wi])

    if (fi + 1) % 10 == 0 or fi == len(test_files) - 1:
        elapsed = time.time() - t0
        print(f"  [{fi+1}/{len(test_files)}] {elapsed:.0f}s")

executor.shutdown()
print(f"Inference done: {len(all_row_ids)} predictions in {time.time()-t0:.1f}s")"""
cells.append(code_cell("inference", INFERENCE))

# ── Submission ──
SUBMISSION = """\
# BUILD SUBMISSION (with file_confidence_scale + season prior MVP)
preds_array = np.stack(all_probs)

_n, _c = preds_array.shape
assert _n % N_WINDOWS == 0, f"preds_array rows {_n} not divisible by {N_WINDOWS}"
_view = preds_array.reshape(-1, N_WINDOWS, _c)
_sorted = np.sort(_view, axis=1)
_topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
_scale = np.power(_topk_mean, FCS_POWER)
preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)
print(f"file_confidence_scale applied (top_k={FCS_TOP_K}, power={FCS_POWER})")
print(f"  scale stats: min={_scale.min():.4f} max={_scale.max():.4f} mean={_scale.mean():.4f}")

# === Season prior MVP (v12: filename signal A) ===
# Pantanal: wet (Nov-Apr) → Insecta/Amphibia 活性 ↑、dry (Jun-Sep) → 活性 ↓
# row_id 形式: BC2026_Test_0001_S05_20250227_010002_5
from datetime import datetime as _dt
SEASON_PRIORS = {
    'wet':        {'Aves': 1.0, 'Insecta': 1.4, 'Amphibia': 1.6, 'Reptilia': 1.1},
    'transition': {'Aves': 1.0, 'Insecta': 1.0, 'Amphibia': 1.0, 'Reptilia': 1.0},
    'dry':        {'Aves': 1.0, 'Insecta': 0.6, 'Amphibia': 0.4, 'Reptilia': 0.9},
}
def _month_to_season(month):
    if month in (11, 12, 1, 2, 3, 4):  return 'wet'
    elif month in (5, 10):              return 'transition'
    else:                               return 'dry'

# Parse row_id → month → season
_months = np.zeros(len(all_row_ids), dtype=np.int8)
for _i, _rid in enumerate(all_row_ids):
    _m = re.search(r'_(\\d{8})_', _rid)
    if _m is not None:
        try:
            _months[_i] = _dt.strptime(_m.group(1), '%Y%m%d').month
        except Exception:
            _months[_i] = 0
_seasons = np.array([_month_to_season(int(_m)) if _m > 0 else 'transition' for _m in _months])

# Build per-species class_name lookup
_species_class = dict(zip(taxonomy['primary_label'].astype(str), taxonomy['class_name']))
_class_per_col = np.array([_species_class.get(_sp, 'Aves') for _sp in PRIMARY_LABELS])

# Multiplier matrix (vectorized via row × col mask product)
_multipliers = np.ones_like(preds_array, dtype=np.float32)
for _season in ('wet', 'transition', 'dry'):
    _row_mask = _seasons == _season
    if not _row_mask.any():
        continue
    for _cls in ('Aves', 'Insecta', 'Amphibia', 'Reptilia'):
        _col_mask = _class_per_col == _cls
        if not _col_mask.any():
            continue
        _mult = SEASON_PRIORS[_season].get(_cls, 1.0)
        if abs(_mult - 1.0) > 1e-6:
            _multipliers[np.ix_(_row_mask, _col_mask)] = _mult

preds_array = (preds_array * _multipliers).astype(np.float32)
print(f"Season prior applied: wet={int((_seasons=='wet').sum())} / "
      f"trans={int((_seasons=='transition').sum())} / dry={int((_seasons=='dry').sum())} rows")
print(f"  multiplier range: [{_multipliers.min():.2f}, {_multipliers.max():.2f}], "
      f"#cells modified: {(_multipliers != 1.0).sum()}")
# === end season prior ===

submission = pd.DataFrame(preds_array, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])

missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling with zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)

extra = our_ids - expected_ids
if extra:
    print(f"Dropping {len(extra)} extra row_ids")
    submission = submission[submission["row_id"].isin(expected_ids)]

submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

total_time = time.time() - START
print(f"\\nSubmission: {submission.shape}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Mean pred: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"Max pred:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head())"""
cells.append(code_cell("submission", SUBMISSION))

# ── Assemble ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb_protossm_only.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
