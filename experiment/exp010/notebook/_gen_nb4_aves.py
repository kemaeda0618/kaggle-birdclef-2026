"""Generate NB4-AVES: NB4 v7 architecture with Perch replaced by AVES.

Same NB4 v7 pipeline (ProtoSSM + MLPHead + retrieval pool + prior + FCS) but:
- Embedding: AVES wav2vec2 768d (instead of Perch 1536d)
- Test inference: AVES ONNX (instead of Perch ONNX)
- No Perch logit (logits=zeros, LAMBDA_KD=0)

Inputs: NB1f-aves-embed (precomputed AVES embeddings) + aves-onnx-export (test inference)
Runs on Kaggle CPU 90min, expected ~25 min.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from _gen_nb4_blend import code_cell, md_cell

cells = []

cells.append(md_cell("hdr",
    "# NB4-AVES: NB4 v7 architecture with Perch → AVES\n"
    "\n"
    "Same NB4 v7 pipeline but uses AVES (wav2vec2-base, 768d) instead of Perch.\n"
    "Goal: test if AVES alone with full NB4 architecture (ProtoSSM + MLP + retrieval + prior + FCS) is competitive."))

cells.append(code_cell("install", r"""# Install onnxruntime from offline wheel (rishikeshjani/perch-onnx-for-birdclef-2026 has it)
import subprocess, sys, os, time
START = time.time()

# Find the perch-onnx dataset (has onnxruntime wheel)
ONNX_WHEEL_DIR = None
for cand in [
    "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026",
    "/kaggle/input/perch-onnx-for-birdclef-2026",
]:
    if os.path.isdir(cand):
        ONNX_WHEEL_DIR = cand
        break
if ONNX_WHEEL_DIR is None:
    for d in os.listdir("/kaggle/input"):
        sub = f"/kaggle/input/{d}"
        if os.path.isdir(sub) and any(f.endswith(".whl") for f in os.listdir(sub)):
            ONNX_WHEEL_DIR = sub; break

assert ONNX_WHEEL_DIR is not None, "onnxruntime wheel dataset not attached"
whls = [f for f in os.listdir(ONNX_WHEEL_DIR) if f.endswith(".whl")]
print(f"Wheels found in {ONNX_WHEEL_DIR}: {whls}")
if whls:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           os.path.join(ONNX_WHEEL_DIR, whls[0])])
    print(f"Installed {whls[0]}")

# transformers/librosa/soundfile are already pre-installed on Kaggle
print(f"Setup done in {time.time()-START:.0f}s")"""))

cells.append(code_cell("imports", r"""import gc, re, warnings, glob, random, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
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
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)
print(f"onnxruntime {ort.__version__}, torch {torch.__version__}")"""))

cells.append(code_cell("config", r"""# CONFIG
SR = 32_000
SR_AVES = 16_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
WINDOW_SAMPLES_AVES = SR_AVES * WINDOW_SEC
N_WINDOWS = 12

BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")
TEST_DIR = BASE / "test_soundscapes"
TRAIN_SC_DIR = BASE / "train_soundscapes"
TAXONOMY_CSV = BASE / "taxonomy.csv"
SC_LABELS_CSV = BASE / "train_soundscapes_labels.csv"

# AVES embeddings (precomputed)
AVES_EMB_DIR = None
for cand in [
    Path("/kaggle/input/datasets/maekeso/nb1f-aves-embed"),
    Path("/kaggle/input/nb1f-aves-embed"),
]:
    if cand.exists():
        AVES_EMB_DIR = cand
        break
if AVES_EMB_DIR is None:
    for p in Path("/kaggle/input").rglob("aves_sc_embeddings.npz"):
        AVES_EMB_DIR = p.parent
        break
assert AVES_EMB_DIR is not None, "AVES embeddings dataset not attached"
print(f"AVES_EMB_DIR: {AVES_EMB_DIR}")

# AVES ONNX (for test inference)
AVES_ONNX_DIR = None
for p in Path("/kaggle/input").rglob("preprocessor_config.json"):
    if (p.parent / "model.onnx").exists():
        AVES_ONNX_DIR = p.parent
        break
assert AVES_ONNX_DIR is not None, "AVES ONNX dataset not attached"
print(f"AVES_ONNX_DIR: {AVES_ONNX_DIR}")

# ProtoSSM / MLPHead config (mirror NB4 v7)
EMB_DIM = 768           # AVES (was 1536 for Perch)
D_MODEL = 128
D_STATE = 16
N_SSM_LAYERS = 2
MLP_HIDDEN = 256
DROPOUT = 0.1
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
N_SITES = 32
META_DIM = 8
SEEDS = [42, 123, 777, 2024, 9999]
USE_SWA = True
SWA_START_FRAC = 0.65
LAMBDA_KD = 0.0   # No Perch teacher in AVES NB

AGG_MODE = "mean"
TTA_SHIFTS = [-1, 0, 1]

LAMBDA_PRIOR = 0.3
PRIOR_STRENGTH_SITE = 8.0
PRIOR_STRENGTH_HOUR = 8.0
PRIOR_STRENGTH_SH = 4.0

# Retrieval (on AVES embeddings)
RETRIEVAL_K = 10
RETRIEVAL_TAU = 0.05
RETRIEVAL_ALPHA_SITE = 1.5
RETRIEVAL_ALPHA_HOUR = 1.2
LAMBDA_RETRIEVAL = 0.10
RETRIEVAL_EPS = 1e-4

USE_TA_RETRIEVAL = True
RETRIEVAL_TA_K = 20
LAMBDA_RETRIEVAL_TA = 0.05

# Blend weights (Proto vs MLP)
W_PROTO = 0.5
W_MLP = 0.5

# file_confidence_scale
FCS_TOP_K = 2
FCS_POWER = 0.4

# E19 (file-level boost)
USE_E19 = False     # NB4 v7 baseline
E19_BETA = 0.0

META_PAT = re.compile(r"_S(\d{2})_(\d{8})_(\d{2})\d{4}")

print(f"BASE={BASE}, AVES_EMB_DIR={AVES_EMB_DIR}")
print(f"AVES_ONNX_DIR={AVES_ONNX_DIR}")"""))

cells.append(code_cell("taxonomy", r"""# Taxonomy (no Perch label mapping)
taxonomy = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

# class_name array (for class-specific LAMBDA_TA, kept compatible with NB4)
class_name_arr = np.array([
    taxonomy.set_index("primary_label").loc[lbl, "class_name"]
    for lbl in PRIMARY_LABELS
])
NON_AVES_MASK = (class_name_arr != "Aves")

# Class-specific LAMBDA_TA (Aves=0.05, non-Aves=0.15) — kept from NB4 v10
LAMBDA_TA_AVES = 0.05
LAMBDA_TA_NONAVES = 0.15
LAMBDA_TA_VEC = np.full(N_CLASSES, LAMBDA_TA_AVES, dtype=np.float32)
LAMBDA_TA_VEC[NON_AVES_MASK] = LAMBDA_TA_NONAVES
print(f"Species: {N_CLASSES}, non-Aves: {NON_AVES_MASK.sum()}")"""))

cells.append(code_cell("load", r"""# Load AVES soundscape + train_audio embeddings, build labels/prior/retrieval
sc_data = np.load(AVES_EMB_DIR / "aves_sc_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)   # (~128k, 768)
sc_meta = pd.read_parquet(AVES_EMB_DIR / "aves_sc_meta.parquet")
print(f"AVES SS emb: {sc_emb.shape}, meta={sc_meta.shape}")

# No Perch logits → use zeros
sc_scores = np.zeros((sc_emb.shape[0], N_CLASSES), dtype=np.float32)

sc_labels_df = pd.read_csv(SC_LABELS_CSV)
labeled_files = set(sc_labels_df["filename"].unique())
print(f"Labeled files: {len(labeled_files)}")

label_map = {}
for _, r in sc_labels_df.iterrows():
    fn = r["filename"]
    end_sec = int(pd.Timedelta(r["end"]).total_seconds())
    rid = f"{Path(fn).stem}_{end_sec}"
    labels_str = str(r["primary_label"]).split(";")
    y = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in labels_str:
        lbl = lbl.strip()
        if lbl in label_to_idx:
            y[label_to_idx[lbl]] = 1.0
    label_map[rid] = y

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
lab_labels_files = np.zeros((len(lab_file_list), N_WINDOWS, N_CLASSES), dtype=np.float32)
for fi, fn in enumerate(lab_file_list):
    stem = Path(fn).stem
    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        rid = f"{stem}_{end_sec}"
        if rid in label_map:
            lab_labels_files[fi, wi] = label_map[rid]
print(f"Labeled: {lab_emb_files.shape[0]} files")

# Prior tables (model-independent)
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
sh_to_pi = {}; sh_n_list, sh_p_list = [], []
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
        nh = hour_n[h_i]; wh = nh / (nh + PRIOR_STRENGTH_HOUR)
        p = wh * hour_p[h_i] + (1 - wh) * p
    s_i = site_to_pi.get(int(site_id), -1)
    if s_i >= 0:
        ns = site_n[s_i]; ws = ns / (ns + PRIOR_STRENGTH_SITE)
        p = ws * site_p[s_i] + (1 - ws) * p
    sh_i = sh_to_pi.get((int(site_id), int(hour)), -1)
    if sh_i >= 0:
        nsh = sh_n[sh_i]; wsh = nsh / (nsh + PRIOR_STRENGTH_SH)
        p = wsh * sh_p[sh_i] + (1 - wsh) * p
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


lab_prior_files = np.stack([compute_prior_logit(s, h) for s, h in zip(lab_site_ids, lab_hours)]).astype(np.float32)
print(f"Prior tables built: sites={len(prior_site_ids)}, hours={len(prior_hours)}, sh={len(sh_to_pi)}")

# Retrieval pool: SC labeled (792 windows) + AVES train_audio (265k)
lab_pool_emb = lab_emb_flat.astype(np.float32)
_lab_pool_norm = np.linalg.norm(lab_pool_emb, axis=1, keepdims=True) + 1e-8
lab_pool_emb_norm = (lab_pool_emb / _lab_pool_norm).astype(np.float32)

lab_pool_labels = np.zeros((len(lab_meta), N_CLASSES), dtype=np.float32)
for _i, _row in lab_meta.iterrows():
    rid = _row["row_id"] if "row_id" in _row else f"{Path(_row['filename']).stem}_{(int(_row.get('window_idx', 0)) + 1) * WINDOW_SEC}"
    if rid in label_map:
        lab_pool_labels[_i] = label_map[rid]
lab_pool_site_ids = np.zeros(len(lab_meta), dtype=np.int64)
lab_pool_hours = np.zeros(len(lab_meta), dtype=np.int64)
for _i, _fn in enumerate(lab_meta["filename"].values):
    _s, _h = parse_meta(_fn)
    lab_pool_site_ids[_i] = _s
    lab_pool_hours[_i] = _h
print(f"SC retrieval pool: {lab_pool_emb_norm.shape[0]} windows")

# AVES train_audio retrieval pool
_ta_npz = AVES_EMB_DIR / "aves_trainaudio_embeddings.npz"
_ta_pq = AVES_EMB_DIR / "aves_trainaudio_meta.parquet"
if USE_TA_RETRIEVAL and _ta_npz.exists() and _ta_pq.exists():
    _ta_data = np.load(_ta_npz)
    _ta_meta = pd.read_parquet(_ta_pq)
    _ta_emb_raw = _ta_data["embeddings"].astype(np.float32)
    del _ta_data; gc.collect()
    ta_pool_labels = np.zeros((len(_ta_meta), N_CLASSES), dtype=np.float32)
    for _i, _lbl in enumerate(_ta_meta["primary_label"].values):
        if _lbl in label_to_idx:
            ta_pool_labels[_i, label_to_idx[_lbl]] = 1.0
    _ta_norm = np.linalg.norm(_ta_emb_raw, axis=1, keepdims=True) + 1e-8
    ta_pool_emb_norm = (_ta_emb_raw / _ta_norm).astype(np.float32)
    del _ta_emb_raw, _ta_norm, _ta_meta; gc.collect()
    print(f"AVES TA retrieval pool: {ta_pool_emb_norm.shape[0]} windows, "
          f"{int((ta_pool_labels.sum(0) > 0).sum())} active classes")
else:
    print("WARN: AVES TA retrieval disabled")
    USE_TA_RETRIEVAL = False
    ta_pool_emb_norm = None
    ta_pool_labels = None"""))

# Model: same ProtoSSM + MLPHead structure (just d_input=768)
cells.append(code_cell("model", r"""# === ProtoSSM + MLPHead (NB4 architecture, d_input=768 for AVES) ===
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state
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
        B = self.proj_B(x); C = self.proj_C(x); D_param = self.proj_D(x)
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
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
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
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_layers = nn.ModuleList([
            BiSSMBlock(d_model, d_state, dropout) for _ in range(n_ssm_layers)])
        self.attn_layers = nn.ModuleList([
            CrossAttnBlock(d_model, num_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_ssm_layers)])
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
        out = alpha * sim + (1 - alpha) * logits   # logits=zeros for AVES → effectively alpha*sim
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


class MLPHead(nn.Module):
    def __init__(self, d_input, d_hidden, n_classes, dropout=0.1, n_sites=32, meta_dim=8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, n_classes))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None, prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb)
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        h = self.mlp(x) * self.temperature
        alpha = torch.sigmoid(self.alpha)
        out = alpha * h + (1 - alpha) * logits  # logits=zeros for AVES
        if prior_logit is not None and lambda_prior > 0:
            out = out + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(out)


print("ProtoSSM + MLPHead defined (d_input=768 for AVES).")"""))

cells.append(code_cell("train", r"""# === Train ProtoSSM + MLPHead × 5 seeds on AVES embeddings ===
emb_flat = torch.tensor(lab_emb_flat, dtype=torch.float32)
lab_flat = torch.zeros(len(lab_meta), N_CLASSES, dtype=torch.float32)
for i, row in lab_meta.iterrows():
    rid = row["row_id"] if "row_id" in row else f"{Path(row['filename']).stem}_{(int(row.get('window_idx', 0)) + 1) * WINDOW_SEC}"
    if rid in label_map:
        lab_flat[i] = torch.tensor(label_map[rid])

train_emb = torch.tensor(lab_emb_files, dtype=torch.float32)
train_logits = torch.tensor(lab_scores_files, dtype=torch.float32)  # zeros
train_labels = torch.tensor(lab_labels_files, dtype=torch.float32)
train_site = torch.tensor(lab_site_ids, dtype=torch.long)
train_hour = torch.tensor(lab_hours, dtype=torch.long)
train_prior = torch.tensor(lab_prior_files, dtype=torch.float32)

pos_counts = train_labels.sum(dim=(0, 1)).clamp(min=1)
neg_counts = train_labels.shape[0] * train_labels.shape[1] - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(max=30.0)


def _train_one(model_factory, seed):
    seed_everything(seed)
    model = model_factory().to(DEVICE)
    if hasattr(model, "init_prototypes"):
        model.init_prototypes(emb_flat, lab_flat)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    swa_model = AveragedModel(model) if USE_SWA else None
    swa_start = int(N_EPOCHS * SWA_START_FRAC); swa_n = 0
    best_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(N_EPOCHS):
        model.train()
        out = model(train_emb, train_logits, site_ids=train_site, hours=train_hour,
                    prior_logit=train_prior, lambda_prior=LAMBDA_PRIOR)
        bce = F.binary_cross_entropy(out, train_labels, reduction="none")
        loss = (bce * pos_weight.unsqueeze(0).unsqueeze(0)).mean()
        # No KD (LAMBDA_KD=0 for AVES NB)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if USE_SWA and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1
    if USE_SWA and swa_n >= 1:
        model.load_state_dict(swa_model.module.state_dict())
    else:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


proto_models = []
t0 = time.time()
for seed in SEEDS:
    factory = lambda: ProtoSSM(d_input=EMB_DIM, d_model=D_MODEL, d_state=D_STATE,
                                n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
                                n_windows=N_WINDOWS, dropout=DROPOUT,
                                n_sites=N_SITES, meta_dim=META_DIM)
    t_s = time.time()
    m, bl = _train_one(factory, seed)
    proto_models.append(m)
    print(f"  ProtoSSM[seed {seed}] best_loss={bl:.4f}, {time.time()-t_s:.0f}s")
print(f"ProtoSSM ensemble: {len(proto_models)} models in {time.time()-t0:.0f}s")

mlp_models = []
t0 = time.time()
for seed in SEEDS:
    factory = lambda: MLPHead(d_input=EMB_DIM, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                               dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM)
    t_s = time.time()
    m, bl = _train_one(factory, seed)
    mlp_models.append(m)
    print(f"  MLPHead[seed {seed}] best_loss={bl:.4f}, {time.time()-t_s:.0f}s")
print(f"MLPHead ensemble: {len(mlp_models)} models in {time.time()-t0:.0f}s")"""))

cells.append(code_cell("aves-onnx-test", r"""# === AVES ONNX session for test inference ===
from transformers import AutoFeatureExtractor
aves_fe = AutoFeatureExtractor.from_pretrained(str(AVES_ONNX_DIR))
AVES_SR = aves_fe.sampling_rate
print(f"AVES SR={AVES_SR}")

_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
aves_sess = ort.InferenceSession(str(AVES_ONNX_DIR / "model.onnx"),
                                  sess_options=_so, providers=["CPUExecutionProvider"])
AVES_INPUT_NAME = aves_sess.get_inputs()[0].name


def read_audio(path, target_samples=None):
    y, sr0 = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    if target_samples is not None:
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
    return y.astype(np.float32)


def aves_infer_file(raw_60s):
    # 60s @ 32k → 60s @ 16k → 12 windows of 5s
    raw_aves = librosa.resample(raw_60s, orig_sr=SR, target_sr=AVES_SR).astype(np.float32)
    target = AVES_SR * 60
    if len(raw_aves) < target:
        raw_aves = np.pad(raw_aves, (0, target - len(raw_aves)))
    elif len(raw_aves) > target:
        raw_aves = raw_aves[:target]
    windows = raw_aves.reshape(N_WINDOWS, WINDOW_SAMPLES_AVES)
    inputs = aves_fe([windows[i] for i in range(N_WINDOWS)],
                     sampling_rate=AVES_SR, return_tensors="np", padding=True)
    out = aves_sess.run(None, {AVES_INPUT_NAME: inputs["input_values"].astype(np.float32)})[0]
    emb = out.mean(axis=1).astype(np.float32)   # (12, 768)
    return emb


# Retrieval helpers (on AVES embeddings)
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
                               K=RETRIEVAL_TA_K, tau=RETRIEVAL_TAU, eps=RETRIEVAL_EPS):
    e_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    sim = e_norm @ ta_pool_emb_norm.T
    K_eff = min(K, sim.shape[1])
    topk_idx = np.argpartition(sim, -K_eff, axis=1)[:, -K_eff:]
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
    topk_label = ta_pool_labels[topk_idx]
    sm = topk_sim - topk_sim.max(axis=1, keepdims=True)
    sm = np.exp(sm / max(tau, 1e-6))
    sm = sm / (sm.sum(axis=1, keepdims=True) + 1e-8)
    p = (sm[..., None] * topk_label).sum(axis=1)
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


print("AVES ONNX + retrieval helpers ready.")"""))

cells.append(code_cell("inference", r"""# === Inference (NB4 v7 logic, AVES embeddings) ===
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
    return torch.stack(ens_probs, dim=0).mean(dim=0)


for fi, fpath in enumerate(test_files):
    stem = Path(fpath).stem
    raw_60s = read_audio(fpath, target_samples=SR * 60)

    # AVES embedding for 12 windows
    emb = aves_infer_file(raw_60s)              # (12, 768)
    scores = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)  # no Perch logits

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
        p_proto = _ensemble_one(proto_models, emb_t, scores_t, site_t, hour_t, prior_t)
        p_mlp = _ensemble_one(mlp_models, emb_t, scores_t, site_t, hour_t, prior_t)
        p_proto_c = p_proto.clamp(min=1e-7, max=1 - 1e-7)
        p_mlp_c = p_mlp.clamp(min=1e-7, max=1 - 1e-7)
        l_proto = torch.log(p_proto_c) - torch.log1p(-p_proto_c)
        l_mlp = torch.log(p_mlp_c) - torch.log1p(-p_mlp_c)
        l_blend = W_PROTO * l_proto + W_MLP * l_mlp

        if LAMBDA_RETRIEVAL > 0:
            ret_logit = compute_retrieval_logit(emb, site_id, hour)
            ret_logit_t = torch.tensor(ret_logit, dtype=torch.float32).unsqueeze(0)
            l_blend = l_blend + LAMBDA_RETRIEVAL * ret_logit_t

        if USE_TA_RETRIEVAL and ta_pool_emb_norm is not None:
            ta_ret_logit = compute_retrieval_ta_logit(emb)
            ta_ret_logit_t = torch.tensor(ta_ret_logit, dtype=torch.float32).unsqueeze(0)
            lam_t = torch.tensor(LAMBDA_TA_VEC, dtype=torch.float32)
            l_blend = l_blend + lam_t * ta_ret_logit_t

        agg = torch.sigmoid(l_blend)
        probs = agg.squeeze(0).numpy()

    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        all_row_ids.append(f"{stem}_{end_sec}")
        all_probs.append(probs[wi])

    if (fi + 1) % 20 == 0 or fi == len(test_files) - 1:
        elapsed = time.time() - t0
        print(f"  [{fi+1}/{len(test_files)}] {elapsed:.0f}s")

print(f"Inference done: {len(all_row_ids)} predictions in {time.time()-t0:.0f}s")"""))

cells.append(code_cell("submission", r"""# Build submission with file_confidence_scale
preds_array = np.stack(all_probs)
_n, _c = preds_array.shape
_view = preds_array.reshape(-1, N_WINDOWS, _c)
_sorted = np.sort(_view, axis=1)
_topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
_scale = np.power(_topk_mean, FCS_POWER)
preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)

submission = pd.DataFrame(preds_array, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)
extra = our_ids - expected_ids
if extra:
    submission = submission[submission["row_id"].isin(expected_ids)]
submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

total = time.time() - START
print(f"\nSubmission: {submission.shape}, total {total:.0f}s ({total/60:.1f} min)")
print(f"Mean: {submission[PRIMARY_LABELS].values.mean():.6f}, Max: {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head())"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb4_aves.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
