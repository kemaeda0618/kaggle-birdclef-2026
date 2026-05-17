"""Generate exp010 NB3: ProtoSSM Submission Notebook.

Train ProtoSSM on Perch embeddings (from NB1) + submit.
CPU only, 90 min limit.

Pipeline:
1. Load pre-computed embeddings (train_soundscapes)
2. Train ProtoSSM on labeled soundscape windows (66 files)
3. Run Perch ONNX on test_soundscapes (CPU)
4. ProtoSSM inference -> submission.csv

Usage: python _gen_nb3_protossm.py
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
    "# exp010 NB3: ProtoSSM on Perch Embeddings\n"
    "\n"
    "CPU submission notebook (90 min limit).\n"
    "1. Load pre-computed Perch embeddings from NB1\n"
    "2. Train ProtoSSM on labeled soundscapes (66 files)\n"
    "3. Perch ONNX inference on test_soundscapes (CPU)\n"
    "4. ProtoSSM prediction -> submission.csv"))

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
# CONFIG
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12

BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")

EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")
TEST_DIR = BASE / "test_soundscapes"
TRAIN_SC_DIR = BASE / "train_soundscapes"
TAXONOMY_CSV = BASE / "taxonomy.csv"
SC_LABELS_CSV = BASE / "train_soundscapes_labels.csv"

# ONNX_DS is set in install cell
LABELS_CSV = os.path.join(ONNX_DS, "labels.csv")
ONNX_MODEL = os.path.join(ONNX_DS, "perch_v2.onnx")

# ProtoSSM config
D_MODEL = 128
D_STATE = 16
N_SSM_LAYERS = 2
DROPOUT = 0.1
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
BATCH_ONNX = 48

# Metadata embedding (site/hour parsed from filename)
N_SITES = 32  # S01..S23 observed, use 32 for safety
META_DIM = 8

# Multi-seed ensemble: train one model per seed, average probabilities at inference
SEEDS = [42, 123, 777, 2024, 9999]

# Ensemble aggregation across (TTA x seeds): "mean" or "max" or "blend"
# - mean: variance reduction, stable baseline
# - max: preserves peak signal per class (rare species detection)
# - blend: BLEND_ALPHA * mean + (1 - BLEND_ALPHA) * max
AGG_MODE = "mean"
BLEND_ALPHA = 0.5

# Test-time augmentation: shift emb sequence by these amounts (0 = no shift)
# v27 5-shift = 0.921 (同値), v28 MLP probe = -0.050 大失敗 -> v26 base に revert
# v29: Hierarchical Site-Conditioned KNN Retrieval (inference-only post-hoc logit blend)
TTA_SHIFTS = [-1, 0, 1]

# Prior tables (site/hour co-occurrence) blending weight
LAMBDA_PRIOR = 0.3
PRIOR_STRENGTH_SITE = 8.0
PRIOR_STRENGTH_HOUR = 8.0
PRIOR_STRENGTH_SH = 4.0

# v29: Inference-time retrieval (Hierarchical Site-Conditioned KNN)
# Pool = 792 labeled soundscape windows (Perch v2 emb 1536-dim)
# Boost weights per tier: same-site x1.5, same-hour x1.2 (cumulative, multiplicative)
# Softmax over top-K with temperature, distance-weighted label avg, blended in logit space
RETRIEVAL_K = 10
RETRIEVAL_TAU = 0.05               # softmax temperature on cosine sim (sharpness)
RETRIEVAL_ALPHA_SITE = 1.5         # site-tier weight multiplier
RETRIEVAL_ALPHA_HOUR = 1.2         # hour-tier weight multiplier
LAMBDA_RETRIEVAL = 0.10            # post-hoc logit blend weight (low to be safe)
RETRIEVAL_EPS = 1e-4               # logit clipping

# Regex to parse site and hour from filename: BC2026_{Train,Test}_NNNN_SXX_YYYYMMDD_HHMMSS.ogg
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

print(f"Species: {N_CLASSES}, Mapped: {MAPPED_MASK.sum()}, Proxies: {len(proxy_map)}")"""
cells.append(code_cell("taxonomy", TAXONOMY))

# ── Load Embeddings & Labels ──
LOAD = """\
# LOAD SOUNDSCAPE EMBEDDINGS + LABELS
sc_data = np.load(EMB_DIR / "soundscape_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)
sc_scores = sc_data["scores"].astype(np.float32)
sc_meta = pd.read_parquet(EMB_DIR / "soundscape_meta.parquet")

print(f"Soundscapes: {sc_emb.shape[0]} windows, {sc_meta['filename'].nunique()} files")

# Parse soundscape labels
sc_labels_df = pd.read_csv(SC_LABELS_CSV)
labeled_files = set(sc_labels_df["filename"].unique())
print(f"Labeled files: {len(labeled_files)}")

# Build per-window label matrix for labeled files
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

# Split labeled vs unlabeled
is_labeled = sc_meta["filename"].isin(labeled_files).values

# Reshape to file-level (n_files, N_WINDOWS, dim)
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

# Labeled files
lab_meta = sc_meta[is_labeled].reset_index(drop=True)
lab_emb_flat = sc_emb[is_labeled]
lab_scores_flat = sc_scores[is_labeled]

lab_emb_files, lab_file_list = reshape_to_files(lab_emb_flat, lab_meta)
lab_scores_files, _ = reshape_to_files(lab_scores_flat, lab_meta)

# Parse site/hour metadata from filename
def parse_meta(fname):
    m = META_PAT.search(fname)
    if m is None:
        return 0, 0
    return int(m.group(1)), int(m.group(3))

lab_site_ids = np.array([parse_meta(fn)[0] for fn in lab_file_list], dtype=np.int64)
lab_hours = np.array([parse_meta(fn)[1] for fn in lab_file_list], dtype=np.int64)
print(f"Labeled metadata: unique sites={np.unique(lab_site_ids).tolist()}, unique hours={np.unique(lab_hours).tolist()}")

# Build file-level labels (N_WINDOWS per file)
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

# PRIOR TABLES (site/hour co-occurrence from labeled files)
# Per-file labels = any positive across 12 windows
file_labels = (lab_labels_files.sum(axis=1) > 0).astype(np.float32)  # (n_files, n_classes)
global_p = file_labels.mean(axis=0).astype(np.float32)

# Site table
prior_site_ids = sorted(set(int(s) for s in lab_site_ids))
site_to_pi = {s: i for i, s in enumerate(prior_site_ids)}
site_n = np.zeros(len(prior_site_ids), dtype=np.float32)
site_p = np.zeros((len(prior_site_ids), N_CLASSES), dtype=np.float32)
for s in prior_site_ids:
    m = (lab_site_ids == s)
    site_n[site_to_pi[s]] = m.sum()
    site_p[site_to_pi[s]] = file_labels[m].mean(axis=0)

# Hour table
prior_hours = sorted(set(int(h) for h in lab_hours))
hour_to_pi = {h: i for i, h in enumerate(prior_hours)}
hour_n = np.zeros(len(prior_hours), dtype=np.float32)
hour_p = np.zeros((len(prior_hours), N_CLASSES), dtype=np.float32)
for h in prior_hours:
    m = (lab_hours == h)
    hour_n[hour_to_pi[h]] = m.sum()
    hour_p[hour_to_pi[h]] = file_labels[m].mean(axis=0)

# Site-hour table
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
    # Shrinkage: start from global prior, layer in hour, site, site-hour
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

# Precompute prior for labeled files (file-level, broadcast to windows at train time)
lab_prior_files = np.stack(
    [compute_prior_logit(s, h) for s, h in zip(lab_site_ids, lab_hours)]
).astype(np.float32)
print(f"Prior tables: sites={len(prior_site_ids)}, hours={len(prior_hours)}, "
      f"sh={len(sh_to_pi)}; lab_prior shape={lab_prior_files.shape}")

# v29: RETRIEVAL POOL (per-window labeled embeddings + metadata)
# Pool = labeled soundscape windows. Used at inference time only.
lab_pool_emb = lab_emb_flat.astype(np.float32)  # (N_pool, 1536)
_lab_pool_norm = np.linalg.norm(lab_pool_emb, axis=1, keepdims=True) + 1e-8
lab_pool_emb_norm = (lab_pool_emb / _lab_pool_norm).astype(np.float32)

lab_pool_labels = np.zeros((len(lab_meta), N_CLASSES), dtype=np.float32)
for _i, _rid in enumerate(lab_meta["row_id"].values):
    if _rid in label_map:
        lab_pool_labels[_i] = label_map[_rid]

# Per-window site/hour (parsed from filename)
lab_pool_site_ids = np.zeros(len(lab_meta), dtype=np.int64)
lab_pool_hours = np.zeros(len(lab_meta), dtype=np.int64)
for _i, _fn in enumerate(lab_meta["filename"].values):
    _s, _h = parse_meta(_fn)
    lab_pool_site_ids[_i] = _s
    lab_pool_hours[_i] = _h

print(f"Retrieval pool: {lab_pool_emb_norm.shape[0]} windows, "
      f"{int((lab_pool_labels.sum(axis=0) > 0).sum())} active classes, "
      f"sites={len(set(lab_pool_site_ids.tolist()))}, hours={len(set(lab_pool_hours.tolist()))}")"""
cells.append(code_cell("load", LOAD))

# ── ProtoSSM Model ──
MODEL = """\
# PROTOSSM MODEL
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


# v26: 2-head self-attention over windows, inserted after each BiSSM layer
class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
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
        # Metadata embedding (site + hour)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_layers = nn.ModuleList([
            BiSSMBlock(d_model, d_state, dropout) for _ in range(n_ssm_layers)
        ])
        # v26: cross-attn after each BiSSM layer
        self.attn_layers = nn.ModuleList([
            CrossAttnBlock(d_model, num_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_ssm_layers)
        ])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(n_classes))

        # Gated fusion with Perch logits
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None, prior_logit=None, lambda_prior=0.0):
        # emb: (B, T, d_input), logits: (B, T, n_classes)
        # site_ids, hours: (B,) long tensors; optional (zero meta if None)
        # prior_logit: (B, n_classes) file-level prior logits; broadcast to windows
        x = self.input_proj(emb) + self.pos_emb[:, :emb.shape[1]]
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))  # (B, d_model)
            x = x + meta.unsqueeze(1)
        for ssm_layer, attn_layer in zip(self.ssm_layers, self.attn_layers):
            x = ssm_layer(x)
            x = attn_layer(x)

        # Prototype similarity
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("btd,cd->btc", x_norm, p_norm) * self.temperature + self.bias

        # Gated fusion
        alpha = torch.sigmoid(self.alpha)
        out = alpha * sim + (1 - alpha) * logits
        if prior_logit is not None and lambda_prior > 0:
            out = out + lambda_prior * prior_logit.unsqueeze(1)  # broadcast to T
        return torch.sigmoid(out)

    def init_prototypes(self, emb_flat, labels_flat):
        # Initialize prototypes from labeled data mean embeddings
        with torch.no_grad():
            x = self.input_proj(emb_flat)
            for ci in range(self.prototypes.shape[0]):
                mask = labels_flat[:, ci] > 0.5
                if mask.sum() > 0:
                    self.prototypes[ci] = x[mask].mean(0)


print("ProtoSSM model defined.")"""
cells.append(code_cell("model", MODEL))

# ── Train ProtoSSM ──
TRAIN = """\
# TRAIN PROTOSSM (multi-seed ensemble)
# Prepare data tensors (deterministic, shared across seeds)
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

# Class weights (inverse frequency)
pos_counts = train_labels.sum(dim=(0, 1)).clamp(min=1)
neg_counts = train_labels.shape[0] * train_labels.shape[1] - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(max=30.0)

models = []
t0_all = time.time()
for si, seed in enumerate(SEEDS):
    seed_everything(seed)
    model = ProtoSSM(
        d_input=1536, d_model=D_MODEL, d_state=D_STATE,
        n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
        n_windows=N_WINDOWS, dropout=DROPOUT,
        n_sites=N_SITES, meta_dim=META_DIM,
    ).to(DEVICE)
    model.init_prototypes(emb_flat, lab_flat)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    if si == 0:
        print(f"Training ProtoSSM (x{len(SEEDS)} seeds): {sum(p.numel() for p in model.parameters())} params/model")
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        model.train()
        out = model(train_emb, train_logits, site_ids=train_site, hours=train_hour,
                    prior_logit=train_prior, lambda_prior=LAMBDA_PRIOR)

        loss = F.binary_cross_entropy(out, train_labels, reduction="none")
        loss = (loss * pos_weight.unsqueeze(0).unsqueeze(0)).mean()

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

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    models.append(model)
    print(f"  [seed {seed}] done in {time.time()-t0:.1f}s, best_loss={best_loss:.4f}, epochs={epoch+1}")

print(f"R0 ensemble: {len(models)} models trained in {time.time()-t0_all:.1f}s")"""
cells.append(code_cell("train", TRAIN))

# ── ONNX for Test ──
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


# v29: Hierarchical Site-Conditioned KNN retrieval logit
def compute_retrieval_logit(test_emb, site_id, hour,
                            K=RETRIEVAL_K, tau=RETRIEVAL_TAU,
                            alpha_site=RETRIEVAL_ALPHA_SITE,
                            alpha_hour=RETRIEVAL_ALPHA_HOUR,
                            eps=RETRIEVAL_EPS):
    # test_emb: (T, 1536) np.float32
    # Returns: (T, N_CLASSES) np.float32 logit
    e_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    sim = e_norm @ lab_pool_emb_norm.T  # (T, N_pool)

    # Tier weights (multiplicative): same-site x alpha_site, same-hour x alpha_hour
    w = np.ones(lab_pool_emb_norm.shape[0], dtype=np.float32)
    w[lab_pool_site_ids == site_id] *= alpha_site
    w[lab_pool_hours == hour] *= alpha_hour
    w_sim = sim * w[None, :]

    # Top-K (use argpartition for speed)
    K_eff = min(K, w_sim.shape[1])
    topk_idx = np.argpartition(w_sim, -K_eff, axis=1)[:, -K_eff:]  # (T, K)
    topk_sim = np.take_along_axis(w_sim, topk_idx, axis=1)  # (T, K)
    topk_label = lab_pool_labels[topk_idx]  # (T, K, N_CLASSES)

    # Softmax over top-K with temperature
    sm = topk_sim - topk_sim.max(axis=1, keepdims=True)
    sm = np.exp(sm / max(tau, 1e-6))
    sm = sm / (sm.sum(axis=1, keepdims=True) + 1e-8)

    p = (sm[..., None] * topk_label).sum(axis=1)  # (T, N_CLASSES)
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


print("Perch ONNX (CPU) + retrieval helper ready.")"""
cells.append(code_cell("onnx-test", ONNX_TEST))

# ── Inference ──
INFERENCE = """\
# INFERENCE ON TEST SOUNDSCAPES
test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))
if len(test_files) == 0:
    print("No test files, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))[:8]

print(f"Test files: {len(test_files)}")

all_row_ids = []
all_probs = []

t0 = time.time()
for _m in models:
    _m.eval()

# Parallel audio prefetch: ThreadPoolExecutor reads next files while ONNX runs (yukiZ 686457)
from concurrent.futures import ThreadPoolExecutor

def _load_windows(fp):
    y = read_audio(fp, target_samples=SR * 60)
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES)

PREFETCH = 4
executor = ThreadPoolExecutor(max_workers=4)
pending = {}
for _i in range(min(PREFETCH, len(test_files))):
    pending[_i] = executor.submit(_load_windows, test_files[_i])

for fi, fpath in enumerate(test_files):
    stem = Path(fpath).stem

    # Kick off next prefetch
    _ni = fi + PREFETCH
    if _ni < len(test_files):
        pending[_ni] = executor.submit(_load_windows, test_files[_ni])

    # Drain current
    windows = pending.pop(fi).result()

    # Perch ONNX
    emb, scores = infer_perch_cpu(windows)

    # Parse site/hour from filename (fallback 0,0 if pattern missing - e.g. hidden test)
    _m = META_PAT.search(fpath)
    site_id = int(_m.group(1)) if _m else 0
    hour = int(_m.group(3)) if _m else 0
    site_t = torch.tensor([site_id], dtype=torch.long)
    hour_t = torch.tensor([hour], dtype=torch.long)
    prior_vec = compute_prior_logit(site_id, hour)
    prior_t = torch.tensor(prior_vec, dtype=torch.float32).unsqueeze(0)  # (1, n_classes)

    # TTA x Multi-seed ensemble: shift emb by TTA_SHIFTS, average across models & shifts
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)  # (1, T, D)
        scores_t = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        ens_probs = []
        for s in TTA_SHIFTS:
            if s == 0:
                e_shift, sc_shift = emb_t, scores_t
            else:
                e_shift = torch.roll(emb_t, shifts=s, dims=1)
                sc_shift = torch.roll(scores_t, shifts=s, dims=1)
            for m in models:
                p = m(e_shift, sc_shift, site_ids=site_t, hours=hour_t,
                      prior_logit=prior_t, lambda_prior=LAMBDA_PRIOR)
                # Undo the roll so predictions align with original window positions
                if s != 0:
                    p = torch.roll(p, shifts=-s, dims=1)
                ens_probs.append(p)
        stacked = torch.stack(ens_probs, dim=0)  # (n_tta*n_models, 1, N_WINDOWS, N_CLASSES)
        if AGG_MODE == "max":
            agg = stacked.max(dim=0).values
        elif AGG_MODE == "blend":
            mean_p = stacked.mean(dim=0)
            max_p = stacked.max(dim=0).values
            agg = BLEND_ALPHA * mean_p + (1 - BLEND_ALPHA) * max_p
        else:
            agg = stacked.mean(dim=0)  # (1, N_WINDOWS, N_CLASSES)

        # v29: Hierarchical Site-Conditioned KNN retrieval blend (post-hoc, logit space)
        if LAMBDA_RETRIEVAL > 0:
            ret_logit = compute_retrieval_logit(emb, site_id, hour)  # (T, N_CLASSES)
            ret_logit_t = torch.tensor(ret_logit, dtype=torch.float32).unsqueeze(0)
            agg_clamped = agg.clamp(min=1e-7, max=1 - 1e-7)
            agg_logit = torch.log(agg_clamped) - torch.log1p(-agg_clamped)
            agg_logit = agg_logit + LAMBDA_RETRIEVAL * ret_logit_t
            agg = torch.sigmoid(agg_logit)
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
# BUILD SUBMISSION
preds_array = np.stack(all_probs)

# v20: file-level confidence scaling (top-K mean ^ power)
# Suppresses files that lack a consistent signal across windows.
# all_probs is appended file-by-file with 12 windows contiguous, so reshape works.
FCS_TOP_K = 2
FCS_POWER = 0.4
_n, _c = preds_array.shape
assert _n % N_WINDOWS == 0, f"preds_array rows {_n} not divisible by {N_WINDOWS}"
_view = preds_array.reshape(-1, N_WINDOWS, _c)
_sorted = np.sort(_view, axis=1)
_topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
_scale = np.power(_topk_mean, FCS_POWER)
preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)
print(f"file_confidence_scale applied (top_k={FCS_TOP_K}, power={FCS_POWER})")
print(f"  scale stats: min={_scale.min():.4f} max={_scale.max():.4f} mean={_scale.mean():.4f}")

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

out_path = HERE / "nb3_protossm.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
