"""Generate exp011 self-distillation + SED training notebook.

This notebook (`nb_pseudo_and_train.ipynb`) runs on Kaggle T4x2 GPU and
performs three resumable phases in a single execution:

Phase 1 - Pseudo-label generation
  1. Load Perch embeddings cached in `maekeso/birdclef2026-exp010-nb1-embedding`
     (`soundscape_embeddings.npz` + `soundscape_meta.parquet`).
  2. Train LightProtoSSM (gate-fake008 architecture) on the 66 labeled
     soundscape files from `train_soundscapes_labels.csv` (40 epochs, OneCycleLR
     + SWA).
  3. Run trained ProtoSSM inference on every cached embedding to obtain
     `proto_scores.npy` of shape (10658, 12, 234).
  4. Run Tucker SED 5-fold ONNX inference on every train_soundscape audio file,
     prefer GPU CUDA execution provider with CPU fallback. Output
     `tucker_scores.npy` of shape (10658, 12, 234).
  5. Rank-blend ProtoSSM 60% + Tucker SED 40% per (file, window) -> save
     `pseudo_labels.npy`.

Phase 2 - SED training (resumable per-fold per-epoch)
  - EfficientNet-B0 then EfficientNet-B1 backbones via timm.
  - Stratified 3-fold by primary_label, group by author.
  - train_audio: hard labels (primary + secondary), random 5s crop.
  - train_soundscapes: soft pseudo-labels per 5s window.
  - Mel cache (uint8 quantized) read from
    `maekeso/birdclef2026-mel-cache-train-audio-256` and
    `maekeso/birdclef2026-mel-cache-train-sc-256`.
  - SpecAugment (freq=30, time=40), spec mixup (alpha=0.5).
  - AdamW lr=5e-4 wd=1e-2, CosineAnnealingLR, 25 epochs/fold, AMP.
  - Atomic checkpoints: temp -> rename. Best checkpoint by val macro AUC.
  - `done.flag` per fold marks completion.

Phase 3 - ONNX export
  - Each `best.pt` -> ONNX with input (B, 1, 256, 313).

Resumability mechanism
  At start, copy any prior outputs from
  `/kaggle/input/notebooks/maekeso/birdclef2026-pseudo-and-train/` (kernel_source
  of the previous version) into `/kaggle/working/`. Each phase skips if its
  outputs already exist.

Graceful early stop
  WALL_START + SOFT_LIMIT_SEC = 11h30m. Each epoch checks remaining time and
  exits cleanly if < 30 min remain.

Run: `python _gen_nb_pseudo_and_train.py` writes `nb_pseudo_and_train.ipynb`.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, lines):
    if isinstance(lines, str):
        lines = lines.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, lines):
    if isinstance(lines, str):
        lines = lines.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

# ===================== Header =====================
cells.append(md_cell("hdr", r"""# exp011: Self-distilled pseudo-labels + EfficientNet-B0/B1 SED training

This notebook runs end-to-end on a single Kaggle T4x2 session and is
**resumable**: rerun the same notebook (mounting the previous version output as
a `kernel_source`) and it picks up wherever it left off.

## Phases
1. **Pseudo-label generation** (~1.5 hr)
   - Train LightProtoSSM on 66 labeled soundscape files (40 epochs).
   - Run ProtoSSM inference on cached embeddings (10,658 files).
   - Run Tucker SED 5-fold ONNX inference on raw audio.
   - Rank blend 60% / 40% -> `pseudo_labels.npy` (10658, 12, 234).
2. **SED training** (~9 hr)
   - EfficientNet-B0 (3 folds) then EfficientNet-B1 (3 folds).
   - 25 epochs each, AdamW + CosineAnnealing, BCE clip + frame.
3. **ONNX export** (~10 min) — `b0_fold{f}.onnx`, `b1_fold{f}.onnx`.

## Inputs
- competition: `birdclef-2026`
- dataset: `tuckerarrants/bc2026-distilled-sed-public`
- kernel_source: `maekeso/birdclef2026-exp010-nb1-embedding`
- dataset: `maekeso/birdclef2026-mel-cache-train-audio-256`
- dataset: `maekeso/birdclef2026-mel-cache-train-sc-256`
- (resume) kernel_source: `maekeso/birdclef2026-pseudo-and-train`

## Outputs (in `/kaggle/working`)
- `proto_scores.npy`, `tucker_scores.npy`, `pseudo_labels.npy`
- `checkpoints/{b0,b1}_fold{f}_best.pt`
- `onnx/{b0,b1}_fold{f}.onnx`
"""))

# ===================== Install =====================
cells.append(code_cell("install", r"""!pip install -q timm onnx onnxruntime-gpu librosa scipy
print("install done")"""))

# ===================== Imports & global config =====================
cells.append(code_cell("imports", r"""import os, sys, gc, ast, glob, time, json, math, random, hashlib, shutil, re, warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

SEED = 42
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# Audio constants (shared)
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC          # 160_000
N_WINDOWS      = 12                       # 12 * 5s = 60s file
FILE_SAMPLES   = N_WINDOWS * WINDOW_SAMPLES

# Mel cache spec (matches maekeso/birdclef2026-mel-cache-* datasets)
N_MELS    = 256
N_FFT     = 2048
HOP       = 512
FMIN      = 20
FMAX      = 16_000
TOP_DB    = 80
DB_MIN    = -80.0
DB_MAX    = 20.0
DB_RANGE  = DB_MAX - DB_MIN               # 100.0
WINDOW_FRAMES = WINDOW_SAMPLES // HOP + 1 # 313 frames per 5s window
print(f"WINDOW_FRAMES={WINDOW_FRAMES}")"""))

# ===================== Resume / time tracking / IO helpers =====================
cells.append(code_cell("resume", r"""# ---- Wall clock & time helpers ----
WALL_START      = time.time()
SOFT_LIMIT_SEC  = 11 * 3600 + 30 * 60     # 11h30m
EXIT_MARGIN_SEC = 30 * 60                 # leave >=30min if quitting

def time_left():
    return SOFT_LIMIT_SEC - (time.time() - WALL_START)

def time_low():
    return time_left() < EXIT_MARGIN_SEC

def fmt_dur(s):
    s = int(s)
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---- IO helpers ----
WORK_DIR = Path("/kaggle/working"); WORK_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = WORK_DIR / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR = WORK_DIR / "onnx"; ONNX_DIR.mkdir(parents=True, exist_ok=True)

def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))

def atomic_save_npy(arr, path):
    path = Path(path)
    # np.save auto-appends .npy if path does not end in .npy, so tmp must end in .npy
    tmp = path.with_suffix(".tmp.npy")
    np.save(str(tmp), arr)
    os.replace(str(tmp), str(path))

# ---- Resume from previous version's output if mounted as kernel_source ----
PREV_KERNEL = Path("/kaggle/input/notebooks/maekeso/birdclef2026-pseudo-and-train")
if PREV_KERNEL.exists():
    print(f"PREV_KERNEL found: {PREV_KERNEL}")
    n_copied = 0
    for src in PREV_KERNEL.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(PREV_KERNEL)
        # Skip notebook artefacts
        if rel.name.startswith("__") or rel.suffix in (".log", ".html"):
            continue
        if "kernel-metadata.json" in str(rel):
            continue
        dst = WORK_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
            n_copied += 1
        except Exception as e:
            print(f"  copy err {rel}: {e}")
    print(f"  copied {n_copied} files from prev kernel")
else:
    print("PREV_KERNEL not mounted (first run)")

print(f"WORK_DIR contents: {sorted(p.name for p in WORK_DIR.iterdir())[:30]}")
print(f"CKPT_DIR contents: {sorted(p.name for p in CKPT_DIR.iterdir())[:50]}")

# ---- One-time migration: rank-blend pseudo (broken) -> prob-blend ----
# Old pseudo_labels.npy was per-class rank-normalized which destroyed absolute
# probabilities. Detect old training artefacts and clear them so Phase 2 retrains
# on the new pseudo_labels_prob.npy.
PSEUDO_MIGRATION_FLAG = WORK_DIR / "pseudo_prob_blend_v2.flag"
if not PSEUDO_MIGRATION_FLAG.exists():
    old_pseudo = WORK_DIR / "pseudo_labels.npy"
    has_old_ckpt = any((CKPT_DIR / f"b0_fold{f}_log.json").exists() for f in range(3)) \
                or any((CKPT_DIR / f"b1_fold{f}_log.json").exists() for f in range(3))
    if old_pseudo.exists() or has_old_ckpt:
        print("MIGRATION: removing old rank-blend artefacts and Phase 2 checkpoints")
        if old_pseudo.exists():
            old_pseudo.unlink()
            print(f"  removed {old_pseudo}")
        cleared = 0
        for p in list(CKPT_DIR.rglob("*")):
            if p.is_file():
                p.unlink()
                cleared += 1
        print(f"  cleared {cleared} checkpoint files")
    PSEUDO_MIGRATION_FLAG.write_text("v2 prob blend - rank blend deprecated")
else:
    print("Pseudo prob-blend migration already done")"""))

# ===================== Paths =====================
cells.append(code_cell("paths", r"""# ---- Competition data ----
COMP_DIR = None
for cand in [Path("/kaggle/input/competitions/birdclef-2026"),
             Path("/kaggle/input/birdclef-2026")]:
    if cand.exists():
        COMP_DIR = cand; break
assert COMP_DIR is not None, "birdclef-2026 not mounted"
print(f"COMP_DIR: {COMP_DIR}")

TRAIN_CSV       = COMP_DIR / "train.csv"
TAXONOMY_CSV    = COMP_DIR / "taxonomy.csv"
SAMPLE_SUB_CSV  = COMP_DIR / "sample_submission.csv"
SC_LABELS_CSV   = COMP_DIR / "train_soundscapes_labels.csv"
TRAIN_AUDIO_DIR = COMP_DIR / "train_audio"
TRAIN_SC_DIR    = COMP_DIR / "train_soundscapes"

sample_sub = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
PRIMARY_LABELS = list(sample_sub.columns[1:])
N_CLASSES      = len(PRIMARY_LABELS)
LABEL_TO_IDX   = {c: i for i, c in enumerate(PRIMARY_LABELS)}
print(f"N_CLASSES={N_CLASSES}")

# ---- Perch embeddings (kernel_source from exp010 NB1) ----
EMB_DIR = None
for cand in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp010-nb1-embedding"),
    Path("/kaggle/input/birdclef2026-exp010-nb1-embedding"),
]:
    if cand.exists() and (cand / "soundscape_embeddings.npz").exists():
        EMB_DIR = cand; break
assert EMB_DIR is not None, "soundscape_embeddings.npz not found"
print(f"EMB_DIR: {EMB_DIR}")

# ---- Mel caches (kernel_sources) ----
def find_mel_dir(slug):
    bases = [Path(f"/kaggle/input/notebooks/maekeso/{slug}"),
             Path(f"/kaggle/input/datasets/maekeso/{slug}"),
             Path(f"/kaggle/input/{slug}")]
    print(f"  find_mel_dir({slug}):")
    for base in bases:
        exists = base.exists()
        print(f"    {base} exists={exists}")
        if not exists:
            continue
        for p in [base / "mel_cache" / "train_audio",
                  base / "mel_cache" / "train_soundscapes",
                  base / "mel_cache",
                  base]:
            if p.exists():
                first_npy = next(p.rglob("*.npy"), None)
                print(f"      subdir {p.name}: first_npy={first_npy}")
                if first_npy is not None:
                    return p
    return None

# Debug: show all mounts
import os as _os
print("/kaggle/input contents:", sorted(_os.listdir("/kaggle/input")))

MEL_TRAIN_DIR = find_mel_dir("birdclef2026-mel-cache-train-audio-256")
MEL_SC_DIR    = find_mel_dir("birdclef2026-mel-cache-train-sc-256")
print(f"MEL train_audio: {MEL_TRAIN_DIR}")
print(f"MEL soundscape:  {MEL_SC_DIR}")
# These may be optional during early phases (Phase 1 only needs raw audio + emb)

# ---- Tucker SED ONNX (dataset) ----
def find_sed_dir():
    cands = [Path("/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public"),
             Path("/kaggle/input/bc2026-distilled-sed-public"),
             Path("/kaggle/input/tuckerarrants/bc2026-distilled-sed-public")]
    for c in cands:
        if c.exists() and any(c.rglob("sed_fold*.onnx")):
            return c
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    return hits[0].parent if hits else None

SED_DIR = find_sed_dir()
print(f"SED_DIR: {SED_DIR}")"""))

# ===================== Phase 1 header =====================
cells.append(md_cell("p1_hdr", r"""## Phase 1 — Self-distilled pseudo-label generation

Train **LightProtoSSM** on 66 labeled soundscape files using cached Perch
embeddings, run inference on **all 10,658 train_soundscapes**, then **rank-blend
60% ProtoSSM + 40% Tucker SED** to produce the pseudo-labels used by Phase 2.

This phase is skipped if `pseudo_labels.npy` already exists in `/kaggle/working`.
"""))

# ===================== Phase 1: load Perch embeddings =====================
cells.append(code_cell("p1_load", r"""# ============================================================
# Load Perch embeddings + meta from exp010 NB1
# ============================================================
PSEUDO_PATH = WORK_DIR / "pseudo_labels.npy"
PROTO_PATH  = WORK_DIR / "proto_scores.npy"
TUCKER_PATH = WORK_DIR / "tucker_scores.npy"

PHASE1_DONE = PSEUDO_PATH.exists()
print(f"PHASE 1 done={PHASE1_DONE}  proto={PROTO_PATH.exists()}  tucker={TUCKER_PATH.exists()}")

if not PHASE1_DONE:
    print(f"Loading {EMB_DIR}/soundscape_embeddings.npz ...")
    _t0 = time.time()
    npz = np.load(EMB_DIR / "soundscape_embeddings.npz")
    SC_EMB    = npz["embeddings"].astype(np.float16)   # (N, 1536)
    SC_SCORES = npz["scores"].astype(np.float16)       # (N, 234)
    print(f"  embeddings={SC_EMB.shape} scores={SC_SCORES.shape} in {time.time()-_t0:.1f}s")

    SC_META = pd.read_parquet(EMB_DIR / "soundscape_meta.parquet")
    print(f"  meta: {len(SC_META)} rows, columns={list(SC_META.columns)}")

    # Order meta to match emb rows; assume same order as written by NB1
    assert len(SC_META) == len(SC_EMB), \
        f"meta {len(SC_META)} != emb {len(SC_EMB)}"

    # Normalize columns
    if "filename" not in SC_META.columns:
        raise RuntimeError("filename column missing")
    if "site" not in SC_META.columns:
        SC_META["site"] = "UNK"
    if "hour_utc" not in SC_META.columns:
        SC_META["hour_utc"] = 0
    if "window_idx" not in SC_META.columns:
        # derive from running counter every 12 rows
        SC_META["window_idx"] = np.tile(np.arange(N_WINDOWS), len(SC_META) // N_WINDOWS)

    SC_FILES = SC_META["filename"].iloc[::N_WINDOWS].reset_index(drop=True).tolist()
    N_SC_FILES = len(SC_FILES)
    print(f"  unique files: {N_SC_FILES}")
    assert N_SC_FILES * N_WINDOWS == len(SC_META)
else:
    SC_EMB = SC_SCORES = SC_META = None
    SC_FILES = []
    N_SC_FILES = 0
    print("Phase 1 already complete — skipping load")"""))

# ===================== Phase 1: LightProtoSSM definition =====================
cells.append(code_cell("p1_proto_def", r"""# ============================================================
# LightProtoSSM (gate-fake008 architecture, cross-attention + SWA)
# ============================================================
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d   = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_model))
        self.B_proj   = nn.Linear(d_model, d_state, bias=False)
        self.C_proj   = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_classes=234, n_windows=12, dropout=0.15,
                 n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=2):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc  = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd  = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd  = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge= nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop     = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads=cross_attn_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(2)])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.prototypes   = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp   = nn.Parameter(torch.tensor(5.0))
        self.class_bias   = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(zip(
                self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)):
            res = h
            h_f = fwd(h); h_b = bwd(h.flip(1)).flip(1)
            h   = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h   = norm(h + res)
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = (torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp)
               + self.class_bias[None, None, :])
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out   = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

print("LightProtoSSM defined")"""))

# ===================== Phase 1: ProtoSSM training =====================
cells.append(code_cell("p1_proto_train", r"""# ============================================================
# Train LightProtoSSM on 66 labeled soundscape files
# ============================================================
PROTO_CKPT_PATH = WORK_DIR / "protossm.pt"

def parse_sc_meta(fname):
    # Parse `BC2026_Train_0001_S05_20250227_010002.ogg` style names.
    stem = Path(fname).stem
    parts = stem.split("_")
    site = "UNK"; hour = 0
    for p in parts:
        if re.fullmatch(r"S\d+", p):
            site = p; break
    # 6-digit time appears as last token (HHMMSS)
    if len(parts) >= 6 and re.fullmatch(r"\d{6}", parts[-1]):
        hour = int(parts[-1][:2])
    return site, hour


def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)


def time_to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t)
    if ":" in s:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(s)


if not PHASE1_DONE and not PROTO_PATH.exists():
    sc_lab = pd.read_csv(SC_LABELS_CSV)
    print(f"Soundscape labels rows: {len(sc_lab)}, files: {sc_lab['filename'].nunique()}")

    # Group rows by (filename, end_sec) to get one window-level label set
    sc_lab["end_sec_int"] = pd.to_timedelta(sc_lab["end"]).dt.total_seconds().astype(int)
    sc_grp = (sc_lab.groupby(["filename", "end_sec_int"])["primary_label"]
              .apply(union_labels).reset_index(name="label_list"))
    sc_grp["row_id"] = sc_grp["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_grp["end_sec_int"].astype(str)

    meta_rows = sc_grp["filename"].apply(lambda f: parse_sc_meta(f)).tolist()
    sc_grp["site"]     = [m[0] for m in meta_rows]
    sc_grp["hour_utc"] = [m[1] for m in meta_rows]

    # Build dense label matrix per (filename, window_idx) covering all 12 windows
    labeled_files = sorted(sc_grp["filename"].unique().tolist())
    n_lab_files = len(labeled_files)
    print(f"Labeled files: {n_lab_files}")
    Y_full = np.zeros((n_lab_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
    file_to_idx = {f: i for i, f in enumerate(labeled_files)}
    for _, r in sc_grp.iterrows():
        fi = file_to_idx[r["filename"]]
        wi = max(0, min(N_WINDOWS - 1, r["end_sec_int"] // WINDOW_SEC - 1))
        for lbl in r["label_list"]:
            if lbl in LABEL_TO_IDX:
                Y_full[fi * N_WINDOWS + wi, LABEL_TO_IDX[lbl]] = 1.0

    # Index into the cached embeddings array
    fname_to_emb_start = {f: i * N_WINDOWS for i, f in enumerate(SC_FILES)}
    sel = []
    for f in labeled_files:
        if f not in fname_to_emb_start:
            print(f"  warn: labeled file not in embeddings: {f}"); continue
        s = fname_to_emb_start[f]
        sel.extend(range(s, s + N_WINDOWS))
    sel = np.array(sel, dtype=np.int64)
    print(f"Selected emb rows: {len(sel)}")
    if len(sel) != n_lab_files * N_WINDOWS:
        # Drop rows for missing files; rebuild Y_full as well
        keep_files = [f for f in labeled_files if f in fname_to_emb_start]
        labeled_files = keep_files
        n_lab_files = len(labeled_files)
        Y_full = np.zeros((n_lab_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
        file_to_idx = {f: i for i, f in enumerate(labeled_files)}
        for _, r in sc_grp.iterrows():
            if r["filename"] not in file_to_idx: continue
            fi = file_to_idx[r["filename"]]
            wi = max(0, min(N_WINDOWS - 1, r["end_sec_int"] // WINDOW_SEC - 1))
            for lbl in r["label_list"]:
                if lbl in LABEL_TO_IDX:
                    Y_full[fi * N_WINDOWS + wi, LABEL_TO_IDX[lbl]] = 1.0
        sel = []
        for f in labeled_files:
            s = fname_to_emb_start[f]
            sel.extend(range(s, s + N_WINDOWS))
        sel = np.array(sel, dtype=np.int64)

    emb_lab = SC_EMB[sel].astype(np.float32)
    log_lab = SC_SCORES[sel].astype(np.float32)
    # Logits from sigmoid scores
    log_lab = np.log(np.clip(log_lab, 1e-7, 1 - 1e-7) / np.clip(1 - log_lab, 1e-7, 1.0))

    # Build site/hour arrays per file
    site_str_per_file = [parse_sc_meta(f)[0] for f in labeled_files]
    hour_per_file     = [parse_sc_meta(f)[1] for f in labeled_files]
    sites_u = sorted(set(site_str_per_file))
    site2i  = {s: i + 1 for i, s in enumerate(sites_u)}  # 0 reserved for unknown
    N_SITES = max(20, len(sites_u) + 1)
    site_ids = np.array([min(site2i.get(s, 0), N_SITES - 1) for s in site_str_per_file], dtype=np.int64)
    hour_ids = np.array([h % 24 for h in hour_per_file], dtype=np.int64)

    emb_f  = emb_lab.reshape(n_lab_files, N_WINDOWS, -1)
    log_f  = log_lab.reshape(n_lab_files, N_WINDOWS, -1)
    Y_f    = Y_full.reshape(n_lab_files, N_WINDOWS, -1)

    print(f"Train tensors: emb={emb_f.shape} log={log_f.shape} Y={Y_f.shape}")

    proto = LightProtoSSM(n_classes=N_CLASSES, n_sites=N_SITES,
                          use_cross_attn=True, cross_attn_heads=2).to(DEVICE)
    proto.init_prototypes(
        torch.tensor(emb_lab, dtype=torch.float32, device=DEVICE),
        torch.tensor(Y_full,  dtype=torch.float32, device=DEVICE))
    print(f"ProtoSSM params: {proto.count_parameters():,}")

    emb_t  = torch.tensor(emb_f,    dtype=torch.float32, device=DEVICE)
    log_t  = torch.tensor(log_f,    dtype=torch.float32, device=DEVICE)
    Y_t    = torch.tensor(Y_f,      dtype=torch.float32, device=DEVICE)
    site_t = torch.tensor(site_ids, dtype=torch.long,    device=DEVICE)
    hour_t = torch.tensor(hour_ids, dtype=torch.long,    device=DEVICE)

    pos_cnt    = Y_t.sum(dim=(0, 1))
    total_cnt  = Y_t.shape[0] * Y_t.shape[1]
    pos_weight = ((total_cnt - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    EPOCHS_PROTO = 40
    PATIENCE = 8
    LR = 1e-3
    opt = torch.optim.AdamW(proto.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS_PROTO, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")
    swa_model = torch.optim.swa_utils.AveragedModel(proto)
    swa_start = int(EPOCHS_PROTO * 0.65)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)

    best_loss = float("inf"); best_state = None; wait = 0; ep_done = -1
    for ep in range(EPOCHS_PROTO):
        proto.train()
        out = proto(emb_t, log_t, site_ids=site_t, hours=hour_t)
        loss = (F.binary_cross_entropy_with_logits(out, Y_t,
                                                   pos_weight=pos_weight[None, None, :])
                + 0.15 * F.mse_loss(out, log_t))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(proto.parameters(), 1.0)
        opt.step()
        if ep >= swa_start:
            swa_model.update_parameters(proto); swa_sched.step()
        else:
            sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in proto.state_dict().items()}
            wait = 0
        else:
            wait += 1
        ep_done = ep
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:02d}/{EPOCHS_PROTO}  loss={loss.item():.4f}  best={best_loss:.4f}")
        if wait >= PATIENCE:
            print(f"  early stop ep {ep+1}")
            break

    if ep_done >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        proto = swa_model
    else:
        proto.load_state_dict(best_state)

    # Save weights for inference reuse
    raw_state = (proto.module.state_dict() if hasattr(proto, "module")
                 else proto.state_dict())
    atomic_save({"state_dict": raw_state,
                 "n_sites": N_SITES,
                 "site2i":  site2i}, PROTO_CKPT_PATH)
    print(f"  saved {PROTO_CKPT_PATH}")

    proto.eval()
    print(f"ProtoSSM trained, best loss={best_loss:.4f}")
elif not PHASE1_DONE:
    print(f"ProtoSSM checkpoint already present (proto_scores.npy exists? {PROTO_PATH.exists()})")
else:
    print("Phase 1 done — skipping ProtoSSM training")"""))

# ===================== Phase 1: ProtoSSM inference on all SS =====================
cells.append(code_cell("p1_proto_inf", r"""# ============================================================
# ProtoSSM inference on all 10,658 train_soundscape files
# ============================================================
if not PHASE1_DONE and not PROTO_PATH.exists():
    print(f"ProtoSSM inference on {N_SC_FILES} files ...")
    if 'proto' not in dir():
        proto = LightProtoSSM(n_classes=N_CLASSES, n_sites=20,
                              use_cross_attn=True, cross_attn_heads=2).to(DEVICE)
        ckpt = torch.load(PROTO_CKPT_PATH, map_location=DEVICE, weights_only=False)
        proto.load_state_dict(ckpt["state_dict"])
        site2i = ckpt["site2i"]; N_SITES = ckpt["n_sites"]
        proto.eval()

    # Build per-file site/hour
    file_meta = SC_META.iloc[::N_WINDOWS][["filename", "site", "hour_utc"]].reset_index(drop=True)
    site_arr = np.zeros(len(file_meta), dtype=np.int64)
    hour_arr = np.zeros(len(file_meta), dtype=np.int64)
    for i, row in file_meta.iterrows():
        s = str(row["site"])
        h = int(row["hour_utc"]) if not pd.isna(row["hour_utc"]) else 0
        site_arr[i] = min(site2i.get(s, 0), N_SITES - 1)
        hour_arr[i] = h % 24

    # Logits from cached scores
    sc_logits = np.log(np.clip(SC_SCORES.astype(np.float32), 1e-7, 1 - 1e-7)
                       / np.clip(1 - SC_SCORES.astype(np.float32), 1e-7, 1.0))

    proto_logits_full = np.zeros((N_SC_FILES, N_WINDOWS, N_CLASSES), dtype=np.float32)
    BATCH = 32
    proto.eval()
    _t0 = time.time()
    with torch.no_grad():
        for s in range(0, N_SC_FILES, BATCH):
            e = min(s + BATCH, N_SC_FILES)
            emb_chunk = SC_EMB[s * N_WINDOWS:e * N_WINDOWS].astype(np.float32)
            log_chunk = sc_logits[s * N_WINDOWS:e * N_WINDOWS]
            emb_t = torch.tensor(emb_chunk.reshape(e - s, N_WINDOWS, -1),
                                 dtype=torch.float32, device=DEVICE)
            log_t = torch.tensor(log_chunk.reshape(e - s, N_WINDOWS, -1),
                                 dtype=torch.float32, device=DEVICE)
            site_t = torch.tensor(site_arr[s:e], dtype=torch.long, device=DEVICE)
            hour_t = torch.tensor(hour_arr[s:e], dtype=torch.long, device=DEVICE)
            out = proto(emb_t, log_t, site_ids=site_t, hours=hour_t)
            proto_logits_full[s:e] = out.detach().float().cpu().numpy()
            if s == 0 or (s // BATCH) % 50 == 0 or e == N_SC_FILES:
                print(f"  proto inf {e}/{N_SC_FILES}  {time.time()-_t0:.1f}s")

    proto_scores = 1.0 / (1.0 + np.exp(-np.clip(proto_logits_full, -50, 50)))
    proto_scores = proto_scores.astype(np.float32)
    print(f"ProtoSSM scores: {proto_scores.shape}")
    atomic_save_npy(proto_scores, PROTO_PATH)
    print(f"  saved {PROTO_PATH}")
    del proto_logits_full, sc_logits
    gc.collect(); torch.cuda.empty_cache()
elif not PHASE1_DONE:
    print(f"proto_scores.npy already exists at {PROTO_PATH}")
else:
    print("Phase 1 done — skipping ProtoSSM inference")"""))

# ===================== Phase 1: Tucker SED inference =====================
cells.append(code_cell("p1_tucker", r"""# ============================================================
# Tucker SED ONNX 5-fold inference on all train_soundscape audio
# ============================================================
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
import onnxruntime as ort

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80


def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = []
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def audio_to_mel_sed(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def file_to_sed_chunks(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = N_WINDOWS * WINDOW_SAMPLES
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES)


def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


# Resumable: per-file partial result file
TUCKER_PARTIAL = WORK_DIR / "tucker_partial.npy"
TUCKER_DONE_IDX = WORK_DIR / "tucker_done_idx.txt"

if not PHASE1_DONE and not TUCKER_PATH.exists():
    if SED_DIR is None:
        raise FileNotFoundError("SED_DIR not found — attach tuckerarrants/bc2026-distilled-sed-public")
    sed_paths = sorted(SED_DIR.glob("sed_fold*.onnx"),
                       key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
    print(f"SED folds: {[p.name for p in sed_paths]}")
    sessions = [make_sed_session(p) for p in sed_paths]

    sc_paths = sorted(TRAIN_SC_DIR.glob("*.ogg"))
    n_files = len(sc_paths)
    assert n_files == N_SC_FILES, f"audio {n_files} != emb files {N_SC_FILES}"

    # Map ogg paths to embedding meta order (by filename)
    name_to_idx = {f: i for i, f in enumerate(SC_FILES)}
    ordered_paths = [None] * N_SC_FILES
    for p in sc_paths:
        if p.name in name_to_idx:
            ordered_paths[name_to_idx[p.name]] = p

    if TUCKER_PARTIAL.exists():
        tucker_scores = np.load(TUCKER_PARTIAL)
        if tucker_scores.shape != (N_SC_FILES, N_WINDOWS, N_CLASSES):
            print(f"  partial shape mismatch {tucker_scores.shape} -> resetting")
            tucker_scores = np.zeros((N_SC_FILES, N_WINDOWS, N_CLASSES), dtype=np.float32)
        if TUCKER_DONE_IDX.exists():
            done_idx = set(int(x) for x in TUCKER_DONE_IDX.read_text().split() if x.strip())
        else:
            done_idx = set()
        print(f"  resuming Tucker inference ({len(done_idx)} files already done)")
    else:
        tucker_scores = np.zeros((N_SC_FILES, N_WINDOWS, N_CLASSES), dtype=np.float32)
        done_idx = set()

    save_every = 200
    _t0 = time.time()
    n_processed = 0
    for i, p in enumerate(ordered_paths):
        if i in done_idx:
            continue
        if p is None:
            done_idx.add(i); continue
        try:
            chunks = file_to_sed_chunks(p)
            mel = audio_to_mel_sed(chunks)
            p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
            for sess in sessions:
                outs = sess.run(None, {sess.get_inputs()[0].name: mel})
                clip_logits = outs[0]
                frame_max   = outs[1].max(axis=1)
                p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
            p_mean = p_sum / len(sessions)
            if len(p_mean) > 1:
                p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0,
                                           mode="nearest").astype(np.float32)
            tucker_scores[i] = p_mean
            done_idx.add(i)
            n_processed += 1
        except Exception as e:
            print(f"  err {p}: {e}")
            done_idx.add(i)

        if n_processed > 0 and (n_processed % save_every == 0):
            atomic_save_npy(tucker_scores, TUCKER_PARTIAL)
            TUCKER_DONE_IDX.write_text(" ".join(str(x) for x in sorted(done_idx)))
            elapsed = time.time() - _t0
            speed = n_processed / max(elapsed, 1e-3)
            remain_files = N_SC_FILES - len(done_idx)
            eta = remain_files / max(speed, 1e-3)
            print(f"  tucker {len(done_idx)}/{N_SC_FILES}  speed={speed:.1f}f/s  eta={fmt_dur(eta)}")
            if time_low():
                print(f"  time_low -> saving partial and exiting")
                atomic_save_npy(tucker_scores, TUCKER_PARTIAL)
                TUCKER_DONE_IDX.write_text(" ".join(str(x) for x in sorted(done_idx)))
                sys.exit(0)

    atomic_save_npy(tucker_scores, TUCKER_PATH)
    print(f"  saved {TUCKER_PATH}, shape={tucker_scores.shape}")
    if TUCKER_PARTIAL.exists():
        TUCKER_PARTIAL.unlink()
    if TUCKER_DONE_IDX.exists():
        TUCKER_DONE_IDX.unlink()
    del sessions
    gc.collect()
elif not PHASE1_DONE:
    print(f"tucker_scores.npy already exists at {TUCKER_PATH}")
else:
    print("Phase 1 done — skipping Tucker inference")"""))

# ===================== Phase 1: probability blend =====================
cells.append(code_cell("p1_blend", r"""# ============================================================
# Probability blend: 0.6 * ProtoSSM + 0.4 * Tucker SED -> pseudo_labels_prob.npy
# (Per-class rank normalization destroys absolute probabilities and makes
#  pseudo-labels uninformative for BCE training, so we use raw probabilities.)
# ============================================================
PSEUDO_PROB_PATH = WORK_DIR / "pseudo_labels_prob.npy"

if not PSEUDO_PROB_PATH.exists():
    print("Loading proto + tucker scores ...")
    proto_scores = np.load(PROTO_PATH).astype(np.float32)   # (N_files, 12, 234)
    tucker_scores = np.load(TUCKER_PATH).astype(np.float32)
    print(f"  proto={proto_scores.shape}  mean={proto_scores.mean():.4f}  max={proto_scores.max():.4f}")
    print(f"  tucker={tucker_scores.shape}  mean={tucker_scores.mean():.4f}  max={tucker_scores.max():.4f}")
    assert proto_scores.shape == tucker_scores.shape

    pseudo = (0.6 * proto_scores + 0.4 * tucker_scores).astype(np.float32)
    pseudo = np.clip(pseudo, 0.0, 1.0)
    print(f"Pseudo labels (prob blend): shape={pseudo.shape}, mean={pseudo.mean():.4f}, max={pseudo.max():.4f}")
    print(f"  positives @0.5: {(pseudo > 0.5).sum()} / {pseudo.size} = {(pseudo > 0.5).mean()*100:.3f}%")
    print(f"  positives @0.3: {(pseudo > 0.3).sum()} / {pseudo.size} = {(pseudo > 0.3).mean()*100:.3f}%")
    atomic_save_npy(pseudo, PSEUDO_PROB_PATH)
    print(f"  saved {PSEUDO_PROB_PATH}")

    # Free large arrays before Phase 2
    del proto_scores, tucker_scores, pseudo
    gc.collect()
else:
    print(f"pseudo_labels_prob.npy already exists at {PSEUDO_PROB_PATH}")

# Use the prob-blend file going forward
PSEUDO_PATH = PSEUDO_PROB_PATH

# Free Phase 1 globals
SC_EMB = SC_SCORES = SC_META = None
gc.collect()
print(f"Phase 1 wall: {fmt_dur(time.time() - WALL_START)}, time_left={fmt_dur(time_left())}")"""))

# ===================== Phase 2 header =====================
cells.append(md_cell("p2_hdr", r"""## Phase 2 — SED training (EfficientNet-B0 then B1)

3-fold StratifiedGroupKFold by `primary_label` grouped by `author`.
- train_audio: hard labels (primary + secondary) + random 5s crop.
- train_soundscapes: soft pseudo-labels (Phase 1 output) per 5s window.
- 25 epochs/fold, AdamW 5e-4 / wd 1e-2 / CosineAnnealing, AMP.
- Atomic checkpoint each epoch; best by macro AUC; `done.flag` after final epoch.
- Each fold checks `time_low()` between epochs and exits cleanly if needed.
"""))

# ===================== Phase 2: build metadata =====================
cells.append(code_cell("p2_meta", r"""# ============================================================
# Phase 2: build training metadata
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)
print(f"train.csv rows: {len(train_df)}")
# Filter to existing audio files (mel cache may not have every file)
def _parse_secondary_labels(s):
    if pd.isna(s) or s in ("[]", ""):
        return []
    try:
        parsed = ast.literal_eval(s)
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []

# Fill author with filename hash if missing
if "author" not in train_df.columns:
    train_df["author"] = train_df["filename"].astype(str)
train_df["author"] = train_df["author"].fillna("").astype(str)
mask_no_author = train_df["author"] == ""
if mask_no_author.any():
    train_df.loc[mask_no_author, "author"] = (
        train_df.loc[mask_no_author, "filename"].astype(str).str.split("/").str[-1])
print(f"unique authors: {train_df['author'].nunique()}")
print(f"unique primary_label: {train_df['primary_label'].nunique()}")

# Pseudo labels
PSEUDO_LABELS = np.load(PSEUDO_PATH).astype(np.float32)  # (N_SC_FILES, 12, 234)
print(f"Pseudo labels loaded: {PSEUDO_LABELS.shape}")

# Soundscape file order — recover from competition listing
sc_paths = sorted(TRAIN_SC_DIR.glob("*.ogg"))
SC_FILE_NAMES = [p.name for p in sc_paths]
N_SC_FILES = len(SC_FILE_NAMES)
print(f"train_soundscape files: {N_SC_FILES}")
# In Phase 1 we ordered tucker_scores by SC_FILES; pseudo_labels.npy follows that order.
# To rebuild the ordering deterministically here, look up the embedding meta.
EMB_META_PATH = EMB_DIR / "soundscape_meta.parquet"
emb_meta = pd.read_parquet(EMB_META_PATH)
SC_FILE_ORDER = emb_meta["filename"].iloc[::N_WINDOWS].reset_index(drop=True).tolist()
assert len(SC_FILE_ORDER) == PSEUDO_LABELS.shape[0]
SC_FILE_TO_PSEUDO_IDX = {f: i for i, f in enumerate(SC_FILE_ORDER)}
print(f"pseudo idx mapping built ({len(SC_FILE_TO_PSEUDO_IDX)})")

# ---- Sanity check: mel cache resolution (debug zero-mel issue) ----
print("\nMel cache sanity check:")
sample_train = train_df.head(5)
hits_train = 0
for _, row in sample_train.iterrows():
    pl = str(row["primary_label"])
    fn = row["filename"]
    cands = []
    if MEL_TRAIN_DIR is not None:
        cands.append(MEL_TRAIN_DIR / fn.replace(".ogg", ".npy"))
        cands.append(MEL_TRAIN_DIR / pl / (Path(fn).stem + ".npy"))
        cands.append(MEL_TRAIN_DIR / (Path(fn).stem + ".npy"))
    found = next((c for c in cands if c.exists()), None)
    print(f"  {fn} -> found={found}")
    if found is not None:
        hits_train += 1
print(f"  {hits_train}/{len(sample_train)} train_audio mels found")
if hits_train == 0:
    print("ERROR: No train_audio mel cache files found - check MEL_TRAIN_DIR above")
    print("  MEL_TRAIN_DIR:", MEL_TRAIN_DIR)
    # Show first few actual files in MEL_TRAIN_DIR to understand structure
    if MEL_TRAIN_DIR is not None:
        actual = sorted(MEL_TRAIN_DIR.rglob("*.npy"))[:5]
        print("  Actual .npy files found:", actual)
    raise RuntimeError("train_audio mel cache missing")

if MEL_SC_DIR is not None and SC_FILE_ORDER:
    print("\nSC mel cache sanity:")
    hits_sc = 0
    for fn in SC_FILE_ORDER[:5]:
        cands = [MEL_SC_DIR / fn.replace(".ogg", ".npy"),
                 MEL_SC_DIR / (Path(fn).stem + ".npy")]
        found = next((c for c in cands if c.exists()), None)
        print(f"  {fn} -> found={found}")
        if found is not None:
            hits_sc += 1
    print(f"  {hits_sc}/5 SC mels found")
    if hits_sc == 0:
        print("  WARN: no SC mels found, soundscape branch will be skipped")
        MEL_SC_DIR = None"""))

# ===================== Phase 2: split =====================
cells.append(code_cell("p2_split", r"""# ============================================================
# 3-fold StratifiedGroupKFold by primary_label, grouped by author
# ============================================================
N_FOLDS = 3
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = []
y = train_df["primary_label"].values
g = train_df["author"].values
for tr_idx, va_idx in sgkf.split(train_df, y=y, groups=g):
    folds.append((tr_idx, va_idx))
for f, (ti, vi) in enumerate(folds):
    print(f"  fold {f}: train={len(ti)} val={len(vi)}")"""))

# ===================== Phase 2: dataset =====================
cells.append(code_cell("p2_dataset", r"""# ============================================================
# Mel-cache backed datasets
# ============================================================
TARGET_FRAMES_5S = WINDOW_FRAMES   # 313


def dequantize_mel(mel_uint8):
    return mel_uint8.astype(np.float32) / 255.0 * DB_RANGE + DB_MIN


def crop_random_5s(mel, target_frames=TARGET_FRAMES_5S):
    Tf = mel.shape[1]
    if Tf >= target_frames:
        s = np.random.randint(0, Tf - target_frames + 1)
        return mel[:, s:s + target_frames]
    pad = np.zeros((mel.shape[0], target_frames), dtype=mel.dtype)
    pad[:, :Tf] = mel
    return pad


def crop_center_5s(mel, target_frames=TARGET_FRAMES_5S):
    Tf = mel.shape[1]
    if Tf >= target_frames:
        s = max(0, (Tf - target_frames) // 2)
        return mel[:, s:s + target_frames]
    pad = np.zeros((mel.shape[0], target_frames), dtype=mel.dtype)
    pad[:, :Tf] = mel
    return pad


def znorm_chunk(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def find_mel_path(mel_dir, filename, primary_label=None):
    if mel_dir is None:
        return None
    cand = []
    cand.append(mel_dir / filename.replace(".ogg", ".npy"))
    if primary_label is not None:
        cand.append(mel_dir / primary_label / (Path(filename).stem + ".npy"))
    cand.append(mel_dir / (Path(filename).stem + ".npy"))
    for c in cand:
        if c.exists():
            return c
    # Fallback: try recursively for filename stem
    stem = Path(filename).stem
    hits = list(mel_dir.rglob(stem + ".npy"))
    return hits[0] if hits else None


class TrainAudioMelDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.mel_dir = MEL_TRAIN_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = find_mel_path(self.mel_dir, row["filename"], str(row["primary_label"]))
        if path is None:
            mel_uint8 = np.zeros((N_MELS, TARGET_FRAMES_5S), dtype=np.uint8)
        else:
            try:
                mel_uint8 = np.load(str(path))
            except Exception:
                mel_uint8 = np.zeros((N_MELS, TARGET_FRAMES_5S), dtype=np.uint8)
        if self.mode == "train":
            mel_uint8 = crop_random_5s(mel_uint8, TARGET_FRAMES_5S)
        else:
            mel_uint8 = crop_center_5s(mel_uint8, TARGET_FRAMES_5S)
        mel_db = dequantize_mel(mel_uint8)
        mel_db = znorm_chunk(mel_db)

        label = np.zeros(N_CLASSES, dtype=np.float32)
        sp = str(row["primary_label"])
        if sp in LABEL_TO_IDX:
            label[LABEL_TO_IDX[sp]] = 1.0
        for sec in _parse_secondary_labels(row.get("secondary_labels", "[]")):
            if sec in LABEL_TO_IDX:
                label[LABEL_TO_IDX[sec]] = 1.0
        return torch.from_numpy(mel_db).float().unsqueeze(0), torch.from_numpy(label).float()


class SoundscapePseudoDataset(Dataset):
    # Window-level dataset over train_soundscapes with pseudo-label targets.
    def __init__(self, sc_files, file_to_pseudo_idx, pseudo, mode="train"):
        self.sc_files = sc_files
        self.f2i = file_to_pseudo_idx
        self.pseudo = pseudo
        self.mode = mode
        self.mel_dir = MEL_SC_DIR
        # Build (file_idx, window_idx) index space
        self.items = []
        for f in self.sc_files:
            if f not in self.f2i:
                continue
            for w in range(N_WINDOWS):
                self.items.append((f, w))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, wi = self.items[idx]
        path = find_mel_path(self.mel_dir, fname, None)
        if path is None:
            mel_uint8 = np.zeros((N_MELS, TARGET_FRAMES_5S), dtype=np.uint8)
        else:
            try:
                mel_uint8 = np.load(str(path))
            except Exception:
                mel_uint8 = np.zeros((N_MELS, TARGET_FRAMES_5S), dtype=np.uint8)
        # Window-anchored crop
        Tf = mel_uint8.shape[1]
        # If full-duration (~3760 frames for 60s) take the wi-th 5s window slice
        full_frames_per_60s = WINDOW_FRAMES * N_WINDOWS  # 3756
        if Tf >= full_frames_per_60s:
            s = wi * WINDOW_FRAMES
            chunk = mel_uint8[:, s:s + TARGET_FRAMES_5S]
        else:
            chunk = crop_random_5s(mel_uint8, TARGET_FRAMES_5S) if self.mode == "train" \
                    else crop_center_5s(mel_uint8, TARGET_FRAMES_5S)
        mel_db = dequantize_mel(chunk)
        mel_db = znorm_chunk(mel_db)
        label = self.pseudo[self.f2i[fname], wi].astype(np.float32)
        return torch.from_numpy(mel_db).float().unsqueeze(0), torch.from_numpy(label).float()


class ConcatDS(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.cum = np.cumsum(self.lens)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        di = int(np.searchsorted(self.cum, idx, side="right"))
        local = idx - (self.cum[di - 1] if di > 0 else 0)
        return self.datasets[di][local]


print("Dataset classes ready")"""))

# ===================== Phase 2: model =====================
cells.append(code_cell("p2_model", r"""# ============================================================
# Model: timm backbone + GeMFreqPool + AttentionSEDHead
# ============================================================
import timm

class GeMFreqPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, F, T)
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            p = self.p.clamp(min=1.0)
            return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn  = nn.BatchNorm1d(in_channels)
        self.att = nn.Conv1d(in_channels, num_classes, 1)
        self.cla = nn.Conv1d(in_channels, num_classes, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn(self.fc1(x))))
        att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla_logit = self.cla(x)                              # (B, C, T)
        clip_prob = (att * torch.sigmoid(cla_logit)).sum(dim=-1)  # (B, C)
        clip_prob = clip_prob.clamp(1e-7, 1 - 1e-7)
        clip_logit = torch.log(clip_prob / (1 - clip_prob))  # logit of clip
        return clip_logit, cla_logit


class SEDModel(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0", num_classes=N_CLASSES,
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            features_only=True, in_chans=1)
        feat_channels = self.backbone.feature_info.channels()[-1]
        self.gem_pool = GeMFreqPool()
        self.head = AttentionSEDHead(feat_channels, num_classes)
        self.backbone_name = backbone_name

    def forward(self, mel):  # mel: (B, 1, n_mels, T)
        feat = self.backbone(mel)[-1]   # (B, C, F', T')
        x = self.gem_pool(feat)         # (B, C, T')
        return self.head(x)             # logits clip + frame


_m = SEDModel("tf_efficientnet_b0", pretrained=False)
print(f"B0 backbone last C: {_m.backbone.feature_info.channels()[-1]}")
print(f"B0 params: {sum(p.numel() for p in _m.parameters())/1e6:.2f}M")
del _m
gc.collect()"""))

# ===================== Phase 2: train_one_fold =====================
cells.append(code_cell("p2_train", r"""# ============================================================
# Resumable per-fold training loop
# ============================================================
import torchaudio.transforms as TA

class SpecAug(nn.Module):
    def __init__(self, freq_param=30, time_param=40):
        super().__init__()
        self.freq = TA.FrequencyMasking(freq_mask_param=freq_param)
        self.time = TA.TimeMasking(time_mask_param=time_param)

    def forward(self, mel):
        return self.time(self.freq(mel))


class SpecMixUp:
    def __init__(self, prob=0.5, alpha=0.5):
        self.prob = prob; self.alpha = alpha

    def __call__(self, mel, labels):
        if torch.rand(1).item() > self.prob:
            return mel, labels
        idx = torch.randperm(mel.size(0), device=mel.device)
        lam = float(np.random.beta(self.alpha, self.alpha))
        mixed = lam * mel + (1.0 - lam) * mel[idx]
        # Soft mix on labels (max OR for hard, weighted for soft pseudo)
        mixed_labels = torch.maximum(labels, labels[idx])
        return mixed, mixed_labels


def compute_macro_auc(preds, targets):
    if preds.size == 0:
        return 0.0
    aucs = []
    for j in range(targets.shape[1]):
        col = targets[:, j]
        # For soft pseudo target use threshold .5; if no positive, skip
        bin_col = (col > 0.5).astype(np.int32)
        if bin_col.sum() > 0 and bin_col.sum() < len(bin_col):
            try:
                aucs.append(roc_auc_score(bin_col, preds[:, j]))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def loss_clip_frame(clip_logit, cla_logit, target):
    # target: (B, num_classes)
    loss_c = F.binary_cross_entropy_with_logits(clip_logit, target)
    frame_max = cla_logit.max(dim=-1)[0]
    loss_f = F.binary_cross_entropy_with_logits(frame_max, target)
    return 0.5 * loss_c + 0.5 * loss_f


def make_loader(ds, batch_size, shuffle, num_workers=2, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last, persistent_workers=(num_workers > 0))


def train_one_fold(backbone_name, fold, train_idx, val_idx,
                   epochs=25, batch_size=64, lr=5e-4, wd=1e-2,
                   ckpt_prefix="b0"):
    # Resumable per-fold training. Skips epochs already completed.
    fold_done_flag = CKPT_DIR / f"{ckpt_prefix}_fold{fold}_done.flag"
    if fold_done_flag.exists():
        print(f"  [{ckpt_prefix} fold {fold}] already done, skipping")
        return

    print(f"\n{'='*60}\n  TRAIN {ckpt_prefix} fold {fold}\n{'='*60}")
    set_seed(SEED + fold)

    tr_df = train_df.iloc[train_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"  train_audio: tr={len(tr_df)} va={len(va_df)}")

    train_audio_ds = TrainAudioMelDataset(tr_df, mode="train")
    val_audio_ds   = TrainAudioMelDataset(va_df, mode="val")

    # Soundscape pseudo data: only for training, not validation
    if MEL_SC_DIR is not None:
        sc_ds = SoundscapePseudoDataset(SC_FILE_ORDER, SC_FILE_TO_PSEUDO_IDX, PSEUDO_LABELS, mode="train")
        print(f"  soundscape pseudo segs: {len(sc_ds)}")
        train_ds = ConcatDS(train_audio_ds, sc_ds)
    else:
        print("  WARN: MEL_SC_DIR not mounted — train_audio only")
        train_ds = train_audio_ds
    print(f"  total train samples: {len(train_ds)}")

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader   = make_loader(val_audio_ds, batch_size=batch_size * 2, shuffle=False,
                               num_workers=2, drop_last=False)

    model = SEDModel(backbone_name, pretrained=True).to(DEVICE)
    spec_aug = SpecAug(30, 40).to(DEVICE)
    mixup = SpecMixUp(prob=0.5, alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda")

    # Resume — check for last checkpoint
    start_ep = 0
    best_auc = 0.0
    best_ep  = -1
    log_rows = []
    log_path = CKPT_DIR / f"{ckpt_prefix}_fold{fold}_log.json"
    if log_path.exists():
        try:
            log_rows = json.loads(log_path.read_text())
            start_ep = max((r["epoch"] for r in log_rows), default=-1) + 1
            best_auc = max((r.get("val_auc", 0.0) for r in log_rows), default=0.0)
            for r in log_rows:
                if r.get("val_auc", 0.0) >= best_auc:
                    best_ep = r["epoch"]
        except Exception as e:
            print(f"  log read err: {e}")
    print(f"  start_ep={start_ep}, best_auc_so_far={best_auc:.4f}")

    last_ckpt = None
    for ep in range(start_ep - 1, -1, -1):
        cand = CKPT_DIR / f"{ckpt_prefix}_fold{fold}_ep{ep}.pt"
        if cand.exists():
            last_ckpt = cand; break
    if last_ckpt is not None:
        try:
            ckpt = torch.load(str(last_ckpt), map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            print(f"  resumed from {last_ckpt.name} (epoch {ckpt['epoch']})")
        except Exception as e:
            print(f"  resume failed: {e}, restart from scratch")
            start_ep = 0

    if start_ep >= epochs:
        # Already finished but flag missing — write flag
        fold_done_flag.write_text("done")
        print(f"  fold {fold} already had {start_ep} epochs of log, marking done")
        return

    for ep in range(start_ep, epochs):
        if time_low():
            print(f"  time_low at start of ep {ep} — exiting cleanly")
            sys.exit(0)
        ep_t0 = time.time()
        model.train()
        run_loss = 0.0; nb = 0
        for batch_idx, (mel, label) in enumerate(tqdm(train_loader, desc=f"  ep{ep:02d} train", leave=False)):
            mel = mel.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            mel, label = mixup(mel, label)
            mel = spec_aug(mel)
            with torch.amp.autocast("cuda"):
                clip_logit, cla_logit = model(mel)
                loss = loss_clip_frame(clip_logit, cla_logit, label)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item(); nb += 1
        scheduler.step()
        train_loss = run_loss / max(nb, 1)

        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for mel, label in tqdm(val_loader, desc=f"  ep{ep:02d}  val ", leave=False):
                mel = mel.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    clip_logit, _ = model(mel)
                preds.append(torch.sigmoid(clip_logit).float().cpu().numpy())
                targs.append(label.numpy())
        preds = np.concatenate(preds, axis=0) if preds else np.zeros((0, N_CLASSES))
        targs = np.concatenate(targs, axis=0) if targs else np.zeros((0, N_CLASSES))
        val_auc = compute_macro_auc(preds, targs)

        elapsed = time.time() - ep_t0
        is_best = val_auc > best_auc
        flag = " <- best" if is_best else ""
        print(f"    ep{ep:02d}/{epochs}  loss={train_loss:.4f}  val_auc={val_auc:.4f}  ({elapsed:.0f}s){flag}")

        log_rows.append(dict(epoch=ep, train_loss=train_loss, val_auc=val_auc, time=elapsed))
        log_path.write_text(json.dumps(log_rows, indent=2))

        ckpt_path = CKPT_DIR / f"{ckpt_prefix}_fold{fold}_ep{ep}.pt"
        atomic_save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_auc": val_auc,
            "backbone": backbone_name,
            "primary_labels": PRIMARY_LABELS,
        }, ckpt_path)

        if is_best:
            best_auc = val_auc; best_ep = ep
            best_path = CKPT_DIR / f"{ckpt_prefix}_fold{fold}_best.pt"
            atomic_save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_auc": val_auc,
                "backbone": backbone_name,
                "primary_labels": PRIMARY_LABELS,
            }, best_path)

        # Cleanup older epoch ckpts (keep last 2 + best)
        for old_ep in range(0, ep - 1):
            old = CKPT_DIR / f"{ckpt_prefix}_fold{fold}_ep{old_ep}.pt"
            if old.exists():
                try: old.unlink()
                except: pass

        # Time check after each epoch
        if time_low() and ep < epochs - 1:
            print(f"  time_low after ep {ep} — exiting cleanly")
            sys.exit(0)

    fold_done_flag.write_text("done")
    print(f"  fold {fold} DONE — best auc={best_auc:.4f} @ ep{best_ep}")
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()


print("train_one_fold ready")"""))

# ===================== Phase 2: B0 training =====================
cells.append(code_cell("p2_b0", r"""# ============================================================
# Train EfficientNet-B0 (3 folds)
# ============================================================
B0_NAME = "tf_efficientnet_b0"
B0_EPOCHS = 25
B0_BATCH = 64

for f in range(N_FOLDS):
    if time_low():
        print(f"time_low — skipping B0 fold {f}")
        sys.exit(0)
    tr_idx, va_idx = folds[f]
    train_one_fold(B0_NAME, f, tr_idx, va_idx,
                   epochs=B0_EPOCHS, batch_size=B0_BATCH,
                   lr=5e-4, wd=1e-2, ckpt_prefix="b0")

print(f"B0 phase wall: {fmt_dur(time.time() - WALL_START)}")"""))

# ===================== Phase 2: B1 training =====================
cells.append(code_cell("p2_b1", r"""# ============================================================
# Train EfficientNet-B1 (3 folds)
# ============================================================
B1_NAME = "tf_efficientnet_b1"
B1_EPOCHS = 25
B1_BATCH = 48   # B1 needs slightly less than B0 on T4

for f in range(N_FOLDS):
    if time_low():
        print(f"time_low — skipping B1 fold {f}")
        sys.exit(0)
    tr_idx, va_idx = folds[f]
    train_one_fold(B1_NAME, f, tr_idx, va_idx,
                   epochs=B1_EPOCHS, batch_size=B1_BATCH,
                   lr=5e-4, wd=1e-2, ckpt_prefix="b1")

print(f"B1 phase wall: {fmt_dur(time.time() - WALL_START)}")"""))

# ===================== Phase 3 header =====================
cells.append(md_cell("p3_hdr", r"""## Phase 3 — ONNX export

Each `best.pt` is exported to ONNX with input `(batch, 1, 256, 313)` and outputs
`(clip_logit, frame_logit)`. Files written to `/kaggle/working/onnx/`.
"""))

# ===================== Phase 3: ONNX export =====================
cells.append(code_cell("p3_onnx", r"""# ============================================================
# Phase 3: ONNX export
# ============================================================
import onnx

class ONNXExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        clip_logit, frame_logit = self.model(mel)
        clip_prob = torch.sigmoid(clip_logit)
        frame_prob = torch.sigmoid(frame_logit)
        return clip_prob, frame_prob


def export_one(prefix, fold):
    best_pt = CKPT_DIR / f"{prefix}_fold{fold}_best.pt"
    if not best_pt.exists():
        print(f"  skip {prefix} fold {fold}: no best.pt")
        return
    onnx_path = ONNX_DIR / f"{prefix}_fold{fold}.onnx"
    if onnx_path.exists():
        print(f"  skip {prefix} fold {fold}: onnx already exists")
        return
    ckpt = torch.load(str(best_pt), map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone", f"tf_efficientnet_{prefix[-2:]}")
    model = SEDModel(backbone_name, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    wrapped = ONNXExportWrapper(model)
    dummy = torch.zeros(1, 1, N_MELS, TARGET_FRAMES_5S, dtype=torch.float32)
    tmp = onnx_path.with_suffix(".onnx.tmp")
    torch.onnx.export(
        wrapped, dummy, str(tmp),
        input_names=["mel"], output_names=["clip_prob", "frame_prob"],
        dynamic_axes={"mel": {0: "batch"},
                      "clip_prob": {0: "batch"},
                      "frame_prob": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    os.replace(str(tmp), str(onnx_path))
    # Quick check
    om = onnx.load(str(onnx_path))
    onnx.checker.check_model(om)
    sz = onnx_path.stat().st_size / 1e6
    print(f"  exported {onnx_path.name}  ({sz:.1f} MB)  val_auc={ckpt.get('val_auc', 0):.4f}")


for prefix in ["b0", "b1"]:
    for f in range(N_FOLDS):
        export_one(prefix, f)

print(f"ONNX export wall: {fmt_dur(time.time() - WALL_START)}")"""))

# ===================== Summary =====================
cells.append(code_cell("summary", r"""# ============================================================
# Final summary
# ============================================================
total_wall = time.time() - WALL_START
print(f"\n{'='*60}\nTOTAL WALL: {fmt_dur(total_wall)}\n{'='*60}")

# Phase 1 outputs
print("Phase 1 outputs:")
for name in ["proto_scores.npy", "tucker_scores.npy", "pseudo_labels.npy"]:
    p = WORK_DIR / name
    if p.exists():
        sz = p.stat().st_size / 1e6
        print(f"  {name}  ({sz:.1f} MB)")

# Phase 2 outputs
print("\nPhase 2 checkpoints:")
for prefix in ["b0", "b1"]:
    for f in range(N_FOLDS):
        best = CKPT_DIR / f"{prefix}_fold{f}_best.pt"
        flag = CKPT_DIR / f"{prefix}_fold{f}_done.flag"
        log  = CKPT_DIR / f"{prefix}_fold{f}_log.json"
        auc = "-"
        if best.exists():
            try:
                ck = torch.load(str(best), map_location="cpu", weights_only=False)
                auc = f"{ck.get('val_auc', 0):.4f}"
            except Exception:
                pass
        flag_s = "DONE" if flag.exists() else "PARTIAL"
        ep = "-"
        if log.exists():
            try:
                rows = json.loads(log.read_text())
                ep = str(len(rows))
            except Exception:
                pass
        print(f"  {prefix} fold{f}  {flag_s:8s}  best_auc={auc}  epochs_logged={ep}")

# Phase 3 outputs
print("\nPhase 3 ONNX:")
for prefix in ["b0", "b1"]:
    for f in range(N_FOLDS):
        p = ONNX_DIR / f"{prefix}_fold{f}.onnx"
        if p.exists():
            sz = p.stat().st_size / 1e6
            print(f"  {p.name}  ({sz:.1f} MB)")

print(f"\nDone. Time left: {fmt_dur(time_left())}")"""))


# ===================== Assemble notebook =====================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_pseudo_and_train.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Stats
import os as _os
size = _os.path.getsize(out_path)
print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
print(f"Size: {size/1024:.1f} KB")
