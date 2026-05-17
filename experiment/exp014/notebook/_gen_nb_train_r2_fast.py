"""Generate exp014 R2 FAST NB: pre-computed Perch emb cache + batch 128.

Round 2 of Multi-Iterative Noisy Student (BC2025 1st-place Babych recipe).
Speed optimization vs nb_train_r2.ipynb:
- A: Perch ONNX forward replaced with pre-computed emb cache lookup (~50% epoch time)
- C: batch 64 → 128
- Net: ~25 min/epoch (vs 60 min/epoch baseline). 25 ep in ~10.5h, fits 12h timeout

Recipe changes vs nb_train_r2.ipynb (required for Perch pre-compute):
1. Focal: random crop → 5s chunk-aligned (chunk_idx random, start = chunk_idx * 160000)
2. Mixup: audio-mixup is kept for student input, but Perch distill target is
   feature-mixup at 1536-d emb level (Babych recipe).

Other settings identical to nb_train_r2:
- Source mix: focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20
- 25 ep / AdamW lr=3e-4 / cosine + warmup 2ep
- Same model: HGNetV2-B0 + GeMFreq + 512d + AttHead + DistillHead

Run: python _gen_nb_train_r2_fast.py  ->  writes nb_train_r2_fast.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}


cells = []

# =============================================================================
# Cell 0: Header
# =============================================================================
cells.append(md_cell(r"""# exp014 R2 FAST — Pre-computed Perch cache + batch 128 (Kaggle T4x2)

Speed-optimized version of nb_train_r2.ipynb:
- **A**: Perch ONNX forward replaced with pre-computed emb cache lookup
- **C**: batch 64 -> 128

Net effect: ~25 min/epoch (vs 60 min/epoch baseline), 25 ep in ~10.5h.

## Recipe changes vs nb_train_r2 (proof tested by Babych BC2025 1st place)
1. Focal: random crop -> 5s chunk-aligned (chunk_idx randomly sampled per file)
2. Mixup: audio-mixup kept for student input, but Perch distill target is
   feature-mixup at 1536-d emb level

## Required input datasets
| Dataset | Purpose | First run | Resume |
|---|---|---|---|
| `birdclef-2026` (competition) | Audio + CSVs | Required | Required |
| `maekeso/birdclef2026-perch-emb-cache` | Pre-computed Perch emb | Required | Required |
| `maekeso/birdclef2026-exp014-pseudo-r1` | R1 pseudo CSV | Required | Required |
| `maekeso/exp014-state-r2-fast` | Resume state | Skip (1st) | Required |

## Output
- `ckpt_latest.pth` / `ckpt_ep{NN}.pth` / `ckpt_best_ns22.pth` / `ckpt_best_macro.pth`
- `history.json`
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup
# ============================================================
import subprocess, sys

# No onnxruntime needed — Perch is via pre-computed cache
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "timm>=1.0.0", "librosa", "soundfile"])

import os, time, json, gc, random, math, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

TRAIN_START = time.time()
MAX_RUNTIME_SEC = 11.0 * 3600
print(f"Train start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(TRAIN_START))}")
print(f"Max runtime: {MAX_RUNTIME_SEC/3600:.1f}h")
""", "setup"))

# =============================================================================
# Cell 2: Paths
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"
print(f"Competition: {BASE}")

TA_DIR = BASE / "train_audio"
TS_DIR_SRC = BASE / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
LABELS_PATH = BASE / "train_soundscapes_labels.csv"

# Pre-copy TS to local SSD
TS_DIR_LOCAL = Path("/kaggle/working/train_soundscapes_local")
TS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
_n_existing = sum(1 for _ in TS_DIR_LOCAL.glob("*.ogg"))
_n_src = sum(1 for _ in TS_DIR_SRC.glob("*.ogg")) if TS_DIR_SRC.exists() else 0
print(f"Pre-copy: local={_n_existing} / src={_n_src}")
if _n_src > 0 and _n_existing < _n_src:
    _t0 = time.time()
    _copied = 0
    for _f in TS_DIR_SRC.glob("*.ogg"):
        _dst = TS_DIR_LOCAL / _f.name
        if not _dst.exists():
            shutil.copy(str(_f), str(_dst))
            _copied += 1
    print(f"  Copied {_copied} in {time.time()-_t0:.1f}s")
TS_DIR = TS_DIR_LOCAL

# Perch emb cache (REQUIRED)
PERCH_CACHE_DIR = None
for p in [Path("/kaggle/input/datasets/maekeso/birdclef2026-perch-emb-cache"),
          Path("/kaggle/input/birdclef2026-perch-emb-cache")]:
    if p.exists() and (p / "emb.npy").exists():
        PERCH_CACHE_DIR = p; break
if PERCH_CACHE_DIR is None:
    # rglob fallback
    for hit in Path("/kaggle/input").rglob("emb.npy"):
        # heuristic: meta.csv must be in same dir
        if (hit.parent / "meta.csv").exists():
            PERCH_CACHE_DIR = hit.parent; break
assert PERCH_CACHE_DIR is not None, (
    "Perch emb cache not found. Attach maekeso/birdclef2026-perch-emb-cache as dataset_sources"
)
print(f"Perch cache: {PERCH_CACHE_DIR}")

# Pseudo CSV (R1 output)
PSEUDO_CSV_PATH = None
for p in [Path("/kaggle/input/datasets/maekeso/birdclef2026-exp014-pseudo-r1"),
          Path("/kaggle/input/birdclef2026-exp014-pseudo-r1")]:
    if p.exists():
        hits = list(p.rglob("pseudo_labels.csv"))
        if hits:
            PSEUDO_CSV_PATH = hits[0]; break
if PSEUDO_CSV_PATH is None:
    for hit in Path("/kaggle/input").rglob("pseudo_labels.csv"):
        PSEUDO_CSV_PATH = hit; break
assert PSEUDO_CSV_PATH is not None, "pseudo_labels.csv not found"
print(f"Pseudo CSV: {PSEUDO_CSV_PATH}")

# Resume state — check (a) state dataset (b) own kernel output
RESUME_DIR = None
RESUME_CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/exp014-state-r2-fast"),
    Path("/kaggle/input/exp014-state-r2-fast"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp014-train-r2-fast"),
    Path("/kaggle/input/birdclef2026-exp014-train-r2-fast"),
]
for p in RESUME_CANDIDATES:
    if p.exists() and (p / "ckpt_latest.pth").exists():
        RESUME_DIR = p; break
    if p.exists():
        hits = list(p.rglob("ckpt_latest.pth"))
        if hits:
            RESUME_DIR = hits[0].parent; break
print(f"Resume dir: {RESUME_DIR if RESUME_DIR else '(none, fresh start)'}")

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
""", "paths"))

# =============================================================================
# Cell 3: Config
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Config
# ============================================================
NUM_CLASSES = 234
SR = 32000

TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 256
FMIN           = 20
FMAX           = 16000

CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC

BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

N_FOLDS = 5
FOLDS = [0]
N_TOTAL_EPOCHS = 25
BATCH = 128                    # speed-up C: 64 -> 128
LR = 3e-4
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 2

AUG_PROB = 0.5
AUG_GAIN_DB_RANGE = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

USE_MIXUP = True
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
MIXUP_HARD = False

FREQ_MASK_PARAM = 25
TIME_MASK_PARAM = 30
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

MIN_SAMPLE = 20

SHARES = {"focal": 0.70, "labeled_sc": 0.10, "pseudo_sc": 0.20}
SOURCE_WEIGHTS = {
    "focal":          1.0,
    "focal_missing":  0.0,
    "labeled_sc":     1.0,
    "pseudo_sc":      0.5,
}

NUM_WORKERS = 4
PERSISTENT_WORKERS = True

print(f"Backbone: {BACKBONE}")
print(f"Batch: {BATCH} | Total epochs: {N_TOTAL_EPOCHS} | Folds: {FOLDS}")
print(f"LR: {LR} | WD: {WD} | warmup: {WARMUP_EPOCHS}ep")
print(f"Source mix: {SHARES}")
print(f"Workers: {NUM_WORKERS}")
""", "config"))

# =============================================================================
# Cell 4: Load data (incl. Perch emb cache)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Load data — focal + ss labels + pseudo CSV + Perch emb cache
# ============================================================
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES

taxonomy = pd.read_csv(TAXO_PATH)
label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                          taxonomy["class_name"].astype(str)))
TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                            if label_to_taxon.get(l, "") == t])
               for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}

# Focal metadata
train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[train_df["primary_label"].astype(str).isin(LABEL2IDX)].reset_index(drop=True)
train_df["filename"] = train_df["filename"].astype(str)
print(f"Focal train.csv: {len(train_df)} rows")

def _check_exists(fn):
    return (TA_DIR / fn).exists()
print("Checking focal file existence...")
_t0 = time.time()
train_df["exists"] = train_df["filename"].map(_check_exists)
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"  {len(train_df)} focal files exist ({time.time()-_t0:.1f}s)")
train_df["original_idx"] = np.arange(len(train_df))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fold
print(f"Focal fold distribution: {train_df['fold'].value_counts().sort_index().to_dict()}")

focal_secondary_labels = {}
for idx, row in train_df.iterrows():
    sec = row.get("secondary_labels", "")
    if pd.isna(sec) or sec in ("", "[]"):
        continue
    try:
        sec_list = eval(sec) if isinstance(sec, str) else []
    except Exception:
        continue
    valid = [s for s in sec_list if s in LABEL2IDX]
    if valid:
        focal_secondary_labels[int(row["original_idx"])] = valid
print(f"Focal secondary labels: {len(focal_secondary_labels)} files")

counts = train_df["primary_label"].value_counts()
rare_species = counts[counts < MIN_SAMPLE].index.tolist()
extra_rows = []
for sp in rare_species:
    sp_rows = train_df[train_df["primary_label"] == sp]
    n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
    for _ in range(n_copies):
        extra_rows.append(sp_rows)
n_before = len(train_df)
if extra_rows:
    train_df = pd.concat([train_df] + extra_rows, ignore_index=True)
print(f"Upsampled {len(rare_species)} rare species: {n_before} -> {len(train_df)}")

# Labeled soundscape windows
if LABELS_PATH.exists():
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    if sc_labels_raw["start"].dtype == object:
        sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    else:
        sc_labels_raw["start_sec"] = sc_labels_raw["start"].astype(int)
    sc_meta = (sc_labels_raw[["filename", "start_sec"]]
               .drop_duplicates().reset_index(drop=True))
    if "site" in sc_labels_raw.columns:
        site_map = sc_labels_raw.groupby("filename")["site"].first().to_dict()
        sc_meta["site"] = sc_meta["filename"].map(site_map).fillna("UNK")
    else:
        sc_meta["site"] = "UNK"
    Y_SC = np.zeros((len(sc_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_meta.iterrows():
        matches = sc_labels_raw[(sc_labels_raw["filename"] == row["filename"]) &
                                 (sc_labels_raw["start_sec"] == row["start_sec"])]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    print(f"SS labels: {len(sc_meta)} windows, {int(Y_SC.sum())} pos, "
          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")
    sc_files = sc_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_meta["fold"] = sc_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    non_s22_mask_sc = (sc_meta["site"].values != "S22")
else:
    sc_meta = pd.DataFrame(columns=["filename", "start_sec", "site", "fold"])
    Y_SC = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    non_s22_mask_sc = np.zeros(0, dtype=bool)

# Pseudo CSV
print(f"\nLoading pseudo CSV: {PSEUDO_CSV_PATH}")
_t0 = time.time()
pseudo_df_full = pd.read_csv(PSEUDO_CSV_PATH)
print(f"  loaded {len(pseudo_df_full)} rows in {time.time()-_t0:.1f}s")
_label_cols_in_csv = [c for c in pseudo_df_full.columns if c in LABEL2IDX]
assert len(_label_cols_in_csv) == NUM_CLASSES
Y_PSEUDO = pseudo_df_full[PRIMARY_LABELS].values.astype(np.float32)
pseudo_meta = pseudo_df_full[["filename", "start_sec"]].copy().reset_index(drop=True)
pseudo_meta["filename"] = pseudo_meta["filename"].astype(str)
pseudo_meta["start_sec"] = pseudo_meta["start_sec"].astype(float)
print(f"  Y_pseudo: {Y_PSEUDO.shape}, mean={Y_PSEUDO.mean():.4f}")

# Perch emb cache
print(f"\nLoading Perch emb cache: {PERCH_CACHE_DIR}")
_t0 = time.time()
# Full in-memory load (~1.3 GB float16); workers share via fork copy-on-write
PERCH_EMB = np.load(str(PERCH_CACHE_DIR / "emb.npy"))
print(f"  emb.npy: {PERCH_EMB.shape}, dtype={PERCH_EMB.dtype}, "
      f"{PERCH_EMB.nbytes/1e9:.2f} GB ({time.time()-_t0:.1f}s)")
PERCH_META = pd.read_csv(PERCH_CACHE_DIR / "meta.csv")
print(f"  meta.csv: {len(PERCH_META)} rows")

# Build lookup: (source, filename, chunk_idx) -> row_idx
print("Building emb lookup...")
_t0 = time.time()
EMB_LOOKUP = {}
for src, fn, ci, ri in zip(
        PERCH_META["source"].values,
        PERCH_META["filename"].values,
        PERCH_META["chunk_idx"].values.astype(int),
        PERCH_META["row_idx"].values.astype(int)):
    EMB_LOOKUP[(src, fn, ci)] = ri
print(f"  EMB_LOOKUP: {len(EMB_LOOKUP)} entries ({time.time()-_t0:.1f}s)")

# Focal file -> available chunk_idx list
FOCAL_FILE_CHUNKS = {}
for fn, ci in zip(PERCH_META[PERCH_META["source"] == "focal"]["filename"].values,
                  PERCH_META[PERCH_META["source"] == "focal"]["chunk_idx"].values.astype(int)):
    FOCAL_FILE_CHUNKS.setdefault(fn, []).append(ci)
print(f"  FOCAL_FILE_CHUNKS: {len(FOCAL_FILE_CHUNKS)} files")

print("OK data loaded")
""", "load_data"))

# =============================================================================
# Cell 5: Model
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Model — Mel + SpecAugment + DistillHead + BirdSEDModel
# ============================================================
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    def forward(self, mel):
        for _ in range(NUM_FREQ_MASKS): mel = self.freq_mask(mel)
        for _ in range(NUM_TIME_MASKS): mel = self.time_mask(mel)
        return mel


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        gap = feature_map.mean(dim=[2, 3])
        return self.proj(gap)


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
            print(f"Backbone out: {tuple(feat.shape)}  (C={self.backbone_dim})")
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.cla.weight)
        self.att.bias.data.fill_(0.)
        self.cla.bias.data.fill_(0.)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        elif return_framewise:
            return clip_logits, fw
        elif return_distill:
            return clip_logits, distill_emb
        return clip_logits


def make_model():
    m = BirdSEDModel().to(device)
    m = m.to(memory_format=torch.channels_last)
    return m

print("OK model defs ready")
""", "model"))

# =============================================================================
# Cell 6: Dataset (chunk-aligned + Perch row_idx passthrough)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Datasets — FocalDS, LabeledSCDS, PseudoScDS (chunk-aligned)
# Returns: (wav, lb, wt, mk, source, perch_row_idx_a, perch_row_idx_b, lam)
#   - ri_b = -1 and lam = 1.0 means "no mixup, use ri_a only"
#   - ri_a = -1 means "no Perch emb available, skip distill loss for this sample"
# ============================================================
import soundfile as sf
import librosa
from functools import lru_cache


@lru_cache(maxsize=128)
def _load_full_audio_cached(path_str):
    try:
        wav, sr = sf.read(path_str, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None


def _load_ogg_at_chunk(path, chunk_idx):
    '''Load 5s chunk at chunk_idx * CHUNK_SAMPLES.'''
    full = _load_full_audio_cached(str(path))
    if full is None:
        return None
    start = chunk_idx * CHUNK_SAMPLES
    end = start + CHUNK_SAMPLES
    if end <= len(full):
        return full[start:end].copy()
    out = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    avail = full[start:start + CHUNK_SAMPLES]
    out[:len(avail)] = avail
    return out


def apply_aug(w):
    if np.random.random() < AUG_PROB:
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))
    return w


class FocalDS(Dataset):
    '''Focal dataset with 5s chunk-aligned crop (no more random start jitter).'''
    def __init__(self, df, l2i, secondary_lookup=None, aug=False):
        self.df = df.reset_index(drop=True)
        self.l2i = l2i
        self.aug = aug
        self.secondary_lookup = secondary_lookup
        self.filenames = self.df["filename"].values
        self.primary = self.df["primary_label"].astype(str).values
        self.original_idx = self.df["original_idx"].values if "original_idx" in self.df.columns else None

    def __len__(self): return len(self.df)

    def _load_chunk(self, i):
        fn = self.filenames[i]
        path = TA_DIR / fn
        # Pick chunk_idx based on available chunks (from precompute cache)
        chunks_avail = FOCAL_FILE_CHUNKS.get(fn, [])
        if not chunks_avail:
            # Fallback: file not in cache; load full + crop at 0
            wav = _load_full_audio_cached(str(path))
            if wav is None:
                return None, None, -1
            chunk = wav[:TRAIN_SAMPLES]
            if len(chunk) < TRAIN_SAMPLES:
                chunk = np.pad(chunk, (0, TRAIN_SAMPLES - len(chunk)))
            row_idx = -1
        else:
            if self.aug:
                chunk_idx = int(chunks_avail[np.random.randint(len(chunks_avail))])
            else:
                chunk_idx = int(chunks_avail[0])
            chunk = _load_ogg_at_chunk(path, chunk_idx)
            if chunk is None:
                return None, None, -1
            row_idx = EMB_LOOKUP.get(("focal", fn, chunk_idx), -1)
        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        if self.primary[i] in self.l2i:
            lb[self.l2i[self.primary[i]]] = 1.0
        if self.secondary_lookup is not None and self.original_idx is not None:
            for s in self.secondary_lookup.get(int(self.original_idx[i]), []):
                if s in self.l2i: lb[self.l2i[s]] = 1.0
        return chunk, lb, row_idx

    def __getitem__(self, i):
        ch1, lb1, ri1 = self._load_chunk(i)
        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES),
                    "focal_missing", -1, -1, 1.0)

        if USE_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            ch2, lb2, ri2 = None, None, -1
            for _ in range(3):
                j = np.random.randint(len(self.df))
                ch2, lb2, ri2 = self._load_chunk(j)
                if ch2 is not None: break
            if ch2 is not None:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                if self.aug: ch_mix = apply_aug(ch_mix)
                lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                return (torch.from_numpy(ch_mix).unsqueeze(0),
                        torch.from_numpy(lb.astype(np.float32)),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES),
                        "focal", int(ri1), int(ri2), lam)

        if self.aug: ch1 = apply_aug(ch1)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES),
                "focal", int(ri1), -1, 1.0)


class LabeledSCDS(Dataset):
    def __init__(self, Y, sc_df, aug=False):
        self.Y = Y
        self.df = sc_df.reset_index(drop=True)
        self.aug = aug
        self.filenames = self.df["filename"].astype(str).values
        self.start_secs = self.df["start_sec"].astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        fn = self.filenames[i]
        fn_ogg = fn if fn.endswith(".ogg") else fn + ".ogg"
        path = TS_DIR / fn_ogg
        start_sec = int(self.start_secs[i])
        chunk_idx = start_sec // CHUNK_SEC
        wav = _load_ogg_at_chunk(path, chunk_idx)
        if wav is None:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        if self.aug:
            wav = apply_aug(wav)
        row_idx = EMB_LOOKUP.get(("ss", fn_ogg, chunk_idx), -1)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES),
                "labeled_sc", int(row_idx), -1, 1.0)


class PseudoScDS(Dataset):
    def __init__(self, meta_df, Y_soft, audio_dir, aug=False):
        self.meta = meta_df.reset_index(drop=True)
        self.Y = Y_soft.astype(np.float32)
        self.audio_dir = Path(audio_dir)
        self.aug = aug
        self.filenames = self.meta["filename"].values
        self.start_secs = self.meta["start_sec"].values

    def __len__(self): return len(self.meta)

    def __getitem__(self, i):
        fn = str(self.filenames[i])
        fn_ogg = fn if fn.endswith(".ogg") else fn + ".ogg"
        path = self.audio_dir / fn_ogg
        start_sec = float(self.start_secs[i])
        chunk_idx = int(start_sec // CHUNK_SEC)
        wav = _load_ogg_at_chunk(path, chunk_idx)
        if wav is None:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        if self.aug:
            wav = apply_aug(wav)
        row_idx = EMB_LOOKUP.get(("ss", fn_ogg, chunk_idx), -1)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i]),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES),
                "pseudo_sc", int(row_idx), -1, 1.0)


class MixSamp(torch.utils.data.Sampler):
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.rng = np.random.default_rng(seed)
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs:
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
    def __len__(self): return self.nst
    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0: continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch


def collate_m(batch):
    return (torch.stack([b[0] for b in batch]),                             # wav
            torch.stack([b[1] for b in batch]),                             # lb
            torch.stack([b[2] for b in batch]),                             # wt
            torch.stack([b[3] for b in batch]),                             # mk
            [b[4] for b in batch],                                          # source
            torch.tensor([b[5] for b in batch], dtype=torch.long),          # ri_a
            torch.tensor([b[6] for b in batch], dtype=torch.long),          # ri_b
            torch.tensor([b[7] for b in batch], dtype=torch.float32))       # lam


def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)

print("OK datasets ready (chunk-aligned, returns Perch row_idx)")
""", "dataset"))

# =============================================================================
# Cell 7: Eval helpers
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Eval
# ============================================================
def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (np.mean(aucs) if aucs else float("nan")), len(aucs)


def full_eval(y_true, y_pred, ns22_mask, taxon_masks):
    r = {}
    a, n = compute_macro_auc(y_true, y_pred)
    r["macro_auc_all"] = round(float(a) if not np.isnan(a) else 0.0, 4)
    r["n_all"] = n
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22_mask)
    r["non_s22_macro"] = round(float(a) if not np.isnan(a) else 0.0, 4)
    r["n_ns22"] = n
    per_taxon = {}
    for t, cm in taxon_masks.items():
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22_mask, class_mask=cm)
        per_taxon[t] = round(float(a) if not np.isnan(a) else 0.0, 4)
    r["per_taxon"] = per_taxon
    return r


def _load_val_waveforms(val_sc_df):
    wavs = []
    for _, row in val_sc_df.iterrows():
        fn = str(row["filename"])
        fn_ogg = fn if fn.endswith(".ogg") else fn + ".ogg"
        start_sec = int(row["start_sec"])
        chunk_idx = start_sec // CHUNK_SEC
        wav = _load_ogg_at_chunk(TS_DIR / fn_ogg, chunk_idx)
        if wav is None:
            wav = np.zeros(VAL_SAMPLES, dtype=np.float32)
        if len(wav) < VAL_SAMPLES:
            wav = np.pad(wav, (0, VAL_SAMPLES - len(wav)))
        else:
            wav = wav[:VAL_SAMPLES]
        wavs.append(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0))
    return wavs


def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=128):
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    with torch.no_grad():
        for s in range(0, len(wav_list), batch_size):
            batch = torch.stack(wav_list[s:s+batch_size]).to(device)
            mel = mel_transform(batch)
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            mel = mel.to(memory_format=torch.channels_last)
            with autocast():
                clip_logits, framewise = model(mel, return_framewise=True)
                frame_max = framewise.max(dim=1).values
                p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()
                p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
                p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip); preds_fmax.append(p_fmax); preds_blend.append(p_blend)
    return {"clip": np.concatenate(preds_clip),
            "fmax": np.concatenate(preds_fmax),
            "blend": np.concatenate(preds_blend)}

print("OK eval helpers ready")
""", "eval"))

# =============================================================================
# Cell 8: build_active_datasets
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: build_active_datasets
# ============================================================
def build_active_datasets(fold_k):
    items = []
    fds = FocalDS(train_df[train_df["fold"] != fold_k],
                  LABEL2IDX, secondary_lookup=focal_secondary_labels, aug=True)
    items.append(("focal", fds, len(fds)))
    if len(sc_meta) > 0:
        vm = sc_meta["fold"].values == fold_k
        sc_train_df = sc_meta[~vm].reset_index(drop=True)
        Y_tr = Y_SC[~vm]
        sds = LabeledSCDS(Y_tr, sc_train_df, aug=True)
        items.append(("labeled_sc", sds, len(sds)))
    pds = PseudoScDS(pseudo_meta, Y_PSEUDO, TS_DIR, aug=True)
    items.append(("pseudo_sc", pds, len(pds)))
    return items

print("OK build_active_datasets ready")
""", "build_active"))

# =============================================================================
# Cell 9: resume_or_init
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 9: resume_or_init
# ============================================================
FOLD_K = FOLDS[0]
print(f"This run trains fold {FOLD_K}")

active = build_active_datasets(FOLD_K)
NAMES, DATASETS, SIZES = zip(*active)
NAMES, DATASETS, SIZES = list(NAMES), list(DATASETS), list(SIZES)
print(f"Streams: {dict(zip(NAMES, SIZES))}")

mds = ConcatDataset(DATASETS)
N_STEPS_PER_EP = max(100, int(sum(SIZES) / BATCH))
print(f"steps/epoch: {N_STEPS_PER_EP}")

model = make_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scaler = GradScaler()

warmup_steps = N_STEPS_PER_EP * WARMUP_EPOCHS
total_steps  = N_STEPS_PER_EP * N_TOTAL_EPOCHS
warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=MIN_LR)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

start_epoch = 0
best_ns22 = -1.0
best_macro = -1.0
history = []
resumed = False

if RESUME_DIR is not None:
    ckpt_latest = RESUME_DIR / "ckpt_latest.pth"
    if ckpt_latest.exists():
        try:
            state = torch.load(str(ckpt_latest), map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt_latest), map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"])
        best_ns22 = float(state.get("best_ns22", -1.0))
        best_macro = float(state.get("best_macro", -1.0))
        history = state.get("history", [])
        resumed = True
        print(f"=== RESUMED from epoch {start_epoch}/{N_TOTAL_EPOCHS} ===")
        print(f"    best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}")
        for fname in ["history.json"]:
            src = RESUME_DIR / fname
            if src.exists():
                shutil.copy(str(src), str(OUT_DIR / fname))
    else:
        print(f"No ckpt at {ckpt_latest}, starting fresh")
else:
    print("No state dataset attached, starting fresh R2-fast")

if start_epoch >= N_TOTAL_EPOCHS:
    print(f"\nAlready trained to epoch {start_epoch}/{N_TOTAL_EPOCHS}. Nothing to do this session.")
""", "resume_or_init"))

# =============================================================================
# Cell 10: Val prep
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 10: Val prep
# ============================================================
if len(sc_meta) > 0:
    vm = sc_meta["fold"].values == FOLD_K
    val_sc_df = sc_meta[vm].reset_index(drop=True)
    Y_val = Y_SC[vm]
    ns22_val = non_s22_mask_sc[vm]
    print(f"Val: {len(val_sc_df)} SS windows for fold {FOLD_K}")
    print(f"  loading val waveforms...")
    _t0 = time.time()
    val_wavs = _load_val_waveforms(val_sc_df)
    print(f"  loaded in {time.time()-_t0:.1f}s")
else:
    val_sc_df = pd.DataFrame()
    Y_val = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    ns22_val = np.zeros(0, dtype=bool)
    val_wavs = []
""", "val_prep"))

# =============================================================================
# Cell 11: Train loop (Perch via cache lookup, feature-mixup)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 11: Train loop — Perch via cache lookup + feature-mixup
# ============================================================
mel_transform = MelSpecTransform().to(device)
spec_augment = SpecAugment().to(device)

epoch_times = []
trained_this_session = 0
ep = start_epoch

for epoch in range(start_epoch, N_TOTAL_EPOCHS):
    elapsed = time.time() - TRAIN_START
    if epoch_times:
        est_next = max(epoch_times[-3:])
        if elapsed + est_next * 1.3 > MAX_RUNTIME_SEC:
            print(f"\n[stop] Time budget exhausted before epoch {epoch+1}/{N_TOTAL_EPOCHS}")
            break

    ep_start = time.time()
    model.train()

    smp = MixSamp(SIZES, NAMES, SHARES, BATCH, N_STEPS_PER_EP, seed=42 + epoch)
    train_loader = DataLoader(
        mds, batch_sampler=smp, collate_fn=collate_m,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        pin_memory=True, prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
    for batch_idx, (wav, lb, wt, mk, sr, ri_a, ri_b, lam) in enumerate(train_loader):
        wav = wav.to(device, non_blocking=True)
        lb  = lb.to(device, non_blocking=True)
        wt  = wt.to(device, non_blocking=True)
        mk  = mk.to(device, non_blocking=True)
        sw = mk_sw(sr).to(device, non_blocking=True)

        # === Perch emb via cache lookup (feature-mixup) ===
        # ri_a >= 0 means valid; ri_a < 0 means skip distill for that sample.
        # ri_b >= 0 means mixup'd; use lam * emb_a + (1-lam) * emb_b.
        with torch.no_grad():
            ri_a_np = ri_a.numpy()
            ri_b_np = ri_b.numpy()
            valid_a = (ri_a_np >= 0)
            valid_b = (ri_b_np >= 0)

            ri_a_safe = ri_a_np.copy(); ri_a_safe[~valid_a] = 0
            ri_b_safe = ri_b_np.copy(); ri_b_safe[~valid_b] = 0

            emb_a_np = PERCH_EMB[ri_a_safe]                                     # (B, 1536) float16
            emb_a = torch.from_numpy(emb_a_np.astype(np.float32)).to(device)
            if valid_b.any():
                emb_b_np = PERCH_EMB[ri_b_safe]
                emb_b = torch.from_numpy(emb_b_np.astype(np.float32)).to(device)
                lam_dev = lam.to(device).unsqueeze(1)                            # (B, 1)
                has_b = torch.from_numpy(valid_b).to(device).unsqueeze(1)        # (B, 1)
                perch_emb = torch.where(has_b,
                                         lam_dev * emb_a + (1.0 - lam_dev) * emb_b,
                                         emb_a)
            else:
                perch_emb = emb_a
            valid_mask = torch.from_numpy(valid_a).to(device).float()            # (B,)

        with torch.no_grad():
            mel = mel_transform(wav)
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            mel = spec_augment(mel)
            mel = mel.to(memory_format=torch.channels_last)

        with autocast():
            clip_logits, framewise, distill_emb = model(mel, return_framewise=True,
                                                          return_distill=True)
            frame_max_logits = framewise.max(dim=1).values
            bce_clip  = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
            bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
            bce = 0.5 * bce_clip + 0.5 * bce_frame
            ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
            cls_loss = (ps * sw).mean()

            # Masked MSE distill loss (skip invalid Perch row_idx samples)
            diff_sq = (distill_emb - perch_emb) ** 2                               # (B, 1536)
            mse_per_sample = diff_sq.mean(dim=1)                                   # (B,)
            denom = valid_mask.sum().clamp(min=1.0)
            distill_loss = (mse_per_sample * valid_mask).sum() / denom

            loss = cls_loss + ALPHA_DISTILL * distill_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        el += float(loss.item())
        el_cls += float(cls_loss.item())
        el_dist += float(distill_loss.item())
        nb_count += 1

        if batch_idx % 20 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            valid_frac = valid_mask.mean().item()
            print(f"    ep{epoch+1:02d} batch {batch_idx:4d}/{N_STEPS_PER_EP}  "
                  f"loss={loss.item():.4f}  cls={cls_loss.item():.4f}  "
                  f"dist={distill_loss.item():.4f}  lr={cur_lr:.2e}  "
                  f"valid={valid_frac:.2f}", flush=True)

    train_loss_avg = el / max(nb_count, 1)
    cls_loss_avg = el_cls / max(nb_count, 1)
    dist_loss_avg = el_dist / max(nb_count, 1)

    if len(val_wavs) > 0:
        val_preds_dict = _predict_from_waveforms(model, mel_transform, val_wavs)
        val_preds = val_preds_dict["blend"]
        r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
        val_metrics = {
            "ns22": r["non_s22_macro"],
            "macro": r["macro_auc_all"],
            "per_taxon": r["per_taxon"],
        }
    else:
        val_metrics = {"ns22": float("nan"), "macro": float("nan"), "per_taxon": {}}

    ep_elapsed = time.time() - ep_start
    epoch_times.append(ep_elapsed)
    trained_this_session += 1
    ep = epoch + 1

    state_to_save = {
        "epoch": ep,
        "fold": FOLD_K,
        "n_total_epochs": N_TOTAL_EPOCHS,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_ns22": best_ns22,
        "best_macro": best_macro,
        "history": history,
    }
    torch.save(state_to_save, OUT_DIR / "ckpt_latest.pth")
    torch.save(state_to_save, OUT_DIR / f"ckpt_ep{ep:02d}.pth")

    if not math.isnan(val_metrics["ns22"]) and val_metrics["ns22"] > best_ns22:
        best_ns22 = val_metrics["ns22"]
        state_to_save["best_ns22"] = best_ns22
        torch.save(state_to_save, OUT_DIR / "ckpt_best_ns22.pth")
    if not math.isnan(val_metrics["macro"]) and val_metrics["macro"] > best_macro:
        best_macro = val_metrics["macro"]
        state_to_save["best_macro"] = best_macro
        torch.save(state_to_save, OUT_DIR / "ckpt_best_macro.pth")

    cur_lr = optimizer.param_groups[0]["lr"]
    history.append({
        "epoch": ep,
        "train_loss": round(train_loss_avg, 5),
        "cls_loss": round(cls_loss_avg, 5),
        "dist_loss": round(dist_loss_avg, 5),
        "val_ns22": val_metrics["ns22"],
        "val_macro": val_metrics["macro"],
        "val_per_taxon": val_metrics["per_taxon"],
        "lr": cur_lr,
        "elapsed_sec": round(ep_elapsed, 1),
    })
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    total_elapsed = time.time() - TRAIN_START
    print(f"\n=== Ep {ep}/{N_TOTAL_EPOCHS}: "
          f"train_loss={train_loss_avg:.4f} cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
          f"val_ns22={val_metrics['ns22']:.4f} val_macro={val_metrics['macro']:.4f} "
          f"({ep_elapsed:.0f}s, total {total_elapsed/60:.1f}min) ===\n")

    gc.collect()
    torch.cuda.empty_cache()

trained_up_to = ep
print(f"\n=== Session: trained epochs {start_epoch+1}-{trained_up_to} ===")
print(f"    Cumulative: {trained_up_to}/{N_TOTAL_EPOCHS} epochs")

del mel_transform, spec_augment
gc.collect()
torch.cuda.empty_cache()
""", "train_loop"))

# =============================================================================
# Cell 12: Upload state
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 12: Upload state to maekeso/exp014-state-r2-fast
# ============================================================
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

DATASET_USER = "maekeso"
DATASET_SLUG = "exp014-state-r2-fast"
DATASET_TITLE = "exp014 R2-fast training state"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_copied = 0
    for f in Path("/kaggle/working").glob("*"):
        if not f.is_file(): continue
        if f.suffix not in {".pth", ".json", ".txt"}: continue
        shutil.copy(str(f), str(td / f.name))
        n_copied += 1
    print(f"Staged {n_copied} files for upload")
    for p in sorted(td.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.2f} MB")

    meta = {
        "title": DATASET_TITLE,
        "id": f"{DATASET_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = f"R2-fast epoch {trained_up_to}/{N_TOTAL_EPOCHS}"
    uploaded = False
    try:
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"OK Uploaded ({version_notes})")
        uploaded = True
    except Exception as e:
        msg = str(e)
        print(f"  dataset_create_version error: {msg[:300]}")
        if "not found" in msg.lower() or "404" in msg or "Could not find dataset" in msg:
            try:
                api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
                print(f"OK Created {DATASET_USER}/{DATASET_SLUG}")
                uploaded = True
            except Exception as e2:
                print(f"  dataset_create_new error: {str(e2)[:300]}")
    if not uploaded:
        print("\nUpload failed — files in /kaggle/working/ for resume via kernel_sources")
""", "upload_state"))

# =============================================================================
# Cell 13: Summary
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 13: Session summary
# ============================================================
total_time = time.time() - TRAIN_START
print(f"\n{'='*60}")
print(f"R2-fast Session summary")
print(f"{'='*60}")
print(f"  Resume:               {'RESUMED' if resumed else 'FRESH'}")
print(f"  Resumed from epoch:   {start_epoch}")
print(f"  Trained this session: {trained_this_session} epoch(s)")
print(f"  Cumulative:           {trained_up_to}/{N_TOTAL_EPOCHS} epochs")
print(f"  Best ns22:            {best_ns22:.4f}")
print(f"  Best macro:           {best_macro:.4f}")
print(f"  Session time:         {total_time/60:.1f} min")
if trained_up_to >= N_TOTAL_EPOCHS:
    print(f"\n  >>> R2-fast COMPLETE <<<")
else:
    print(f"\n  ... continue at ep {trained_up_to}/{N_TOTAL_EPOCHS}")
""", "summary"))

# =============================================================================
# Assemble
# =============================================================================
nb_out = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb_train_r2_fast.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
