"""Generate exp015 NB: self-resuming ConvNeXt-Pico SED training on Kaggle T4x2.

Design philosophy:
- Single Kaggle NB. Each Save & Run All trains as many epochs as fits in ~11h.
- Saves full state (model + optimizer + scheduler + scaler + epoch + best + history)
  to a persistent Kaggle Dataset (`maekeso/exp015-state`).
- Next Save & Run All auto-resumes from `ckpt_latest.pth` in that dataset.
- After enough sessions, training completes (25 epochs total).

Architecture (no competitor outputs):
- Backbone: ConvNeXt-Pico (timm, ImageNet pretrained, allowed)
- Perch v2 teacher (Google's model, allowed)
- BC2026 competition data only
- From-scratch SED head (GeMFreq + bottleneck + attention) — Tucker-recipe-inspired
  but our own implementation. No Tucker SED weights, no konbu17, no 0.94+ ONNX.

Run: python _gen_nb_train.py  ->  writes nb_train.ipynb
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
cells.append(md_cell(r"""# exp015 — Self-resuming ConvNeXt-Pico SED training (Kaggle T4x2)

**Goal:** Train ConvNeXt-Pico SED with Perch v2 distillation, 25 epochs total, using
multiple Save & Run All sessions on Kaggle.

## How self-resume works
1. First run: `maekeso/exp015-state` Kaggle Dataset doesn't exist yet → start fresh
2. End of each run: save model + optimizer + scheduler + scaler + epoch + best
   metrics + full history to `/kaggle/working/ckpt_latest.pth`, then upload all
   `/kaggle/working/*.pth` + `*.json` to `maekeso/exp015-state` (new version)
3. Next Save & Run All: attach `maekeso/exp015-state` as input dataset → cell
   `resume_or_init` finds `ckpt_latest.pth` and continues from where we stopped
4. After ~3 sessions (depending on epoch time), training finishes at epoch 25

## Constraints
- **No competitor outputs.** Only ImageNet pretrained timm backbone (ConvNeXt-Pico)
  + Google's Perch v2 ONNX teacher + BC2026 competition data.
- **Kaggle T4x2, 12h limit.** Stop at 11h to leave buffer for ckpt + dataset upload.
- **Maximum speed:** AMP fp16, num_workers=4, persistent_workers, channels_last.

## Required input datasets (attach when pushing)
| Dataset | Purpose | First run | Resume |
|---|---|---|---|
| `birdclef-2026` (competition) | Audio + CSVs | Required | Required |
| `tuckerarrants/perch-v2-no-dft-onnx` | Perch v2 ONNX (Google model) | Required | Required |
| `maekeso/exp015-state` | Resume state | Skip (doesn't exist) | Required |

## Output
- `ckpt_latest.pth` — full state for resume
- `ckpt_ep{NN}.pth` — per-epoch checkpoint
- `ckpt_best_ns22.pth` / `ckpt_best_macro.pth` — best by val metric
- `history.json` — full training log
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — install, imports, paths, timer
# ============================================================
import subprocess, sys

# Install minimal deps. onnxruntime-gpu for Perch teacher inference.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "timm>=1.0.0", "onnxruntime-gpu", "librosa", "soundfile"])

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

# Reproducibility
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

# ===== Time budget =====
TRAIN_START = time.time()
MAX_RUNTIME_SEC = 11.0 * 3600   # 11h safety margin (Kaggle T4x2 GPU limit = 12h)
print(f"Train start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(TRAIN_START))}")
print(f"Max runtime: {MAX_RUNTIME_SEC/3600:.1f}h (will stop before this)")
""", "setup"))

# =============================================================================
# Cell 2: Paths
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths — locate competition data, Perch ONNX, resume state
# ============================================================
# BC2026 competition data
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"
print(f"Competition: {BASE}")

TA_DIR = BASE / "train_audio"
TS_DIR = BASE / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
LABELS_PATH = BASE / "train_soundscapes_labels.csv"
print(f"  train_audio: {TA_DIR.exists()}")
print(f"  train_soundscapes: {TS_DIR.exists()}")

# Perch v2 ONNX (Google's model, allowed)
_perch_cands = list(Path("/kaggle/input").rglob("perch_v2*.onnx"))
assert _perch_cands, "Perch v2 ONNX not found under /kaggle/input"
PERCH_PATH = _perch_cands[0]
print(f"Perch ONNX: {PERCH_PATH}")

# Resume state (Kaggle Dataset) — may not exist on first run
RESUME_DIR = None
for p in [Path("/kaggle/input/datasets/maekeso/exp015-state"),
          Path("/kaggle/input/exp015-state")]:
    if p.exists():
        RESUME_DIR = p; break
print(f"Resume dir: {RESUME_DIR if RESUME_DIR else '(none, fresh start)'}")

# Working dir
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
""", "paths"))

# =============================================================================
# Cell 3: Config
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Config — hyperparameters
# ============================================================
NUM_CLASSES = 234
SR = 32000

# Duration / Mel
TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 256
FMIN           = 20
FMAX           = 16000

# Model
BACKBONE = "convnext_pico.d1_in1k"

# Perch distillation
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

# Training
N_FOLDS = 5
FOLDS = [0]                  # 1-fold for now; expand later by editing this list
N_TOTAL_EPOCHS = 25
BATCH = 64
LR = 3e-4
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 2

# Augmentation
AUG_PROB = 0.5
AUG_GAIN_DB_RANGE = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

# Mixup
USE_MIXUP = True
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
MIXUP_HARD = False           # soft mixup (label blending)

# SpecAugment
FREQ_MASK_PARAM = 25
TIME_MASK_PARAM = 30
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

# Upsampling
MIN_SAMPLE = 20

# Source mixing
SHARES = {"focal": 0.85, "sc": 0.15}
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "sc": 1.0}

# Data loader
NUM_WORKERS = 4
PERSISTENT_WORKERS = True

print(f"Backbone: {BACKBONE}")
print(f"Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
print(f"Distill: ON (alpha={ALPHA_DISTILL})")
print(f"Batch: {BATCH} | Total epochs: {N_TOTAL_EPOCHS} | Folds: {FOLDS}")
print(f"LR: {LR} | WD: {WD} | warmup: {WARMUP_EPOCHS}ep")
print(f"Workers: {NUM_WORKERS}")
""", "config"))

# =============================================================================
# Cell 4: Load data
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Load data — train.csv, taxonomy, labeled SS, fold assignment
# ============================================================
# Label ordering from sample_submission
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
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

# Filter to files that actually exist on disk (some XC/iNat may be missing)
def _check_exists(fn):
    return (TA_DIR / fn).exists()
print("Checking focal file existence...")
_t0 = time.time()
train_df["exists"] = train_df["filename"].map(_check_exists)
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"  {len(train_df)} focal files exist ({time.time()-_t0:.1f}s)")
train_df["original_idx"] = np.arange(len(train_df))

# Focal: StratifiedKFold by species
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fold
print(f"Focal fold distribution: {train_df['fold'].value_counts().sort_index().to_dict()}")

# Secondary labels (parse list-string)
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

# Upsample rare species
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
print(f"Upsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): {n_before} -> {len(train_df)}")

# --- Labeled soundscape windows ---
if LABELS_PATH.exists():
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    # start_sec may be a timedelta string
    if sc_labels_raw["start"].dtype == object:
        sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    else:
        sc_labels_raw["start_sec"] = sc_labels_raw["start"].astype(int)

    # Build SS window meta: one row per (filename, start_sec) seen in labels
    sc_meta = (sc_labels_raw[["filename", "start_sec"]]
               .drop_duplicates()
               .reset_index(drop=True))
    # Add 'site' col if available
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
    print(f"Soundscape labels: {len(sc_meta)} windows, {int(Y_SC.sum())} positives, "
          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")

    # SS GroupKFold by filename
    sc_files = sc_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_meta["fold"] = sc_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    print(f"SS fold distribution: {sc_meta['fold'].value_counts().sort_index().to_dict()}")

    # Non-S22 mask (S22 has known label noise)
    non_s22_mask_sc = (sc_meta["site"].values != "S22")
    print(f"Non-S22: {non_s22_mask_sc.sum()}/{len(sc_meta)}")
else:
    print("LABELS_PATH missing — labeled SS source disabled")
    sc_meta = pd.DataFrame(columns=["filename", "start_sec", "site", "fold"])
    Y_SC = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    non_s22_mask_sc = np.zeros(0, dtype=bool)

print("OK data loaded")
""", "load_data"))

# =============================================================================
# Cell 5: Model
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Model — Mel + SpecAugment + Perch teacher + DistillHead + BirdSEDModel
# ============================================================
import onnxruntime as ort

# ---- Mel spec ----
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

# ---- SpecAugment ----
class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    def forward(self, mel):
        for _ in range(NUM_FREQ_MASKS): mel = self.freq_mask(mel)
        for _ in range(NUM_TIME_MASKS): mel = self.time_mask(mel)
        return mel

# ---- Perch teacher (frozen ONNX) ----
class PerchTeacher:
    def __init__(self, onnx_path, device_str="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device_str == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._embed_idx = None
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self._embed_idx = i; break
        if self._embed_idx is None:
            self._embed_idx = 1
        print(f"PerchTeacher: embed_idx={self._embed_idx}, providers={self.session.get_providers()}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()

# ---- DistillHead: GAP + Linear -> Perch emb ----
class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        gap = feature_map.mean(dim=[2, 3])
        return self.proj(gap)

# ---- GeM freq pool ----
class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        # x: (B, C, F, T) -> (B, C, T)
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)

# ---- Main SED model ----
class BirdSEDModel(nn.Module):
    '''SED model: ConvNeXt-Pico backbone + GeMFreq + 512-d bottleneck + attention head
    + optional Perch distillation branch (stop-grad on backbone for SED head).

    The backbone trains primarily from the distillation MSE (Perch teacher),
    while the SED head learns on detached backbone features. This avoids the
    cls loss fighting the distill loss for backbone weights.
    '''
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
# Cell 6: Dataset
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Datasets — FocalDS, LabeledSCDS, augmentation, samplers
# ============================================================
import soundfile as sf
import librosa

def _load_ogg(path, n_samples_target, start_sec=None):
    '''Load ogg with optional seek. Returns float32 numpy in [-1, 1] of length n_samples_target.'''
    try:
        if start_sec is not None:
            wav, sr = sf.read(str(path),
                              start=int(start_sec * SR),
                              frames=n_samples_target,
                              dtype="float32", always_2d=False)
        else:
            wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None

def extract_chunk_np(waveform, start_sample, n_samples):
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[start_sample:start_sample + n_samples]

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
    '''Focal recording dataset. Reads .ogg, random crop, applies mixup.'''
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
        # Load full file (focal recordings are short, typically 10-60s)
        wav = _load_ogg(path, n_samples_target=None)
        if wav is None:
            return None, None
        if self.aug and len(wav) > TRAIN_SAMPLES:
            start = np.random.randint(0, len(wav) - TRAIN_SAMPLES + 1)
        else:
            start = 0
        chunk = extract_chunk_np(wav, start, TRAIN_SAMPLES)
        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        if self.primary[i] in self.l2i:
            lb[self.l2i[self.primary[i]]] = 1.0
        if self.secondary_lookup is not None and self.original_idx is not None:
            for s in self.secondary_lookup.get(int(self.original_idx[i]), []):
                if s in self.l2i: lb[self.l2i[s]] = 1.0
        return chunk, lb

    def __getitem__(self, i):
        ch1, lb1 = self._load_chunk(i)
        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal_missing")

        # Focal-Focal Mixup
        if USE_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            ch2, lb2 = None, None
            for _ in range(3):
                j = np.random.randint(len(self.df))
                ch2, lb2 = self._load_chunk(j)
                if ch2 is not None: break
            if ch2 is not None:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                if self.aug: ch_mix = apply_aug(ch_mix)
                lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                return (torch.from_numpy(ch_mix).unsqueeze(0),
                        torch.from_numpy(lb.astype(np.float32)),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")

        if self.aug: ch1 = apply_aug(ch1)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")


class LabeledSCDS(Dataset):
    '''Labeled soundscape window dataset. Reads 5s windows directly via sf.read.'''
    def __init__(self, Y, sc_df, aug=False):
        self.Y = Y
        self.df = sc_df.reset_index(drop=True)
        self.aug = aug
        self.filenames = self.df["filename"].astype(str).values
        self.start_secs = self.df["start_sec"].astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        fn = self.filenames[i]
        if not fn.endswith(".ogg"): fn = fn + ".ogg"
        path = TS_DIR / fn
        wav = _load_ogg(path, n_samples_target=TRAIN_SAMPLES,
                         start_sec=float(self.start_secs[i]))
        if wav is None:
            wav = np.zeros(TRAIN_SAMPLES, dtype=np.float32)
        if len(wav) < TRAIN_SAMPLES:
            wav = np.pad(wav, (0, TRAIN_SAMPLES - len(wav)))
        else:
            wav = wav[:TRAIN_SAMPLES]
        if self.aug:
            wav = apply_aug(wav)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "sc")


class MixSamp(torch.utils.data.Sampler):
    '''Multi-source batch sampler controlling per-source ratios per batch.'''
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
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            [b[4] for b in batch])

def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)

print("OK datasets ready")
""", "dataset"))

# =============================================================================
# Cell 7: Eval helpers
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Eval — AUC computation + validation predictor
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
        if not fn.endswith(".ogg"): fn = fn + ".ogg"
        wav = _load_ogg(TS_DIR / fn, n_samples_target=VAL_SAMPLES,
                         start_sec=float(row["start_sec"]))
        if wav is None or len(wav) == 0:
            wav = np.zeros(VAL_SAMPLES, dtype=np.float32)
        if len(wav) < VAL_SAMPLES:
            wav = np.pad(wav, (0, VAL_SAMPLES - len(wav)))
        else:
            wav = wav[:VAL_SAMPLES]
        wavs.append(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0))
    return wavs


def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
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
# Cell 8: Build active datasets (per-fold)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: build_active_datasets per fold
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
        items.append(("sc", sds, len(sds)))
    return items

print("OK build_active_datasets ready")
""", "build_active"))

# =============================================================================
# Cell 9: resume_or_init  -- KEY CELL
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 9: resume_or_init — load ckpt_latest.pth from RESUME_DIR if available
# ============================================================
# We support only single-fold resume in this NB (FOLDS[0]).
# To train multiple folds, run the NB once per fold list (edit FOLDS, change SLUG).
FOLD_K = FOLDS[0]
print(f"This run trains fold {FOLD_K}")

# Determine sizes / steps before constructing optimizer/scheduler
active = build_active_datasets(FOLD_K)
NAMES, DATASETS, SIZES = zip(*active)
NAMES, DATASETS, SIZES = list(NAMES), list(DATASETS), list(SIZES)
print(f"Streams: {dict(zip(NAMES, SIZES))}")

mds = ConcatDataset(DATASETS)
N_STEPS_PER_EP = max(100, int(sum(SIZES) / BATCH))
print(f"steps/epoch: {N_STEPS_PER_EP}")

# Build model + optimizer + scheduler + scaler
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
            # older torch without weights_only kw
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
        print(f"    best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}, history_len={len(history)}")

        # Mirror history.json from resume dir into working dir, in case it was saved separately
        for fname in ["history.json"]:
            src = RESUME_DIR / fname
            if src.exists():
                shutil.copy(str(src), str(OUT_DIR / fname))
                print(f"    pre-copied {fname} from RESUME_DIR")
    else:
        print(f"No ckpt at {ckpt_latest}, starting fresh")
else:
    print("No exp015-state dataset attached, starting fresh")

if start_epoch >= N_TOTAL_EPOCHS:
    print(f"\nAlready trained to epoch {start_epoch}/{N_TOTAL_EPOCHS}. Nothing to do this session.")
""", "resume_or_init"))

# =============================================================================
# Cell 10: Validation prep (load wavs + ns22 + targets for FOLD_K)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 10: Val prep — load val waveforms + Y_val + ns22 once
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
    print("No labeled SS for validation; metrics will be NaN")
""", "val_prep"))

# =============================================================================
# Cell 11: Train loop with time-aware stopping
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 11: Training loop — time-aware self-stopping
# ============================================================
mel_transform = MelSpecTransform().to(device)
spec_augment = SpecAugment().to(device)
perch_teacher = PerchTeacher(PERCH_PATH, "cuda" if torch.cuda.is_available() else "cpu") \
                if USE_PERCH_DISTILL else None

epoch_times = []
trained_this_session = 0
ep = start_epoch  # for end-of-session summary

for epoch in range(start_epoch, N_TOTAL_EPOCHS):
    # Time check before starting epoch
    elapsed = time.time() - TRAIN_START
    if epoch_times:
        est_next = max(epoch_times[-3:])
        if elapsed + est_next * 1.3 > MAX_RUNTIME_SEC:
            print(f"\n[stop] Time budget exhausted before epoch {epoch+1}/{N_TOTAL_EPOCHS}")
            print(f"  elapsed={elapsed/60:.1f}min, est_next={est_next:.0f}s, "
                  f"projected_end={(elapsed + est_next * 1.3)/60:.1f}min > "
                  f"max={MAX_RUNTIME_SEC/60:.0f}min")
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
    for batch_idx, (wav, lb, wt, mk, sr) in enumerate(train_loader):
        wav, lb, wt, mk = wav.to(device, non_blocking=True), lb.to(device, non_blocking=True), \
                          wt.to(device, non_blocking=True), mk.to(device, non_blocking=True)
        sw = mk_sw(sr).to(device, non_blocking=True)

        with torch.no_grad():
            mel = mel_transform(wav)
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            mel = spec_augment(mel)
            mel = mel.to(memory_format=torch.channels_last)

        with autocast():
            if USE_PERCH_DISTILL:
                clip_logits, framewise, distill_emb = model(mel, return_framewise=True,
                                                              return_distill=True)
            else:
                clip_logits, framewise = model(mel, return_framewise=True)

            frame_max_logits = framewise.max(dim=1).values
            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
            bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
            bce = 0.5 * bce_clip + 0.5 * bce_frame
            ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
            cls_loss = (ps * sw).mean()

            if USE_PERCH_DISTILL and perch_teacher is not None:
                with torch.no_grad():
                    wav_5s = wav.squeeze(1)
                    N = wav_5s.shape[1]
                    if N > 160000:
                        start_off = (N - 160000) // 2
                        wav_5s = wav_5s[:, start_off:start_off + 160000]
                    elif N < 160000:
                        wav_5s = F.pad(wav_5s, (0, 160000 - N))
                    perch_emb = perch_teacher.embed(wav_5s).to(device)
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss
            else:
                distill_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

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

        # Per-batch verbose log (every 20 batches)
        if batch_idx % 20 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    ep{epoch+1:02d} batch {batch_idx:4d}/{N_STEPS_PER_EP}  "
                  f"loss={loss.item():.4f}  cls={cls_loss.item():.4f}  "
                  f"dist={distill_loss.item():.4f}  lr={cur_lr:.2e}", flush=True)

    train_loss_avg = el / max(nb_count, 1)
    cls_loss_avg = el_cls / max(nb_count, 1)
    dist_loss_avg = el_dist / max(nb_count, 1)

    # Validation
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
    ep = epoch + 1  # 1-indexed epoch completed

    # Save full state
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

    # Best by ns22
    if not math.isnan(val_metrics["ns22"]) and val_metrics["ns22"] > best_ns22:
        best_ns22 = val_metrics["ns22"]
        state_to_save["best_ns22"] = best_ns22
        torch.save(state_to_save, OUT_DIR / "ckpt_best_ns22.pth")
    # Best by macro
    if not math.isnan(val_metrics["macro"]) and val_metrics["macro"] > best_macro:
        best_macro = val_metrics["macro"]
        state_to_save["best_macro"] = best_macro
        torch.save(state_to_save, OUT_DIR / "ckpt_best_macro.pth")

    # Append history (drop large arrays)
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

    # Free memory
    gc.collect()
    torch.cuda.empty_cache()

trained_up_to = ep  # how many epochs done (cumulative)
print(f"\n=== Session done: trained epochs {start_epoch+1}-{trained_up_to} this session ===")
print(f"    Cumulative: {trained_up_to}/{N_TOTAL_EPOCHS} epochs done")

# Clean up GPU before upload
del perch_teacher, mel_transform, spec_augment
gc.collect()
torch.cuda.empty_cache()
""", "train_loop"))

# =============================================================================
# Cell 12: Upload state to Kaggle Dataset
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 12: Upload state to maekeso/exp015-state (KEY CELL)
# ============================================================
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle env auto-mounts /root/.kaggle/kaggle.json if Kaggle Username & Key
# are set in NB settings. Use api.authenticate() to fetch them.
api = KaggleApi(); api.authenticate()

DATASET_USER = "maekeso"
DATASET_SLUG = "exp015-state"
DATASET_TITLE = "exp015 training state"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # Copy output artifacts
    n_copied = 0
    for f in Path("/kaggle/working").glob("*"):
        if not f.is_file(): continue
        if f.suffix not in {".pth", ".json", ".txt"}: continue
        # Skip per-epoch ckpts to keep dataset small? No — we want full per-epoch ckpt log.
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

    version_notes = f"epoch {trained_up_to}/{N_TOTAL_EPOCHS}"
    uploaded = False
    # Try update existing
    try:
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"OK Uploaded to {DATASET_USER}/{DATASET_SLUG} (new version, {version_notes})")
        uploaded = True
    except Exception as e:
        msg = str(e)
        print(f"  dataset_create_version error: {msg[:300]}")
        # If dataset doesn't exist yet, create new
        if "not found" in msg.lower() or "404" in msg or "Could not find dataset" in msg:
            try:
                api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
                print(f"OK Created {DATASET_USER}/{DATASET_SLUG} (first time)")
                uploaded = True
            except Exception as e2:
                print(f"  dataset_create_new error: {str(e2)[:300]}")

    if not uploaded:
        print("\nUpload failed — files remain in /kaggle/working/ for manual upload via Kaggle UI")
        print("Files:")
        for p in sorted(Path('/kaggle/working').iterdir()):
            if p.is_file():
                print(f"  {p.name}  {p.stat().st_size/1e6:.2f} MB")
""", "upload_state"))

# =============================================================================
# Cell 13: Summary
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 13: Session summary
# ============================================================
total_time = time.time() - TRAIN_START
print(f"\n{'='*60}")
print(f"Session summary")
print(f"{'='*60}")
print(f"  Resume status:        {'RESUMED' if resumed else 'FRESH START'}")
print(f"  Resumed from epoch:   {start_epoch}")
print(f"  Trained this session: {trained_this_session} epoch(s)")
print(f"  Cumulative:           {trained_up_to}/{N_TOTAL_EPOCHS} epochs")
print(f"  Best ns22 AUC:        {best_ns22:.4f}")
print(f"  Best macro AUC:       {best_macro:.4f}")
print(f"  Total session time:   {total_time/60:.1f} min")
if trained_up_to >= N_TOTAL_EPOCHS:
    print(f"\n  >>> Training COMPLETE <<<")
    print(f"  Final ckpt: ckpt_latest.pth (epoch {trained_up_to})")
    print(f"  Best ckpts: ckpt_best_ns22.pth ({best_ns22:.4f}), "
          f"ckpt_best_macro.pth ({best_macro:.4f})")
else:
    print(f"\n  ... Save & Run All again to continue from epoch {trained_up_to}/{N_TOTAL_EPOCHS}")
""", "summary"))

# =============================================================================
# Assemble notebook
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

out_path = HERE / "nb_train.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
