"""Build exp080: 1-stage combined training NB.

Design (simplified, single-fold, single-metric):
  Backbone:  tf_efficientnet_b0.ns_jft_in1k
  Init:      Babych iter3 b0 partial (strict=False)
  Window:    5s @ 32kHz
  Mel:       Tucker spec (n_mels=256, n_fft=2048, hop=512, fmin=20, fmax=16000, top_db=80)
  Head:      SED AttHead (Babych style, framewise + clip_max)

Data (1-stage combined union, no rank-pct, no power transform, no filter):
  - BC2026 train_audio (hard label, multi-hot primary+secondary) — focal
  - BC2026 train_soundscapes (soft pseudo from exp080a 3-stream adaptive blend)
  - XC audio Part 1+2+3 (soft pseudo from exp080b Tucker single)

Augmentation:
  - SpecAug (freq + time mask)
  - Mixup Beta(0.4) p=0.5
  - Soundscape background mix (focal × 0.7 + bg × 0.3 at p=0.5)

Loss:    BCE (0.5 * clip + 0.5 * framewise_max)
Optim:   AdamW lr=3e-4, cosine + warmup 2 ep, weight_decay 1e-4
Epoch:   20, Batch: 32 (T4) / 64 (L4)
Seed:    42

Val metric (single):
  - val_focal_macro (chunk-level, stratified 20% holdout of train_audio)

Output:
  - m_single_ckpt_best.pth (weights at best val_focal_macro)
  - m_single_ckpt_final.pth (last epoch)
  - history.json (per-epoch metrics)
  - m_single_summary.json
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_train.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# Cell 0: Title
cells.append(md_cell("""# exp080: 1-stage Combined Training (b0 + Babych init + Simple Pseudo Union)

**Design**: tf_efficientnet_b0 + Babych iter3 partial init + 5s window + 1-stage combined training.

**Data (union)**:
  - BC2026 train_audio (focal, hard label)
  - BC2026 train_soundscapes (3-stream adaptive blend pseudo from exp080a)
  - XC audio (Tucker pseudo from exp080b)

**Val** (single metric): val_focal_macro on stratified 20% holdout of train_audio (chunk-level macro AUC, NO labeled SS contamination)

**Output**: best ckpt (.pth) + history JSON
"""))

# Cell 1: Install
cells.append(code_cell("""!pip install -q librosa timm
import sys
print(f"Python: {sys.version[:50]}")
"""))

# Cell 2: Imports
cells.append(code_cell("""import os, time, json, gc, math, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import timm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tqdm.auto as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, torch {torch.__version__}, timm {timm.__version__}")
START = time.time()
"""))

# Cell 3: CFG
cells.append(code_cell("""# CFG
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC   # 160_000

# Mel (Tucker SED spec, matches our pseudo teacher mel)
N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000
TOP_DB = 80

# Train
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
DROP_PATH = 0.10
LABEL_SMOOTHING = 0.05

# Aug
MIXUP_ALPHA = 0.4
MIXUP_P = 0.5
SPECAUG_FREQ = 10
SPECAUG_TIME = 10
BG_MIX_P = 0.5    # focal × 0.7 + soundscape_bg × 0.3 mix p

# Val
VAL_FRACTION = 0.20
N_CLASSES = 234
LOG_STEP_INTERVAL = 100

SEED = 42

# Paths
def find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None

DATA_PATH = find_dir([
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
])
assert DATA_PATH is not None
TRAIN_CSV = Path(DATA_PATH) / "train.csv"
TRAIN_AUDIO_DIR = Path(DATA_PATH) / "train_audio"
TRAIN_SC_DIR = Path(DATA_PATH) / "train_soundscapes"
TAXONOMY_CSV = Path(DATA_PATH) / "taxonomy.csv"
SAMPLE_SUB = Path(DATA_PATH) / "sample_submission.csv"

# Babych b0 init
BABYCH_DIR = find_dir([
    "/kaggle/input/birdclef2025-1st-place-ensemble",
    "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble",
])
assert BABYCH_DIR is not None, "Babych dataset not mounted"
BABYCH_B0_CKPT = None
for f in BABYCH_DIR.glob("tf_efficientnet_b0*.pt"):
    BABYCH_B0_CKPT = f; break
assert BABYCH_B0_CKPT is not None, f"Babych b0 ckpt not found in {BABYCH_DIR}"
print(f"Babych b0 ckpt: {BABYCH_B0_CKPT.name}")

# Soundscape pseudo (exp080a)
SOUNDSCAPE_PSEUDO_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp080a-adaptive-blend",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp080a-adaptive-blend",
])
assert SOUNDSCAPE_PSEUDO_DIR is not None, "exp080a output not mounted"
print(f"Soundscape pseudo: {SOUNDSCAPE_PSEUDO_DIR}")

# XC pseudo (exp080b)
XC_PSEUDO_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp080b-xc-pseudo-tucker",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp080b-xc-pseudo-tucker",
])
assert XC_PSEUDO_DIR is not None, "exp080b output not mounted"
print(f"XC pseudo: {XC_PSEUDO_DIR}")

# XC audio sources (kernel outputs)
XC_PART1 = find_dir([
    "/kaggle/input/birdclef2026-exp047-xc-api-dl-part1",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp047-xc-api-dl-part1",
])
XC_PART2 = find_dir([
    "/kaggle/input/birdclef2026-exp047-xc-api-dl-part2",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp047-xc-api-dl-part2",
])
XC_PART3 = find_dir([
    "/kaggle/input/birdclef2026-xc-api-dl-part3",
])
print(f"XC Part 1: {XC_PART1}, Part 2: {XC_PART2}, Part 3: {XC_PART3}")

OUT_DIR = Path("/kaggle/working")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
"""))

# Cell 4: Load taxonomy and split
cells.append(code_cell("""# Load taxonomy + sample_submission for 234 sp ordering
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES, f"Expected {N_CLASSES}, got {len(PRIMARY_LABELS)}"
LABEL2IDX = {label: i for i, label in enumerate(PRIMARY_LABELS)}
print(f"234 species: {len(PRIMARY_LABELS)}")

taxo = pd.read_csv(TAXONOMY_CSV)
label_to_taxon = dict(zip(taxo["primary_label"].astype(str), taxo["class_name"].astype(str)))

# Load train.csv (focal recordings)
train_df = pd.read_csv(TRAIN_CSV)
train_df["primary_label"] = train_df["primary_label"].astype(str)
# Filter existing files
def _exists(fn): return (TRAIN_AUDIO_DIR / fn).exists()
train_df["exists"] = train_df["filename"].map(_exists)
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"train_df (focal recordings, existing): {len(train_df)}")

# Stratified 80/20 split by primary_label
# Some rare species may have <2 samples → fall back to random split for those
from collections import Counter
label_counts = Counter(train_df["primary_label"])
single_label = [lbl for lbl, c in label_counts.items() if c < 2]
multi_label_df = train_df[~train_df["primary_label"].isin(single_label)].reset_index(drop=True)
single_label_df = train_df[train_df["primary_label"].isin(single_label)].reset_index(drop=True)
print(f"  multi-label (>=2 samples): {len(multi_label_df)}")
print(f"  single-label (rare, all to train): {len(single_label_df)} from {len(single_label)} species")

train_multi, val_multi = train_test_split(
    multi_label_df, test_size=VAL_FRACTION,
    stratify=multi_label_df["primary_label"], random_state=SEED,
)
train_focal_df = pd.concat([train_multi, single_label_df], ignore_index=True).reset_index(drop=True)
val_focal_df = val_multi.reset_index(drop=True)
print(f"  train_focal: {len(train_focal_df)}, val_focal: {len(val_focal_df)}")
print(f"  train sp coverage: {train_focal_df['primary_label'].nunique()} / {len(PRIMARY_LABELS)}")
print(f"  val sp coverage:   {val_focal_df['primary_label'].nunique()} / {len(PRIMARY_LABELS)}")
"""))

# Cell 5: Soundscape and XC pseudo loading
cells.append(code_cell("""# Load soundscape pseudo (exp080a adaptive blend)
ss_npz_path = SOUNDSCAPE_PSEUDO_DIR / "pseudo_adaptive_234.npz"
assert ss_npz_path.exists(), f"Missing {ss_npz_path}"
ss_npz = np.load(ss_npz_path, allow_pickle=True)
ss_probs = ss_npz["probs"].astype(np.float32)  # (N_files, 12, 234)
ss_file_ids = ss_npz["file_ids"]
print(f"Soundscape pseudo: {ss_probs.shape}, mean={ss_probs.mean():.5f}")

# Map file_id (stem) → .ogg file path
ss_id_to_path = {fid: TRAIN_SC_DIR / f"{fid}.ogg" for fid in ss_file_ids}
ss_exists = {fid: p.exists() for fid, p in ss_id_to_path.items()}
print(f"  Soundscape files exist: {sum(ss_exists.values())} / {len(ss_file_ids)}")

# Load XC pseudo (exp080b)
xc_npz_path = XC_PSEUDO_DIR / "xc_pseudo_tucker.npz"
assert xc_npz_path.exists(), f"Missing {xc_npz_path}"
xc_npz = np.load(xc_npz_path, allow_pickle=True)
xc_probs = xc_npz["probs"].astype(np.float32)
xc_file_ids = xc_npz["file_ids"]
xc_sci = xc_npz["sci_names"]
xc_primary = xc_npz["primary_labels"]
xc_durations = xc_npz["durations_sec"]
xc_n_actual = xc_npz["n_actual_chunks"]
print(f"\\nXC pseudo: {xc_probs.shape}, mean={xc_probs.mean():.5f}")

# Build XC id → path map by full rglob (O(N) once, O(1) lookup)
print(f"  Building XC audio path index (rglob)...")
xc_id_to_path_full = {}
for base in [XC_PART1, XC_PART2, XC_PART3]:
    if base is None: continue
    for fp in base.rglob("*.mp3"):
        xc_id_to_path_full[fp.stem] = fp
print(f"  Total XC mp3 files indexed: {len(xc_id_to_path_full)}")

xc_id_to_path = {}
xc_missing = 0
for xc_id in xc_file_ids:
    fp = xc_id_to_path_full.get(str(xc_id))
    if fp is not None:
        xc_id_to_path[str(xc_id)] = fp
    else:
        xc_missing += 1
print(f"  XC audio files matched to pseudo: {len(xc_id_to_path)} / {len(xc_file_ids)} ({xc_missing} missing)")
"""))

# Cell 6: Model architecture (Babych SED b0)
cells.append(code_cell("""# Babych SED architecture (same as M7 but 234-class)
def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class AttHead(nn.Module):
    def __init__(self, in_chans, p=0.5, num_class=234, hidden_dim=512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
        norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
        return (Xstd - norm_min) / (norm_max - norm_min + self.eps)


class SpecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(sample_rate=SR, normalized=True, n_fft=N_FFT,
                             hop_length=HOP_LENGTH, win_length=N_FFT,
                             f_max=F_MAX, n_mels=N_MELS, f_min=F_MIN),
            T.AmplitudeToDB(top_db=TOP_DB),
        )
        self.norm = NormalizeMelSpec()
    def forward(self, x):
        return self.norm(self.feature_extractor(x))


class CLEFClassifierSED(nn.Module):
    def __init__(self, num_classes=N_CLASSES, drop_path_rate=DROP_PATH):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0.ns_jft_in1k", pretrained=True, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)
        self.num_classes = num_classes

    def forward(self, wav, return_framewise=False):
        spec = self.mel_spectr_generator(wav)
        spec3 = torch.stack([spec, spec, spec], 1)
        feat = self.backbone(spec3)[-1]
        head_output = self.head(feat)
        framewise_logit = head_output["framewise_logit"]
        clip_logit = framewise_logit.max(dim=-1).values
        if return_framewise:
            return clip_logit, framewise_logit
        return clip_logit


def make_model_with_babych_init():
    model = CLEFClassifierSED()
    babych_state = torch.load(str(BABYCH_B0_CKPT), weights_only=True, map_location="cpu")
    # Only backbone (head will be re-initialized for 234 classes)
    backbone_state = {k: v for k, v in babych_state.items() if k.startswith("backbone.")}
    msg = model.load_state_dict(backbone_state, strict=False)
    print(f"  Babych load (strict=False): missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return model


_tmp = make_model_with_babych_init()
print(f"Model: {sum(p.numel() for p in _tmp.parameters())/1e6:.1f}M params")
del _tmp; gc.collect()
"""))

# Cell 7: Datasets (focal + soundscape + xc)
cells.append(code_cell("""# Datasets — return (wav_5s, label_234) tuple
class FocalHardDS(Dataset):
    \"\"\"BC2026 train_audio with hard multi-hot labels (primary+secondary).\"\"\"
    def __init__(self, df, train_audio_dir, label2idx, ss_pseudo_for_bg=None, ss_paths_for_bg=None, train_mode=True):
        self.df = df.reset_index(drop=True)
        self.dir = Path(train_audio_dir)
        self.label2idx = label2idx
        self.train_mode = train_mode
        # For BG mix (training only)
        self.ss_paths = ss_paths_for_bg if (train_mode and ss_paths_for_bg) else None

    def __len__(self):
        return len(self.df)

    def load_audio(self, filename):
        try:
            y, _ = librosa.load(str(self.dir / filename), sr=SR, mono=True)
            return y.astype(np.float32)
        except Exception:
            return np.zeros(SR * 5, dtype=np.float32)

    def crop_5s(self, y):
        if len(y) < WINDOW_SAMPLES:
            pad = WINDOW_SAMPLES - len(y)
            left = np.random.randint(0, pad + 1) if self.train_mode else pad // 2
            y = np.pad(y, (left, pad - left))
        elif len(y) > WINDOW_SAMPLES:
            start = np.random.randint(0, len(y) - WINDOW_SAMPLES + 1) if self.train_mode else (len(y) - WINDOW_SAMPLES) // 2
            y = y[start: start + WINDOW_SAMPLES]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.load_audio(row["filename"])
        y = self.crop_5s(y)

        # Optional BG mix (training only)
        if self.train_mode and self.ss_paths and np.random.random() < BG_MIX_P:
            bg_path = self.ss_paths[np.random.randint(len(self.ss_paths))]
            try:
                bg, _ = librosa.load(str(bg_path), sr=SR, mono=True)
                bg = self.crop_5s(bg.astype(np.float32))
                y = 0.7 * y + 0.3 * bg
            except Exception:
                pass

        m = np.abs(y).max()
        if m > 0: y = y / m

        # Multi-hot label
        label = np.zeros(N_CLASSES, dtype=np.float32)
        if row["primary_label"] in self.label2idx:
            label[self.label2idx[row["primary_label"]]] = 1.0
        sec = str(row.get("secondary_labels", "")).strip()
        if sec and sec != "[]" and sec != "nan":
            for s in sec.replace("[", "").replace("]", "").replace("'", "").split(","):
                s = s.strip()
                if s in self.label2idx:
                    label[self.label2idx[s]] = 1.0

        # Label smoothing
        if LABEL_SMOOTHING > 0:
            label = label * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / N_CLASSES

        return torch.from_numpy(y), torch.from_numpy(label)


class SoundscapePseudoDS(Dataset):
    \"\"\"BC2026 soundscape with 3-stream adaptive blend pseudo. Random chunk per __getitem__.\"\"\"
    def __init__(self, file_ids, id_to_path, pseudo_array):
        # Filter to existing files
        self.entries = []
        for i, fid in enumerate(file_ids):
            p = id_to_path.get(fid)
            if p and p.exists():
                self.entries.append((str(fid), p, i))
        self.pseudo = pseudo_array  # (N, 12, 234)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fid, path, pseudo_idx = self.entries[idx]
        # Pick random window 0-11
        win = np.random.randint(12)
        try:
            y, _ = librosa.load(str(path), sr=SR, mono=True,
                                offset=win * 5.0, duration=5.0)
        except Exception:
            y = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        else:
            y = y[:WINDOW_SAMPLES]
        m = np.abs(y).max()
        if m > 0: y = y / m
        target = self.pseudo[pseudo_idx, win].astype(np.float32)
        return torch.from_numpy(y.astype(np.float32)), torch.from_numpy(target)


class XCPseudoDS(Dataset):
    \"\"\"XC audio (MP3) with Tucker pseudo.

    ★ MP3 seek 精度問題回避のため、full audio load → pad 60s → reshape 12 chunks
       (exp080b と同じパターンで chunk alignment を保証)
    \"\"\"
    def __init__(self, file_ids, id_to_path, pseudo_array, n_actual_chunks):
        self.entries = []
        for i, fid in enumerate(file_ids):
            p = id_to_path.get(str(fid))
            n_act = int(n_actual_chunks[i])
            if p and p.exists() and n_act >= 1:
                self.entries.append((str(fid), p, i, n_act))
        self.pseudo = pseudo_array
        self.target_60s_samples = SR * 60  # 60s padded total

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        fid, path, pseudo_idx, n_act = self.entries[idx]
        # Full audio load + pad 60s + reshape (matches exp080b pipeline)
        try:
            y, _ = librosa.load(str(path), sr=SR, mono=True)
            y = y.astype(np.float32)
            if len(y) < self.target_60s_samples:
                y = np.pad(y, (0, self.target_60s_samples - len(y)))
            else:
                y = y[:self.target_60s_samples]
        except Exception:
            y = np.zeros(self.target_60s_samples, dtype=np.float32)

        chunks = y.reshape(12, WINDOW_SAMPLES)
        # Random window 0..n_act-1 (within actual audio range)
        win = np.random.randint(n_act)
        chunk = chunks[win]
        m = np.abs(chunk).max()
        if m > 0: chunk = chunk / m
        target = self.pseudo[pseudo_idx, win].astype(np.float32)
        return torch.from_numpy(chunk.astype(np.float32)), torch.from_numpy(target)


# Construct datasets
ss_path_list = [p for fid, p in ss_id_to_path.items() if p.exists()][:200]  # BG mix pool

train_focal_ds = FocalHardDS(train_focal_df, TRAIN_AUDIO_DIR, LABEL2IDX,
                              ss_paths_for_bg=ss_path_list, train_mode=True)
val_focal_ds = FocalHardDS(val_focal_df, TRAIN_AUDIO_DIR, LABEL2IDX, train_mode=False)
ss_ds = SoundscapePseudoDS(ss_file_ids, ss_id_to_path, ss_probs)
xc_ds = XCPseudoDS(xc_file_ids, xc_id_to_path, xc_probs, xc_n_actual)

print(f"train_focal_ds: {len(train_focal_ds)}")
print(f"val_focal_ds:   {len(val_focal_ds)}")
print(f"ss_ds:          {len(ss_ds)}")
print(f"xc_ds:          {len(xc_ds)}")

train_combined_ds = ConcatDataset([train_focal_ds, ss_ds, xc_ds])
print(f"\\nCombined train dataset: {len(train_combined_ds)} samples")
"""))

# Cell 8: Train helpers (mixup, specaug, val)
cells.append(code_cell("""# Helpers
def mixup_audio(wav, label, alpha=MIXUP_ALPHA, p=MIXUP_P):
    if np.random.random() >= p:
        return wav, label
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(wav.size(0), device=wav.device)
    return lam * wav + (1 - lam) * wav[idx], lam * label + (1 - lam) * label[idx]


@torch.no_grad()
def eval_val_focal_macro(model, val_dl, device):
    \"\"\"Chunk-level macro AUC over val_focal (BC2026 train_audio holdout).\"\"\"
    model.eval()
    all_preds, all_labels = [], []
    for wav, label in val_dl:
        wav = wav.to(device, non_blocking=True)
        with autocast():
            clip_logit = model(wav)
        all_preds.append(torch.sigmoid(clip_logit).float().cpu().numpy())
        all_labels.append(label.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    # Threshold label (label smoothing → ~0.95 for positive; treat >0.5 as positive)
    labels_bin = (labels > 0.5).astype(np.float32)
    # Per-species AUC
    aucs = []
    for sp in range(N_CLASSES):
        if labels_bin[:, sp].sum() > 0 and labels_bin[:, sp].sum() < len(labels_bin):
            try:
                aucs.append(roc_auc_score(labels_bin[:, sp], preds[:, sp]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)
"""))

# Cell 9: Train loop
cells.append(code_cell("""from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

print(f"=== Training ===")
print(f"  train_combined: {len(train_combined_ds)}, val_focal: {len(val_focal_ds)}")

train_dl = DataLoader(train_combined_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
val_dl = DataLoader(val_focal_ds, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=2, pin_memory=True)

model = make_model_with_babych_init().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
warmup_iters = WARMUP_EPOCHS * len(train_dl)
total_iters = N_EPOCHS * len(train_dl)
sched_warmup = LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_iters)
sched_cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=LR_MIN)
scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_iters])
scaler = GradScaler()

best_val = -1.0
history = []
total_steps = len(train_dl)

for epoch in range(N_EPOCHS):
    t0_ep = time.time()
    model.train()
    tr_loss_sum = 0.0
    n_seen = 0

    for step, (wav, label) in enumerate(train_dl):
        wav = wav.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        wav, label = mixup_audio(wav, label)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            clip_logit, framewise_logit = model(wav, return_framewise=True)
            frame_max_logit = framewise_logit.max(dim=-1).values
            loss_clip = F.binary_cross_entropy_with_logits(clip_logit, label)
            loss_frame = F.binary_cross_entropy_with_logits(frame_max_logit, label)
            loss = 0.5 * loss_clip + 0.5 * loss_frame

        # NaN guard
        if not torch.isfinite(loss):
            print(f"  [ep{epoch+1} step{step}] WARN: non-finite loss, skip")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tr_loss_sum += loss.item() * wav.size(0)
        n_seen += wav.size(0)

        if step % LOG_STEP_INTERVAL == 0 or step == total_steps - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  [ep{epoch+1} step {step}/{total_steps}] loss={loss.item():.4f} lr={cur_lr:.2e}")

    tr_loss = tr_loss_sum / max(n_seen, 1)

    # Val
    val_focal_macro, n_valid_sp = eval_val_focal_macro(model, val_dl, DEVICE)

    ep_time = (time.time() - t0_ep) / 60
    total_time = (time.time() - START) / 60
    cur_lr = optimizer.param_groups[0]["lr"]
    is_best = val_focal_macro > best_val
    best_tag = "BEST " if is_best else ""
    print(f"=== Ep {epoch+1}/{N_EPOCHS}: loss={tr_loss:.4f} val_focal_macro={val_focal_macro:.4f} "
          f"({n_valid_sp} sp) {best_tag}lr={cur_lr:.2e} ({ep_time:.1f}min, total {total_time:.1f}min) ===")

    if is_best:
        best_val = val_focal_macro
        torch.save({
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "epoch": epoch, "val_focal_macro": val_focal_macro, "n_valid_sp": n_valid_sp,
        }, OUT_DIR / "m_single_ckpt_best.pth")
        print(f"    BEST saved val_focal_macro={val_focal_macro:.4f}")

    history.append({
        "ep": epoch, "tr_loss": tr_loss, "val_focal_macro": val_focal_macro,
        "n_valid_sp": n_valid_sp, "lr": cur_lr, "ep_time_min": ep_time, "best": is_best,
    })

# Final ckpt + history
torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "epoch": N_EPOCHS - 1, "history": history},
           OUT_DIR / "m_single_ckpt_final.pth")
with open(OUT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\\nTraining DONE: best val_focal_macro={best_val:.4f}")
print(f"Total time: {(time.time()-START)/60:.1f} min")
"""))

# Cell 10: ONNX export (best ckpt)
cells.append(code_cell("""# ONNX export of BEST ckpt
print("=== ONNX export ===")
best_ckpt = torch.load(OUT_DIR / "m_single_ckpt_best.pth", weights_only=False, map_location="cpu")
export_model = CLEFClassifierSED().cpu().eval()
export_model.load_state_dict(best_ckpt["model_state"])

# Dummy input: 5s wav at 32kHz
dummy_wav = torch.randn(1, WINDOW_SAMPLES, dtype=torch.float32)

# Trace test
with torch.no_grad():
    out = export_model(dummy_wav)
    print(f"  Trace output shape: {out.shape}")  # expect (1, 234)

# Export
onnx_path = OUT_DIR / "m_single_best.onnx"
try:
    torch.onnx.export(
        export_model, dummy_wav, str(onnx_path),
        input_names=["wav"], output_names=["clip_logits"],
        dynamic_axes={"wav": {0: "batch"}, "clip_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"  ONNX exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    print(f"  ONNX export FAILED: {e}")
    import traceback; traceback.print_exc()

# Quick numerical check
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"wav": dummy_wav.numpy()})[0]
    diff = np.abs(out.numpy() - onnx_out).max()
    print(f"  PyTorch vs ONNX max diff: {diff:.6e}")
except Exception as e:
    print(f"  ONNX runtime check failed: {e}")
"""))

# Cell 11: Summary
cells.append(code_cell("""# Summary
summary = {
    "exp": "exp080",
    "backbone": "tf_efficientnet_b0.ns_jft_in1k",
    "init": "Babych iter3 partial (strict=False)",
    "window_sec": WINDOW_SEC,
    "mel": {"n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP_LENGTH, "fmin": F_MIN, "fmax": F_MAX, "top_db": TOP_DB},
    "n_epochs": N_EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR, "wd": WEIGHT_DECAY, "drop_path": DROP_PATH, "label_smoothing": LABEL_SMOOTHING,
    "mixup_alpha": MIXUP_ALPHA, "mixup_p": MIXUP_P, "bg_mix_p": BG_MIX_P,
    "data": {
        "focal_train": len(train_focal_ds),
        "focal_val": len(val_focal_ds),
        "soundscape": len(ss_ds),
        "xc": len(xc_ds),
        "combined_train_total": len(train_combined_ds),
    },
    "best_val_focal_macro": float(best_val),
    "history": history,
    "total_time_min": (time.time() - START) / 60,
}
with open(OUT_DIR / "m_single_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Built: {OUT_PATH}")
print(f"  cells: {len(cells)}")
for c in cells:
    n_lines = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} L={n_lines}")
