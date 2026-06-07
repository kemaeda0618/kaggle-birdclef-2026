"""Build M7 training NB: tf_efficientnet_b0 + 20s Babych mel + Babych b0 init + 63-class supervised.

M7 = Amphibia/Insecta sub model
- Backbone: tf_efficientnet_b0.ns_jft_in1k
- Init: Babych BC25 iter4 b0 (Amphibia/Insecta 700-class ckpt)
- Mel: 20s × 224 × 4096 × 1252 (Babych spec)
- Output: 63 classes (35 Amphibia + 28 Insecta from BC26 taxonomy)
- Loss: BCE
- Pseudo: なし (supervised only、Babych b0 sub model 流の Stage 1)
- Aug: MixUp + SpecAug
- Train: 40 epoch × 3-fold StratifiedKFold
- Platform: Kaggle T4x2 GPU
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_train_m7.ipynb")

cells = []


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


# ─────────── Cell 0: Title ───────────
cells.append(md_cell("""# exp076 (M7): tf_efficientnet_b0 + Babych init + 63-class supervised

**Babych sub model 流の path**:
  - Backbone: `tf_efficientnet_b0.ns_jft_in1k`
  - Init: Babych BC25 iter4 b0 (Amphibia/Insecta 700-class) ckpt — strict=False で head 除外
  - Mel: 20s × n_mels=224, n_fft=4096, hop=1252, fmin=0, fmax=16000 (Babych spec)
  - Output: **63 class** (35 Amphibia + 28 Insecta from BC26 taxonomy)
  - Loss: BCE clip + framewise max
  - Aug: MixUp (audio level、alpha=0.4) + SpecAug 10/10
  - Pseudo: **なし** (supervised only、Babych Stage 1 流)
  - Fold: 3-fold StratifiedKFold (by species)
  - Epoch: 40 (Babych spec)
  - Platform: Kaggle T4x2 GPU

**Output**: 3 fold × ckpt + ONNX (各 ~5MB) を /kaggle/working/ に
"""))


# ─────────── Cell 1: Install ───────────
cells.append(code_cell("""# Kaggle env: torch, timm, librosa, torchaudio はデフォルトある
!pip install onnxruntime --quiet
import sys
print(f"Python: {sys.version[:50]}")
"""))


# ─────────── Cell 2: Imports + setup ───────────
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
import tqdm.auto as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, torch {torch.__version__}, timm {timm.__version__}")
START = time.time()
"""))


# ─────────── Cell 3: CFG ───────────
cells.append(code_cell("""# CFG (Babych spec 準拠)
SR = 32_000
WINDOW_SEC = 20                # ★ Babych 20s
WINDOW_SAMPLES = SR * WINDOW_SEC

# Mel (Babych)
N_MELS = 224
N_FFT = 4096
HOP_LENGTH = 1252
F_MIN = 0
F_MAX = 16000
TOP_DB = 80

# Training (Babych b0 Amphibia/Insecta spec)
N_FOLDS = 3
N_EPOCHS = 40
BATCH_SIZE = 32
LR = 5e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

# Aug
MIXUP_ALPHA = 0.4
MIXUP_P = 0.5
SPECAUG_FREQ = 10
SPECAUG_TIME = 10
DROP_PATH = 0.0  # Babych Stage 1: drop_path=0

SEED = 42

# Paths
_data_path_candidates = [
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
]
DATA_PATH = None
for _p in _data_path_candidates:
    if Path(_p).exists():
        DATA_PATH = _p; break
assert DATA_PATH is not None

TRAIN_CSV = Path(DATA_PATH) / "train.csv"
TRAIN_AUDIO_DIR = Path(DATA_PATH) / "train_audio"
TAXONOMY_CSV = Path(DATA_PATH) / "taxonomy.csv"

# Babych b0 weight (Amphibia/Insecta iter4)
BABYCH_DIR = None
for _p in ["/kaggle/input/birdclef2025-1st-place-ensemble", "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble"]:
    if Path(_p).exists():
        BABYCH_DIR = Path(_p); break
assert BABYCH_DIR is not None

# Find b0 incest_amphibia ckpt
BABYCH_B0_CKPT = None
for f in BABYCH_DIR.glob("tf_efficientnet_b0*.pt"):
    if "incest_amphibia" in f.name:
        BABYCH_B0_CKPT = f; break
assert BABYCH_B0_CKPT is not None, f"Babych b0 sub ckpt not found in {BABYCH_DIR}"
print(f"BABYCH_B0_CKPT: {BABYCH_B0_CKPT.name}")

OUT_DIR = Path("/kaggle/working")
"""))


# ─────────── Cell 4: 63 species filter + fold split ───────────
cells.append(code_cell("""# Load taxonomy, filter 63 Amphibia + Insecta species
taxo = pd.read_csv(TAXONOMY_CSV)
amph_ins = taxo[taxo["class_name"].isin(["Amphibia", "Insecta"])].reset_index(drop=True)
SPECIES_63 = amph_ins["primary_label"].astype(str).tolist()
N_CLASSES = len(SPECIES_63)
assert N_CLASSES == 63, f"Expected 63, got {N_CLASSES}"
LABEL2IDX = {label: i for i, label in enumerate(SPECIES_63)}
print(f"63 species: {SPECIES_63[:5]} ... {SPECIES_63[-5:]}")

# Load train.csv, filter to 63 species
train_df = pd.read_csv(TRAIN_CSV)
train_df["primary_label"] = train_df["primary_label"].astype(str)
train_63 = train_df[train_df["primary_label"].isin(SPECIES_63)].reset_index(drop=True)
print(f"train.csv filtered to {len(train_63)} rows (out of {len(train_df)})")

# Filter to existing files
def _exists(fn):
    return (TRAIN_AUDIO_DIR / fn).exists()
train_63["exists"] = train_63["filename"].map(_exists)
train_63 = train_63[train_63["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"Files exist: {len(train_63)}")

# 3-fold StratifiedKFold by primary_label
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_63["fold"] = -1
for fi, (_, val_idx) in enumerate(skf.split(train_63, train_63["primary_label"])):
    train_63.loc[val_idx, "fold"] = fi
print(f"Fold distribution: {train_63['fold'].value_counts().to_dict()}")
"""))


# ─────────── Cell 5: Babych SED architecture (from exp069a) ───────────
cells.append(code_cell("""# Babych SED Model Architecture (CLEFClassifierSED)
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
    def __init__(self, in_chans, p=0.5, num_class=63, hidden_dim=512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


class NormalizeMelSpec(nn.Module):
    def __init__(self, norm_type="default", eps=1e-6):
        super().__init__()
        self.eps = eps
        self.norm_type = norm_type
    def forward(self, X):
        if self.norm_type == "default":
            mean = X.mean((1, 2), keepdim=True)
            std = X.std((1, 2), keepdim=True)
            Xstd = (X - mean) / (std + self.eps)
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)
        return X


class SpecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(sample_rate=SR, normalized=True, n_fft=N_FFT,
                             hop_length=HOP_LENGTH, win_length=N_FFT,
                             f_max=F_MAX, n_mels=N_MELS, f_min=F_MIN),
            T.AmplitudeToDB(top_db=TOP_DB),
        )
        self.norm = NormalizeMelSpec(norm_type="default")

    def forward(self, x):
        s = self.feature_extractor(x)  # (B, n_mels, T)
        s = self.norm(s)
        return s


class CLEFClassifierSED(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", num_classes=63, drop_path_rate=0.0):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)
        self.num_classes = num_classes

    def forward(self, wav, return_framewise=False):
        # wav: (B, samples)  for 20s = (B, 640000)
        spec = self.mel_spectr_generator(wav)        # (B, n_mels, T)
        spec3 = torch.stack([spec, spec, spec], 1)   # (B, 3, n_mels, T)
        feat = self.backbone(spec3)[-1]              # (B, C, F, T_feat)
        head_output = self.head(feat)
        framewise_logit = head_output["framewise_logit"]  # (B, num_class, T_feat')
        clip_logit = framewise_logit.max(dim=-1).values
        if return_framewise:
            return clip_logit, framewise_logit.permute(0, 2, 1)
        return clip_logit


print("CLEFClassifierSED architecture defined.")
"""))


# ─────────── Cell 6: Build model + load Babych weights ───────────
cells.append(code_cell("""# Build model + load Babych b0 weights (strict=False)
model = CLEFClassifierSED(num_classes=N_CLASSES, drop_path_rate=DROP_PATH)

print(f"Loading Babych b0 weights: {BABYCH_B0_CKPT.name}")
babych_state = torch.load(str(BABYCH_B0_CKPT), weights_only=True, map_location="cpu")

# Filter: only backbone weights, skip head (different class count)
backbone_state = {k: v for k, v in babych_state.items() if k.startswith("backbone.") or k.startswith("mel_spectr_generator.")}
print(f"Babych state keys: {len(babych_state)} total, {len(backbone_state)} backbone+mel keys")

msg = model.load_state_dict(backbone_state, strict=False)
print(f"  Missing keys: {len(msg.missing_keys)} (head + new layers)")
print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
print(f"  Backbone + mel loaded from Babych init ✓")

# Reference model_init (for fold reset later)
import io as _io
_init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
del model
gc.collect()
print(f"Model init state cached for fold reset ({time.time()-START:.0f}s)")
"""))


# ─────────── Cell 7: Dataset class ───────────
cells.append(code_cell("""# Dataset (20s random crop + 63-class hard label)
class FocalDS(Dataset):
    def __init__(self, df, label2idx, train_audio_dir, sr=SR, win_samples=WINDOW_SAMPLES, train_mode=True):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.train_audio_dir = Path(train_audio_dir)
        self.sr = sr
        self.win_samples = win_samples
        self.train_mode = train_mode

    def __len__(self):
        return len(self.df)

    def load_audio(self, filename):
        try:
            y, _ = librosa.load(str(self.train_audio_dir / filename), sr=self.sr, mono=True)
            return y.astype(np.float32)
        except Exception as e:
            return np.zeros(self.sr * 5, dtype=np.float32)

    def random_or_center_crop(self, y):
        if len(y) < self.win_samples:
            pad = self.win_samples - len(y)
            if self.train_mode:
                left = np.random.randint(0, pad + 1)
            else:
                left = pad // 2
            y = np.pad(y, (left, pad - left))
        elif len(y) > self.win_samples:
            if self.train_mode:
                start = np.random.randint(0, len(y) - self.win_samples + 1)
            else:
                start = (len(y) - self.win_samples) // 2
            y = y[start: start + self.win_samples]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.load_audio(row["filename"])
        y = self.random_or_center_crop(y)
        # absmax normalize (Babych)
        m = np.abs(y).max()
        if m > 0:
            y = y / m
        # Hard label (one-hot for primary + secondary)
        label = np.zeros(len(self.label2idx), dtype=np.float32)
        if row["primary_label"] in self.label2idx:
            label[self.label2idx[row["primary_label"]]] = 1.0
        # secondary labels
        sec = str(row.get("secondary_labels", "")).strip()
        if sec and sec != "[]" and sec != "nan":
            for s in sec.replace("[", "").replace("]", "").replace("'", "").split(","):
                s = s.strip()
                if s in self.label2idx:
                    label[self.label2idx[s]] = 1.0
        return torch.from_numpy(y), torch.from_numpy(label)


print("FocalDS defined.")
"""))


# ─────────── Cell 8: MixUp + SpecAug ───────────
cells.append(code_cell("""# MixUp (audio level, Babych alpha=0.4)
def mixup_audio(wav, label, alpha=MIXUP_ALPHA, p=MIXUP_P):
    if np.random.random() >= p:
        return wav, label
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(wav.size(0), device=wav.device)
    wav_mix = lam * wav + (1 - lam) * wav[idx]
    label_mix = lam * label + (1 - lam) * label[idx]
    return wav_mix, label_mix


# SpecAug (mel level)
class SpecAug(nn.Module):
    def __init__(self, freq_mask=SPECAUG_FREQ, time_mask=SPECAUG_TIME):
        super().__init__()
        self.fa = T.FrequencyMasking(freq_mask)
        self.ta = T.TimeMasking(time_mask)

    def forward(self, spec):
        return self.ta(self.fa(spec))


print("MixUp + SpecAug defined.")
"""))


# ─────────── Cell 9: Training loop (3-fold × 40 epoch) ───────────
cells.append(code_cell("""# 3-fold training loop
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def train_fold(fold_k):
    print(f"\\n{'='*60}\\n[Fold {fold_k}] M7 training {N_EPOCHS} epochs\\n{'='*60}")
    t0_fold = time.time()

    tr_df = train_63[train_63["fold"] != fold_k].reset_index(drop=True)
    val_df = train_63[train_63["fold"] == fold_k].reset_index(drop=True)
    print(f"  train: {len(tr_df)}, val: {len(val_df)}")

    tr_ds = FocalDS(tr_df, LABEL2IDX, TRAIN_AUDIO_DIR, train_mode=True)
    val_ds = FocalDS(val_df, LABEL2IDX, TRAIN_AUDIO_DIR, train_mode=False)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Fresh model + load Babych init
    model = CLEFClassifierSED(num_classes=N_CLASSES, drop_path_rate=DROP_PATH)
    model.load_state_dict(_init_state, strict=True)
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_iters = WARMUP_EPOCHS * len(tr_dl)
    total_iters = N_EPOCHS * len(tr_dl)
    sched_warmup = LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_iters)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=LR_MIN)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_iters])
    scaler = GradScaler()
    spec_aug = SpecAug().to(DEVICE)

    best_loss = float("inf")
    history = []

    for epoch in range(N_EPOCHS):
        t0_ep = time.time()
        model.train()
        tr_loss_sum = 0.0
        n_seen = 0
        for wav, label in tr_dl:
            wav = wav.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            wav, label = mixup_audio(wav, label)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                clip_logit, framewise_logit = model(wav, return_framewise=True)
                frame_max_logit = framewise_logit.max(dim=1).values
                loss_clip = F.binary_cross_entropy_with_logits(clip_logit, label)
                loss_frame = F.binary_cross_entropy_with_logits(frame_max_logit, label)
                loss = 0.5 * loss_clip + 0.5 * loss_frame
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss_sum += loss.item() * wav.size(0)
            n_seen += wav.size(0)

        tr_loss = tr_loss_sum / max(n_seen, 1)

        # Val
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for wav, label in val_dl:
                wav = wav.to(DEVICE, non_blocking=True)
                label = label.to(DEVICE, non_blocking=True)
                with autocast():
                    clip_logit = model(wav)
                    val_loss = F.binary_cross_entropy_with_logits(clip_logit, label)
                val_loss_sum += val_loss.item() * wav.size(0)
                n_val += wav.size(0)
        val_loss = val_loss_sum / max(n_val, 1)
        ep_time = time.time() - t0_ep
        history.append({"ep": epoch, "tr_loss": tr_loss, "val_loss": val_loss, "time": ep_time})
        lr = optimizer.param_groups[0]["lr"]
        print(f"  ep{epoch:02d} tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} ({ep_time:.0f}s)")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "val_loss": val_loss},
                       OUT_DIR / f"m7_fold{fold_k}_ckpt_best.pth")

    # Final ckpt
    torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": N_EPOCHS - 1, "val_loss": val_loss, "history": history},
               OUT_DIR / f"m7_fold{fold_k}_ckpt_final.pth")
    with open(OUT_DIR / f"m7_fold{fold_k}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    fold_elapsed = time.time() - t0_fold
    print(f"[Fold {fold_k}] DONE in {fold_elapsed/60:.1f}min, best_val_loss={best_loss:.4f}")
    return best_loss


fold_results = {}
for fold_k in range(N_FOLDS):
    bl = train_fold(fold_k)
    fold_results[fold_k] = bl
    gc.collect()
    torch.cuda.empty_cache()

print(f"\\nAll folds DONE: {fold_results}")
print(f"Total time: {(time.time()-START)/60:.1f} min")
"""))


# ─────────── Cell 10: Save summary ───────────
cells.append(code_cell("""# Save species index + summary
with open(OUT_DIR / "m7_species_63.json", "w") as f:
    json.dump(SPECIES_63, f, indent=2)
with open(OUT_DIR / "m7_label2idx.json", "w") as f:
    json.dump(LABEL2IDX, f, indent=2)
with open(OUT_DIR / "m7_summary.json", "w") as f:
    json.dump({
        "n_classes": N_CLASSES,
        "n_folds": N_FOLDS,
        "n_epochs": N_EPOCHS,
        "fold_best_loss": fold_results,
        "total_time_min": (time.time() - START) / 60,
    }, f, indent=2)
print(f"OK M7 training DONE: {sorted(OUT_DIR.glob('m7_*'))}")
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
print(f"Built: {OUT_PATH} ({len(cells)} cells)")
