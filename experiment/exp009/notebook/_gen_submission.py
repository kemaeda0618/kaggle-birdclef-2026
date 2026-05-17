"""Generate exp009 submission notebook."""
import json

def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src
    }

def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": src
    }

cells = []

# ── Header ──
cells.append(md_cell("hdr", """# exp009: 5-Fold SED Submission

EfficientNet-B0 SED + 5-Fold Ensemble (Phase 3 weights)
- Mel params: n_mels=256, fmin=20, fmax=16000 (matches mel cache)
- 5-fold average for robust predictions
- CPU inference (no GPU required)"""))

# ── Imports ──
cells.append(code_cell("imports", """import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import torchaudio
import torchaudio.transforms as T
import os, time, glob
from pathlib import Path
from dataclasses import dataclass

START = time.time()
print(f"PyTorch {torch.__version__}")"""))

# ── Config & Paths ──
cells.append(code_cell("config", r"""# ══════════════════════════════════════════════════════════════
# CONFIG & PATHS
# ══════════════════════════════════════════════════════════════
@dataclass
class Config:
    sr: int = 32_000
    chunk_duration: float = 5.0
    target_size: tuple = (256, 256)

    # Mel (must match training / mel cache)
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 16_000
    top_db: float = 80.0

    # Model
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0
    n_folds: int = 5

    @property
    def chunk_samples(self) -> int:
        return int(self.sr * self.chunk_duration)

cfg = Config()

DATA_ROOT = "/kaggle/input/competitions/birdclef-2026"
TEST_DIR = os.path.join(DATA_ROOT, "test_soundscapes")
TRAIN_SC_DIR = os.path.join(DATA_ROOT, "train_soundscapes")

# Find weights from training notebook output
WEIGHT_DIR_CANDIDATES = [
    "/kaggle/input/birdclef2026-exp009-train/weights",
    "/kaggle/input/birdclef2026-exp009-train",
]

WEIGHT_DIR = None
for d in WEIGHT_DIR_CANDIDATES:
    if os.path.isdir(d) and any(f.endswith(".pth") for f in os.listdir(d)):
        WEIGHT_DIR = d
        break

if WEIGHT_DIR is None:
    # Debug: show training NB output structure only (avoid scanning mel cache)
    train_nb_root = "/kaggle/input/birdclef2026-exp009-train"
    print(f"Searching under {train_nb_root}:")
    if os.path.isdir(train_nb_root):
        for root, dirs, files in os.walk(train_nb_root):
            level = root.replace(train_nb_root, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 3:
                for f in sorted(files)[:20]:
                    print(f"{indent}  {f}")
            if any(f.endswith(".pth") for f in files):
                WEIGHT_DIR = root
                break
    else:
        print(f"  NOT FOUND: {train_nb_root}")

assert WEIGHT_DIR is not None, "Could not find weight directory"
print(f"Weight dir: {WEIGHT_DIR}")
print(f"Files: {sorted(os.listdir(WEIGHT_DIR))}")

sub_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"), nrows=1)
SPECIES = list(sub_df.columns[1:])
print(f"Species: {len(SPECIES)}")"""))

# ── Model ──
cells.append(code_cell("model", r"""# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION (must match training)
# ══════════════════════════════════════════════════════════════
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)

class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        clipwise_logit = (att * cls).sum(dim=-1)
        return {
            "clipwise_prob": torch.sigmoid(clipwise_logit),
            "segmentwise_logit": cls.permute(0, 2, 1),
        }

class SEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=False,
            in_chans=cfg.in_channels, features_only=False,
            global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(self.backbone.num_features,
                                     cfg.num_classes, cfg.dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))

# ══════════════════════════════════════════════════════════════
# MEL TRANSFORM (on-the-fly for test data)
# ══════════════════════════════════════════════════════════════
class MelSpectrogramTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=2.0,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.resize = torchvision.transforms.Resize(cfg.target_size, antialias=True)

    @torch.no_grad()
    def forward(self, waveforms):
        mel = self.db(self.mel(waveforms))
        mel = self.resize(mel)
        B = mel.shape[0]
        mel_flat = mel.reshape(B, -1)
        mel_min = mel_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = mel_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)

mel_transform = MelSpectrogramTransform(cfg)
mel_transform.eval()
print("Mel transform ready")"""))

# ── Load Models ──
cells.append(code_cell("load-models", r"""# ══════════════════════════════════════════════════════════════
# LOAD 5-FOLD MODELS
# ══════════════════════════════════════════════════════════════
models = []
for fold in range(cfg.n_folds):
    weight_path = os.path.join(WEIGHT_DIR, f"best_fold{fold}.pth")
    if not os.path.exists(weight_path):
        print(f"  Fold {fold}: NOT FOUND at {weight_path}, skipping")
        continue
    model = SEDModel(cfg)
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    auc = ckpt.get("metrics", {}).get("macro_auc", "?")
    print(f"  Fold {fold}: loaded (AUC={auc})")
    models.append(model)

print(f"\nLoaded {len(models)} fold models")
assert len(models) > 0, "No models loaded!" """))

# ── Find Test Files ──
cells.append(code_cell("test-files", r"""# ══════════════════════════════════════════════════════════════
# FIND TEST FILES
# ══════════════════════════════════════════════════════════════
test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.ogg")))
if len(test_files) == 0:
    print("No test soundscapes found, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(os.path.join(TRAIN_SC_DIR, "*.ogg")))[:8]
print(f"Test files: {len(test_files)}")"""))

# ── Inference ──
cells.append(code_cell("inference", r"""# ══════════════════════════════════════════════════════════════
# INFERENCE: 5-FOLD ENSEMBLE
# ══════════════════════════════════════════════════════════════
CHUNK = cfg.chunk_samples
BATCH_SIZE = 32
all_row_ids = []
all_preds = []

print(f"Running inference with {len(models)}-fold ensemble...")
t0 = time.time()

for fpath in test_files:
    stem = Path(fpath).stem

    # Load audio
    audio, sr = torchaudio.load(fpath)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != cfg.sr:
        audio = torchaudio.functional.resample(audio, sr, cfg.sr)
    audio = audio.squeeze(0).numpy()

    # Split into 5s chunks
    n_chunks = max(1, len(audio) // CHUNK)
    padded_len = n_chunks * CHUNK
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))
    else:
        audio = audio[:padded_len]

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    chunks_tensor = torch.from_numpy(audio.reshape(n_chunks, CHUNK)).float()

    # Compute mel once (shared across models)
    with torch.no_grad():
        all_mels = []
        for bi in range(0, n_chunks, BATCH_SIZE):
            batch = chunks_tensor[bi:bi + BATCH_SIZE]
            mel = mel_transform(batch)
            all_mels.append(mel)
        mels = torch.cat(all_mels, dim=0)  # (n_chunks, 3, 256, 256)

    # Run each model and average
    fold_preds = []
    with torch.no_grad():
        for model in models:
            model_probs = []
            for bi in range(0, len(mels), BATCH_SIZE):
                batch = mels[bi:bi + BATCH_SIZE]
                probs = model(batch)["clipwise_prob"].numpy()
                model_probs.append(probs)
            fold_preds.append(np.concatenate(model_probs, axis=0))

    # Average across folds
    probs = np.mean(fold_preds, axis=0)  # (n_chunks, 234)

    for i in range(n_chunks):
        end_sec = (i + 1) * int(cfg.chunk_duration)
        all_row_ids.append(f"{stem}_{end_sec}")
        all_preds.append(probs[i])

elapsed = time.time() - t0
print(f"  Inference done: {len(all_row_ids)} predictions in {elapsed:.1f}s")"""))

# ── Build Submission ──
cells.append(code_cell("submission", r"""# ══════════════════════════════════════════════════════════════
# BUILD SUBMISSION
# ══════════════════════════════════════════════════════════════
preds_array = np.stack(all_preds)
submission = pd.DataFrame(preds_array, columns=SPECIES)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])

missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling with zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in SPECIES:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)

extra = our_ids - expected_ids
if extra:
    print(f"Dropping {len(extra)} extra row_ids")
    submission = submission[submission["row_id"].isin(expected_ids)]

submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

total_time = time.time() - START
print(f"\nSubmission saved: {submission.shape}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Mean prediction: {submission[SPECIES].values.mean():.6f}")
print(f"Max prediction:  {submission[SPECIES].values.max():.6f}")
print(submission.head())"""))

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = "C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp009/notebook/submission.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
