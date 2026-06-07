"""exp041: BirdSet EffNet-B1 BC2026 fine-tune inference NB (Kaggle CPU 90 min).

Loads fine-tuned weights from `maekeso/birdclef2026-exp041-birdset-b1-weights`
(uploaded from Colab train NB).

Pipeline:
- audio (60s @ 32kHz) → 12 × 5s windows
- mel-spec (BirdSet 仕様: n_fft=2048, hop=2048, n_mels=256, normalize -4.268/4.569)
- EfficientNet-B1 (HF transformers) → 234-class logits
- sigmoid → submission.csv

Expected LB: 0.93+ (Antoine 0.936 with B0)
Run time: 15-30 min on Kaggle CPU
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr", r"""# exp041 — BirdSet EffNet-B1 BC2026 inference (Kaggle CPU)

Antoine 路線 (discussion 700763) の 4th stream: Perch 完全独立軸。

## Inputs (要 attach、V2 SDK bug 注意)
- `birdclef-2026` (competition)
- `maekeso/birdclef2026-exp041-birdset-b1-weights` (fine-tuned ckpt + birdset_config.json)

## Pipeline
1. 60s test audio → 12 × 5s windows
2. mel-spec (BirdSet 仕様: hop=2048、normalize -4.268/4.569)
3. EfficientNet-B1 (HF transformers) forward
4. sigmoid → 234-class probs
5. submission.csv

期待 LB: 0.93+、Run time: 15-30 min
"""))

# Setup
cells.append(code_cell("setup", r"""# Setup
import os, sys, time, json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
torch.set_num_threads(4)
print(f"Device: {device}")
"""))

# Paths
cells.append(code_cell("paths", r"""# Locate competition data + exp041 ckpt
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"), Path("/kaggle/input/birdclef-2026")]:
    if p.exists(): BASE = p; break
assert BASE is not None, "BC2026 data not found"
TEST_DIR = BASE / "test_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
print(f"BASE: {BASE}")
print(f"  test_soundscapes exists: {TEST_DIR.exists()}")

# Locate exp041 weights Dataset
STATE_DIR = None
for p in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp041-birdset-b1-weights"),
    Path("/kaggle/input/birdclef2026-exp041-birdset-b1-weights"),
]:
    if p.exists() and (any(p.rglob("*ckpt_best_ns22.pth")) or any(p.rglob("*ckpt_latest*.pth"))):
        STATE_DIR = p; break
if STATE_DIR is None:
    for hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        if "exp041" in str(hit).lower():
            STATE_DIR = hit.parent; break
assert STATE_DIR is not None, "exp041 ckpt not found. Attach maekeso/birdclef2026-exp041-birdset-b1-weights"
print(f"State dir: {STATE_DIR}")
for f in sorted(STATE_DIR.rglob("*.pth"))[:5]:
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
"""))

# Config
cells.append(code_cell("config", r"""# Config — must match training
NUM_CLASSES = 234
SR = 32000
DURATION_SEC = 5
SAMPLES_PER_CHUNK = SR * DURATION_SEC
N_WINDOWS = 12
N_FFT = 2048
HOP_LENGTH = 2048
N_MELS = 256
FMIN = 0
FMAX = None
POWER = 2.0
TOP_DB = 80.0
NORM_MEAN = -4.268
NORM_STD = 4.569

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
"""))

# Load model
cells.append(code_cell("model", r"""# Build EfficientNet-B1 (HF transformers) with BirdSet config + load fine-tuned weights
from transformers import EfficientNetForImageClassification, EfficientNetConfig

# Find birdset_config.json (from training upload)
config_path = None
for hit in STATE_DIR.rglob("birdset_config.json"):
    config_path = hit; break
if config_path is None:
    for hit in STATE_DIR.rglob("config.json"):
        config_path = hit; break
assert config_path is not None, "birdset_config.json not found"
print(f"Config: {config_path}")

# Load config dict from JSON directly (避: from_pretrained の config_file_name 引数 NG)
with open(config_path) as _f:
    _cfg_dict = json.load(_f)
# Override for BC2026 fine-tune compatibility
_cfg_dict["num_labels"] = NUM_CLASSES   # 234 (override BirdSet 9736)
_cfg_dict["num_channels"] = 1
_cfg_dict.pop("id2label", None)   # 9736 entries 削除 (num_labels と矛盾防止)
_cfg_dict.pop("label2id", None)
cfg = EfficientNetConfig(**_cfg_dict)

# Build model from config (no pretrain load, weights come from our ckpt)
model = EfficientNetForImageClassification(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.1f}M")
print(f"  num_labels: {cfg.num_labels}, num_channels: {cfg.num_channels}")
assert 5_000_000 < n_params < 15_000_000, f"unexpected param count {n_params/1e6:.1f}M (EffNet-B1 should be ~8M)"

# Load fine-tuned ckpt
ckpt_path = None
for name in ["ckpt_best_ns22.pth", "ckpt_latest.pth"]:
    hits = list(STATE_DIR.rglob(name))
    if hits:
        ckpt_path = hits[0]; break
assert ckpt_path is not None, "no ckpt"
print(f"Loading: {ckpt_path.name}")

try:
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
except TypeError:
    state = torch.load(str(ckpt_path), map_location=device)
msg = model.load_state_dict(state["model_state"], strict=False)
print(f"  loaded (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
if msg.missing_keys:
    print(f"    missing (sample): {msg.missing_keys[:5]}")
if msg.unexpected_keys:
    print(f"    unexpected (sample): {msg.unexpected_keys[:5]}")
print(f"  epoch={state.get('epoch')}, best_val_auc={state.get('best_val_auc', float('nan')):.4f}")
model.eval()
"""))

# Mel-spec transform
cells.append(code_cell("mel", r"""# Mel-spec (matching training, BirdSet 仕様)
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=POWER,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
    def forward(self, wav):
        if wav.ndim == 1: wav = wav.unsqueeze(0)
        mel = self.db_transform(self.mel_spec(wav))
        mel = (mel - NORM_MEAN) / NORM_STD
        return mel  # (B, 256, T_mel)

mel_tf = MelSpecTransform().to(device)
"""))

# Inference loop
cells.append(code_cell("infer", r"""# Inference loop
try:
    import soundfile as sf
    DECODER = "soundfile"
except ImportError:
    DECODER = "librosa"
import librosa
print(f"Audio decoder: {DECODER}")


def load_audio_60s(path, sr=SR):
    if DECODER == "soundfile":
        wav, fsr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if fsr != sr:
            wav = librosa.resample(wav, orig_sr=fsr, target_sr=sr)
    else:
        wav, _ = librosa.load(str(path), sr=sr, mono=True)
        wav = wav.astype(np.float32)
    target = 60 * sr
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    return wav.astype(np.float32)


test_files = sorted(glob.glob(f"{TEST_DIR}/*.ogg")) if TEST_DIR.is_dir() else []
if not test_files:
    fallback = BASE / "train_soundscapes"
    if fallback.is_dir():
        test_files = sorted(glob.glob(f"{fallback}/*.ogg"))[:5]
        print(f"No test_soundscapes — fallback {len(test_files)} train files")
print(f"Test files: {len(test_files)}")

all_rows, all_probs = [], []
t0 = time.time()

BATCH_SIZE = 12  # 1 file = 12 windows

with torch.no_grad():
    for fi, fp in enumerate(test_files):
        stem = Path(fp).stem
        wav = load_audio_60s(fp)
        chunks = wav.reshape(N_WINDOWS, SAMPLES_PER_CHUNK)
        wav_t = torch.from_numpy(chunks).float()       # (12, 160000)
        mel = mel_tf(wav_t).unsqueeze(1)               # (12, 1, 256, ~78)
        out = model(pixel_values=mel).logits           # (12, 234)
        probs = torch.sigmoid(out).cpu().numpy()
        end_times = np.arange(1, N_WINDOWS + 1) * DURATION_SEC
        all_rows.extend([f"{stem}_{int(t)}" for t in end_times])
        all_probs.append(probs)
        if (fi + 1) % 50 == 0 or fi == 0 or fi == len(test_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 1e-6)
            eta = (len(test_files) - fi - 1) / max(rate, 1e-6)
            print(f"  [{fi+1:4d}/{len(test_files)}] {elapsed:.0f}s  {rate:.2f} files/s  ETA {eta/60:.1f}min")

if all_probs:
    probs_arr = np.concatenate(all_probs, axis=0).astype(np.float32)
else:
    probs_arr = np.zeros((0, NUM_CLASSES), dtype=np.float32)
print(f"\\nInference: {len(all_rows)} rows in {(time.time()-t0)/60:.1f} min")
"""))

# Submission
cells.append(code_cell("submit", r"""# Write submission.csv
sub = pd.DataFrame(probs_arr, columns=PRIMARY_LABELS)
sub.insert(0, "row_id", all_rows)

# Align to sample_sub row_id order
expected_ids = set(sample_sub["row_id"])
our_ids = set(sub["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"  WARN: {len(missing)} missing row_ids — filling zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    sub = pd.concat([sub, missing_df], ignore_index=True)
extra = our_ids - expected_ids
if extra:
    sub = sub[sub["row_id"].isin(expected_ids)]
sub = sub.set_index("row_id").loc[sample_sub["row_id"]].reset_index()

out_path = Path("/kaggle/working/submission.csv")
sub.to_csv(out_path, index=False)
print(f"submission.csv: {len(sub)} rows, {out_path.stat().st_size/1e6:.1f}MB")
print(sub.head(3))
"""))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_infer_birdset_b1.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path.name}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
