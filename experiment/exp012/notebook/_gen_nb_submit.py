"""Generate exp012 nb_submit.ipynb (Kaggle CPU inference).

Strategy:
  - Load fold0_best_macro.pt from kernel_source `maekeso/birdclef2026-exp012-train-fold0`
  - Reconstruct BirdSEDModel in PyTorch (CPU)
  - Load test_soundscapes/*.ogg, compute mel via librosa, run inference
  - 0.5*sigmoid(clip) + 0.5*sigmoid(frame_max)
  - 5-tap Gaussian smoothing in logit space (Tucker's approach)
  - Write submission.csv

Future extensions:
  - Auto-detect 5-fold ONNX from `maekeso/birdclef2026-exp012-sed-onnx` (when Colab finishes)
  - If ONNX available, use ONNX ensemble. Else fall back to single-fold PyTorch.

Run: python _gen_nb_submit.py  -> writes nb_submit.ipynb
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

cells.append(md_cell(r"""# exp012 — Sanity Submit (single-fold, CPU inference)

Loads `fold0_best_macro.pt` from the trained Kaggle kernel and runs CPU PyTorch
inference on test_soundscapes. Applies 5-tap Gaussian smoothing in logit space.

## Inputs
- `birdclef-2026` (competition)
- kernel_source: `maekeso/birdclef2026-exp012-train-fold0` (provides `fold0_best_macro.pt`)

## Output
- `submission.csv`

## Expected LB
~0.85-0.90 (single fold + Gaussian smoothing, no ensemble yet)
""", "hdr"))

cells.append(code_cell(r"""# Setup
import os, sys, time, json, glob, gc, math
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
print(f"Device: {device}")

# ============================================================
# Paths
# ============================================================
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
assert COMP_DIR.exists(), f"competition dir not found"

TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
print(f"COMP_DIR: {COMP_DIR}")
print(f"TEST_DIR: {TEST_DIR}, exists={TEST_DIR.exists()}")

# Find fold0_best_macro.pt from kernel_source
CKPT_CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp012-train-fold0/fold0_best_macro.pt"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp012-train-fold0/fold0_best_ns22.pt"),
]
CKPT_PATH = next((p for p in CKPT_CANDIDATES if p.exists()), None)
assert CKPT_PATH is not None, f"checkpoint not found in {CKPT_CANDIDATES}"
print(f"CKPT: {CKPT_PATH} ({CKPT_PATH.stat().st_size/1e6:.1f}MB)")

# ============================================================
# Constants (must match training)
# ============================================================
SR        = 32000
N_FFT     = 2048
HOP       = 512
N_MELS    = 256
FMIN      = 20
FMAX      = 16000
TOP_DB    = 80
CHUNK_SEC = 5
CHUNK_N   = SR * CHUNK_SEC          # 160_000
N_FRAMES  = CHUNK_N // HOP + 1      # 313
N_WINDOWS = 12                      # 12 * 5s = 60s
NUM_CLASSES = 234
BACKBONE_NAME = "tf_efficientnet_b0.ns_jft_in1k"
""", "setup"))

# Model definition (same as training)
cells.append(code_cell(r"""# ============================================================
# Model definition (matches training)
# ============================================================
import timm

class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)

class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, fmap):
        return self.proj(fmap.mean(dim=[2, 3]))

class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512, use_distill=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate)
        with torch.no_grad():
            dummy = torch.randn(1, 1, N_MELS, N_FRAMES)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.att = nn.Conv1d(hidden_dim, num_classes, 1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, 1, bias=True)
        if use_distill:
            self.distill_head = DistillHead(self.backbone_dim, 1536)

    def forward(self, x, return_framewise=True):
        h = self.backbone(x)
        h_cls = h.detach()  # SED head was trained with stop-grad
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise = self.cla(h_cls)
        clip = torch.sum(norm_att * framewise, dim=2)
        if return_framewise:
            return clip, framewise.permute(0, 2, 1)
        return clip

# Build & load
model = BirdSEDModel(use_distill=True).eval()
ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"loaded; missing={len(missing)} unexpected={len(unexpected)}")
if missing[:5]: print(f"  missing[:5]={missing[:5]}")
if unexpected[:5]: print(f"  unexpected[:5]={unexpected[:5]}")

# Throw away distillation head for inference (saves memory, no effect on outputs)
if hasattr(model, 'distill_head'):
    del model.distill_head
gc.collect()
print(f"model ready, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
""", "model_load"))

# Mel + inference
cells.append(code_cell(r"""# ============================================================
# Mel + inference utilities
# ============================================================
import librosa
try:
    import soundfile as sf
    DECODER = "soundfile"
except ImportError:
    DECODER = "librosa"
print(f"Audio decoder: {DECODER}")

def load_audio_32k_mono(path):
    if DECODER == "soundfile":
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    else:
        wav, _ = librosa.load(path, sr=SR, mono=True)
        return wav.astype(np.float32)

def file_to_chunks(path):
    wav = load_audio_32k_mono(path)
    target_len = 60 * SR
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    n_chunks = target_len // CHUNK_N
    chunks = wav[:n_chunks * CHUNK_N].reshape(n_chunks, CHUNK_N).astype(np.float32)
    end_times = np.arange(1, n_chunks + 1) * CHUNK_SEC
    return chunks, end_times

def audio_to_mel(chunks):
    # (N, 160000) -> (N, 1, 256, 313) normalized mel dB
    mels = []
    for i in range(chunks.shape[0]):
        S = librosa.feature.melspectrogram(
            y=chunks[i], sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        S_dB = librosa.power_to_db(S, top_db=TOP_DB)
        S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
        mels.append(S_dB)
    return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)

def model_infer_logits(mel_np):
    # mel_np: (N, 1, 256, 313) -> (clip_logits, frame_max_logits) each (N, 234)
    with torch.no_grad():
        mel = torch.from_numpy(mel_np)
        clip, framewise = model(mel, return_framewise=True)  # framewise: (N, T, 234)
        frame_max = framewise.max(dim=1).values  # (N, 234)
    return clip.numpy(), frame_max.numpy()

print("inference utilities ready")
""", "infer_utils"))

cells.append(code_cell(r"""# ============================================================
# Main inference loop
# ============================================================
from scipy.ndimage import convolve1d

GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

def gauss_smooth_logits(logits_2d):
    # (n_files * 12, 234) -> smoothed in logit space across 12 windows
    R = logits_2d.reshape(-1, N_WINDOWS, logits_2d.shape[1]).copy()
    for i in range(R.shape[0]):
        R[i] = convolve1d(R[i], GAUSSIAN_KERNEL, axis=0, mode="nearest")
    return R.reshape(-1, logits_2d.shape[1])

def sigmoid_inf(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
        np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
    ).astype(np.float32)

# Get test files (fallback to train_soundscapes for local debug)
test_files = sorted(glob.glob(f"{TEST_DIR}/*.ogg")) if TEST_DIR.is_dir() else []
if len(test_files) == 0:
    fb = COMP_DIR / "train_soundscapes"
    if fb.is_dir():
        test_files = sorted(glob.glob(f"{fb}/*.ogg"))[:5]
        print(f"No test_soundscapes; using {len(test_files)} train files for debug")
print(f"Test files: {len(test_files)}")

# Sample submission for column ordering
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, nrows=1)
PRIMARY_LABELS = list(sample_sub.columns[1:])
assert len(PRIMARY_LABELS) == NUM_CLASSES

t0 = time.time()
all_rows, all_logits = [], []
for fi, path in enumerate(test_files):
    basename = os.path.basename(path).replace(".ogg", "")
    chunks, end_times = file_to_chunks(path)
    mel = audio_to_mel(chunks)
    clip_logits, frame_max_logits = model_infer_logits(mel)
    logits = 0.5 * clip_logits + 0.5 * frame_max_logits  # (N=12, 234)

    all_rows.extend([f"{basename}_{int(t)}" for t in end_times])
    all_logits.append(logits)

    if (fi + 1) % 10 == 0 or fi == 0 or fi == len(test_files) - 1:
        el = time.time() - t0
        rate = (fi + 1) / el
        print(f"  [{fi+1:4d}/{len(test_files)}] {el:.0f}s  {rate:.2f} files/s")

logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, NUM_CLASSES), np.float32)
print(f"\nInference: {len(all_rows)} rows, {time.time()-t0:.0f}s total")

# 5-tap Gaussian smoothing in logit space, then sigmoid
logits_smoothed = gauss_smooth_logits(logits_arr)
probs = sigmoid_inf(logits_smoothed)
print(f"probs shape: {probs.shape}, mean={probs.mean():.4f}, max={probs.max():.4f}")
""", "infer_main"))

cells.append(code_cell(r"""# ============================================================
# Write submission.csv
# ============================================================
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_rows)

assert submission.shape[1] == NUM_CLASSES + 1
assert submission["row_id"].is_unique
assert not submission.iloc[:, 1:].isna().any().any()
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv: {len(submission)} rows x {submission.shape[1]} cols")
print(submission.head(3).iloc[:, :6])
""", "write_sub"))

cells.append(md_cell(r"""## Done

If LB ≥ 0.85, pipeline is healthy and we proceed with 5-fold ensemble + improvements.
""", "footer"))

nb_out = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_submit.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote: {out_path}")
print(f"Cells: {len(cells)}")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
