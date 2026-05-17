"""Generate exp017 inference NB (Kaggle CPU 90min, internet=off).

Loads ckpt_best_ns22.pth from Kaggle Dataset `maekeso/birdclef2026-exp017-weights`
(uploaded from train NB Colab Cell 17), rebuilds eca_nfnet_l0 SED model,
runs inference on test_soundscapes (700 files × 12 windows), writes submission.csv.

The distill head is unused at inference (just SED clip + frame_max blend + Gaussian smooth).
R1 v2: val_ns22 0.9166 (after LR fix from v1 0.9065)
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

cells.append(md_cell(r"""# exp017 — Inference (Kaggle CPU 90min, internet=off)

Load best eca_nfnet_l0 SED ckpt from Kaggle Dataset, run inference on test_soundscapes,
write submission.csv.

## Constraints
- Kaggle Code-Only competition: CPU only, no internet, 90 min total
- Submit format: `row_id` + 234 species columns

## Inputs (attach when pushing)
- `birdclef-2026` (competition data; test_soundscapes is auto-mounted at score time)
- `maekeso/birdclef2026-exp017-weights` (R1 v2 / R2 ckpts; uploaded from train NB Cell 17)

## Pipeline
1. Locate ckpt under /kaggle/input
2. Rebuild model architecture (identical to training)
3. Per test file: decode -> 12 chunks (5s each) -> mel -> model -> sigmoid
4. Blend in **logit space**: 0.5*clip + 0.5*frame_max, then Gaussian smooth, then sigmoid
5. Write submission.csv
""", "hdr"))

cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup
# ============================================================
import os, sys, time, json, math, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_num_threads(4)
""", "setup"))

cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths — locate competition data + R1 ckpt
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"

TEST_DIR = BASE / "test_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
print(f"BASE: {BASE}")
print(f"  test_soundscapes exists: {TEST_DIR.exists()}")

# Locate ckpt under /kaggle/input
STATE_DIR = None
CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights"),
    Path("/kaggle/input/birdclef2026-exp017-weights"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp017-train"),
]
for p in CANDIDATES:
    if p.exists():
        if any(p.rglob("ckpt_best_*.pth")) or any(p.rglob("ckpt_latest*.pth")):
            STATE_DIR = p; break

if STATE_DIR is None:
    for hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        STATE_DIR = hit.parent; break

assert STATE_DIR is not None, (
    "ckpt not found. Attach maekeso/birdclef2026-exp017-weights as dataset_sources"
)
print(f"State dir: {STATE_DIR}")
for f in sorted(STATE_DIR.rglob("ckpt_*.pth"))[:6]:
    print(f"  {f.relative_to(STATE_DIR)!s}  {f.stat().st_size/1e6:.2f} MB")
""", "paths"))

cells.append(code_cell(r"""# ============================================================
# Cell 3: Config — must match training exactly
# ============================================================
NUM_CLASSES = 234
SR = 32000

TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 256
FMIN       = 20
FMAX       = 16000

BACKBONE = "eca_nfnet_l0"

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES

print(f"Backbone: {BACKBONE}")
""", "config"))

cells.append(code_cell(r"""# ============================================================
# Cell 4: Model — rebuild eca_nfnet_l0 SED architecture
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


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


# Load best ckpt — prefer R2 (r2_* prefix) over R1
CKPT_PRIORITY = [
    "r2_ckpt_best_ns22.pth",   # R2 best (TBD val)
    "r2_ckpt_best_macro.pth",
    "r2_ckpt_latest.pth",
    "ckpt_best_ns22.pth",      # R1 v2 fallback (val 0.9166)
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]
ckpt_path = None
for name in CKPT_PRIORITY:
    hits = list(STATE_DIR.rglob(name))
    if hits:
        ckpt_path = hits[0]; break
assert ckpt_path is not None, f"No ckpt found under {STATE_DIR}"
print(f"Loading ckpt: {ckpt_path.name}")

try:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(ckpt_path), map_location="cpu")
print(f"  epoch={state.get('epoch')}, "
      f"best_ns22={state.get('best_ns22', float('nan')):.4f}, "
      f"best_macro={state.get('best_macro', float('nan')):.4f}")

model = BirdSEDModel().to(device)
model.load_state_dict(state["model_state"], strict=False)
model.eval()
print(f"OK model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
""", "model"))

cells.append(code_cell(r"""# ============================================================
# Cell 5: Inference — main loop
# ============================================================
try:
    import soundfile as sf
    DECODER = "soundfile"
except ImportError:
    DECODER = "librosa"
print(f"Audio decoder: {DECODER}")

import librosa
from scipy.ndimage import convolve1d

GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
N_WINDOWS = 12
CHUNK_N = SR * TRAIN_DURATION

def load_audio_32k_mono(path):
    if DECODER == "soundfile":
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    else:
        wav, _ = librosa.load(str(path), sr=SR, mono=True)
        return wav.astype(np.float32)

def file_to_chunks(path):
    wav = load_audio_32k_mono(path)
    target_len = 60 * SR
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    chunks = wav.reshape(N_WINDOWS, CHUNK_N)
    end_times = np.arange(1, N_WINDOWS + 1) * TRAIN_DURATION
    return chunks.astype(np.float32), end_times

def sigmoid_np(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                    np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50)))
                    ).astype(np.float32)

def gauss_smooth(scores):
    smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], GAUSSIAN_KERNEL, axis=0, mode="nearest")
    return smoothed.reshape(-1, scores.shape[1])

mel_tf = MelSpecTransform().to(device)

# Discover test files (with train_soundscapes fallback for local debug)
test_files = sorted(glob.glob(f"{TEST_DIR}/*.ogg")) if TEST_DIR.is_dir() else []
if len(test_files) == 0:
    fallback = BASE / "train_soundscapes"
    if fallback.is_dir():
        test_files = sorted(glob.glob(f"{fallback}/*.ogg"))[:5]
        print(f"No test_soundscapes — using {len(test_files)} train files for debug")
print(f"Test files: {len(test_files)}")

all_rows, all_logits = [], []
t0 = time.time()

with torch.no_grad():
    for fi, fp in enumerate(test_files):
        basename = os.path.basename(fp).replace(".ogg", "")
        chunks, end_times = file_to_chunks(fp)

        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)
        mel = mel_tf(wav_t)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)

        clip_logits, framewise = model(mel, return_framewise=True)
        frame_max = framewise.max(dim=1).values
        blend_logits = 0.5 * clip_logits + 0.5 * frame_max
        all_rows.extend([f"{basename}_{int(t)}" for t in end_times])
        all_logits.append(blend_logits.float().cpu().numpy())

        if (fi + 1) % 50 == 0 or fi == 0 or fi == len(test_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 1e-6)
            eta = (len(test_files) - fi - 1) / max(rate, 1e-6)
            print(f"  [{fi+1:4d}/{len(test_files)}]  {elapsed:.1f}s  {rate:.2f} files/s  ETA {eta/60:.1f}min")

if all_logits:
    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
    logits_smoothed = gauss_smooth(logits_arr)
    probs = sigmoid_np(logits_smoothed)
else:
    probs = np.zeros((0, NUM_CLASSES), dtype=np.float32)

print(f"\nInference: {len(all_rows)} rows in {(time.time()-t0)/60:.1f} min")
""", "infer"))

cells.append(code_cell(r"""# ============================================================
# Cell 6: Write submission.csv
# ============================================================
sub = pd.DataFrame(probs, columns=PRIMARY_LABELS)
sub.insert(0, "row_id", all_rows)
out_path = Path("/kaggle/working/submission.csv")
sub.to_csv(out_path, index=False)
print(f"submission.csv: {len(sub)} rows, {sub.shape[1]-1} species, "
      f"{out_path.stat().st_size/1e6:.1f}MB")
print(sub.head(3))
""", "submit"))

# Assemble
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
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_infer.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
