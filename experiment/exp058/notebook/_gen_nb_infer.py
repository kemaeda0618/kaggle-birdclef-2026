"""Generate exp058 standalone inference NB (Kaggle CPU, internet=off).

Loads exp058 effv2_b0 SED ckpt, runs inference on test_soundscapes, outputs submission.csv.
For LB verification of exp058 standalone.
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp058\notebook\nb_infer.ipynb")


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# exp058 Standalone Inference (Kaggle CPU)

**Purpose**: exp058 effv2_b0 SED single LB 確認

**Model**: effv2_b0 + SED head
- Training: BC2026 train_audio (hard) + XC pseudo (frame-level from exp020 R2 5-fold teacher)
- Best ckpt: ep 11, val_macro 0.9678, val_ns22 0.8341

**Input**:
- birdclef-2026 (test_soundscapes)
- `maekeso/birdclef2026-exp058-effv2b0-combined` (effv2b0_combined_best.pth)

**Expected LB**: 0.91-0.94 (single model)

**CPU time**: ~30-50 min
"""),

        code_cell("setup", """import os, sys, time, gc, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d

import warnings; warnings.filterwarnings("ignore")
device = torch.device("cpu")
torch.set_num_threads(4)
print(f"Device: {device}")
"""),

        code_cell("paths", """# Locate competition data + exp058 ckpt
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None
TEST_DIR = BASE / "test_soundscapes"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
print(f"BASE: {BASE}")
print(f"  test files: {len(list(TEST_DIR.glob('*.ogg'))) if TEST_DIR.exists() else 0}")

CKPT_CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp058-effv2b0-combined"),
    Path("/kaggle/input/birdclef2026-exp058-effv2b0-combined"),
]
CKPT_DIR = None
for p in CKPT_CANDIDATES:
    if p.exists() and any(p.rglob("effv2b0_combined_best.pth")):
        CKPT_DIR = p; break
if CKPT_DIR is None:
    for h in Path("/kaggle/input").rglob("effv2b0_combined_best.pth"):
        CKPT_DIR = h.parent; break
assert CKPT_DIR is not None, "exp058 ckpt not found"

CKPT = next(CKPT_DIR.rglob("effv2b0_combined_best.pth"))
print(f"ckpt: {CKPT} ({CKPT.stat().st_size/1e6:.1f} MB)")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
print(f"N_CLASSES: {N_CLASSES}")
"""),

        code_cell("config", """# Config (must match NB2 training)
SR = 32000
CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC
N_WINDOWS = 12
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000
BACKBONE = "tf_efficientnetv2_b0.in1k"
HIDDEN = 512
IN_CHANS = 1
"""),

        code_cell("model", """# Model definition (matches NB2 BirdSEDModelEffv2)
class MelSpecTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x): return self.db(self.mel_spec(x))


class GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init))); self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class SEDModel(nn.Module):
    def __init__(self, num_classes=N_CLASSES, drop_path_rate=0.1, hidden_dim=HIDDEN, in_chans=IN_CHANS):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = CHUNK_SAMPLES // HOP + 1
            dummy = torch.randn(1, in_chans, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


# Load ckpt
try:
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(CKPT), map_location="cpu")
sd = state.get("state_dict", state)
print(f"ckpt ep={state.get('ep')}, val_ns22={state.get('val_ns22'):.4f}, val_macro={state.get('val_macro'):.4f}")

model = SEDModel().to(device)
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"  missing={len(miss)}, unexpected={len(unexp)}")
model.eval()
mel_tf = MelSpecTF().to(device)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
"""),

        code_cell("inference", """# Inference loop
test_files = sorted(TEST_DIR.glob("*.ogg"))
print(f"test files: {len(test_files)}")

probs_all = []  # (N_files, 12, N_CLASSES)
all_row_ids = []
GAUSS_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

t0 = time.time()
with torch.no_grad():
    for fi, fp in enumerate(test_files):
        try:
            wav, sr = sf.read(str(fp), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        except Exception as e:
            print(f"  err loading {fp.name}: {e}")
            wav = np.zeros(60 * SR, dtype=np.float32)

        # pad/cut to 60s
        target_len = 60 * SR
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]

        # 12 chunks of 5s
        chunks = wav.reshape(N_WINDOWS, CHUNK_SAMPLES).astype(np.float32)
        wav_t = torch.from_numpy(chunks).unsqueeze(1)  # (12, 1, T)
        mel = mel_tf(wav_t)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        clip_logits, frame_logits = model(mel, return_framewise=True)
        frame_max = frame_logits.max(dim=1).values
        p_clip = torch.sigmoid(clip_logits).numpy().astype(np.float32)
        p_frame = torch.sigmoid(frame_max).numpy().astype(np.float32)
        p_mean = 0.5 * p_clip + 0.5 * p_frame  # (12, N_CLASSES)
        p_smooth = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_all.append(p_smooth)

        # row_ids: <filename>_<sec> per chunk (5,10,15,...,60)
        base = fp.stem
        for ci in range(N_WINDOWS):
            sec = (ci + 1) * 5
            all_row_ids.append(f"{base}_{sec}")

        if (fi + 1) % 25 == 0 or fi == len(test_files) - 1:
            elapsed = (time.time() - t0) / 60
            rate = (fi + 1) / elapsed
            eta = (len(test_files) - fi - 1) / rate
            print(f"  [{fi+1}/{len(test_files)}] {rate:.1f}/min elapsed={elapsed:.1f}min eta={eta:.1f}min")

probs_all = np.stack(probs_all).astype(np.float32)
print(f"\\ninference done: {probs_all.shape}, {(time.time()-t0)/60:.1f}min")
"""),

        code_cell("submission", """# Build submission
flat_pred = probs_all.reshape(-1, N_CLASSES).astype(np.float32)
submission = pd.DataFrame(flat_pred, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)

# Align with sample_submission row_ids
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)
extra = our_ids - expected_ids
if extra:
    submission = submission[submission["row_id"].isin(expected_ids)]
submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

print(f"submission.csv: {submission.shape}")
print(f"Mean: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"Max:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head(3))
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    cid = c.get("id", "?")
    src = "".join(c["source"])
    print(f"    {c['cell_type']:8s} id={cid:15s} lines={len(src.splitlines())}")
