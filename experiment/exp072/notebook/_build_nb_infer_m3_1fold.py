"""Build M3' best 1-fold PyTorch inference NB (Kaggle CPU sub).

M3' has only .pth (ONNX export failed in train NB due to onnxscript missing).
Use PyTorch CPU inference, 1-fold only for safety.

Architecture: CLEFClassifierSED (Babych SED, 20s mel, in_chans=3)
Best fold: fold 1 (val_ns22 0.9536)

Required kernel_sources:
  - maekeso/birdclef2026-exp072-m3-nfnet-l0-20s-babych (M3' train NB output)
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_infer_m3_1fold.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# exp072 M3' best 1-fold PyTorch inference (Kaggle CPU sub)

**Strategy**: M3' V4 fold 1 (val_ns22 0.9536) を PyTorch CPU で 1-fold sub
- ONNX export failed (onnxscript not available in M3' train NB)
- PyTorch .pth direct load via kernel_sources
- 1-fold only (90min safety)

**Architecture (CLEFClassifierSED, M3' V4)**:
- Backbone: eca_nfnet_l0 (features_only=True, in_chans=3)
- Mel: 20s × 224 × 4096 × 1252
- Head: AttHead (GeMFreq + Dense + Conv1d)
- Output: clip_logit (234) + framewise_logit (234, T)

**Expected runtime**: 30-45min
""")

# =============================================================
code("""# Cell 1: Setup + Imports
import time, gc, math, warnings
warnings.filterwarnings("ignore")
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
import timm

torch.set_num_threads(4)
print(f"torch: {torch.__version__}, timm: {timm.__version__}")
START = time.time()
""")

# =============================================================
code("""# Cell 2: Config (M3' V4 spec)
NUM_CLASSES = 234
SR = 32000
WINDOW_SEC = 20                  # ★ M3' Babych 20s
CHUNK_SAMPLES = SR * WINDOW_SEC  # 640,000
N_WINDOWS = 12                   # 12 chunks per file (60s / 5s)
WINDOW_SEC_INFER = 5             # inference chunk (5s alignment for row_id)
CHUNK_SAMPLES_INFER = SR * WINDOW_SEC_INFER

# Mel (Babych spec)
N_MELS = 224
N_FFT = 4096
HOP_LENGTH = 1252
F_MIN = 0
F_MAX = 16000
TOP_DB = 80

BACKBONE = "eca_nfnet_l0.ra2_in1k"
DROP_PATH = 0.0   # inference

BATCH_SIZE = 8   # 20s mel batch (memory考慮)

# Best fold
BEST_FOLD = 1   # ★ M3' fold 1 = val_ns22 0.9536

# Paths
DATA_PATHS = ["/kaggle/input/competitions/birdclef-2026",
              "/kaggle/input/birdclef-2026"]
DATA_PATH = next((Path(p) for p in DATA_PATHS if Path(p).exists()), None)
assert DATA_PATH is not None
print(f"Data path: {DATA_PATH}")

TEST_DIR = DATA_PATH / "test_soundscapes"
SAMPLE_SUB_PATH = DATA_PATH / "sample_submission.csv"

# M3' train NB output (via kernel_sources)
M3_CKPT_DIR = None
candidates = [
    Path("/kaggle/input/birdclef2026-exp072-m3-nfnet-l0-20s-babych"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp072-m3-nfnet-l0-20s-babych"),
]
for p in candidates:
    if p.exists():
        M3_CKPT_DIR = p; break
assert M3_CKPT_DIR is not None, "M3 train NB output not attached"
print(f"M3 ckpt dir: {M3_CKPT_DIR}")
print(f"  files: {[f.name for f in sorted(M3_CKPT_DIR.iterdir())[:20]]}")

pth_paths = sorted(M3_CKPT_DIR.glob("m3_fold*_ckpt_best.pth"))
assert len(pth_paths) > 0, f"No m3 ckpt found in {M3_CKPT_DIR}"
print(f"\\nFound {len(pth_paths)} .pth ckpts")
for p in pth_paths:
    print(f"  {p.name}")

# Select best fold ckpt
import re
best_pth = None
for p in pth_paths:
    m = re.search(r"fold(\\d+)", p.name)
    if m and int(m.group(1)) == BEST_FOLD:
        best_pth = p; break
if best_pth is None:
    best_pth = pth_paths[0]
print(f"\\nSelected: {best_pth.name}  (BEST_FOLD={BEST_FOLD})")
""")

# =============================================================
code("""# Cell 3: Load BC26 labels
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"BC26 labels: {len(PRIMARY_LABELS)}")
""")

# =============================================================
code("""# Cell 4: M3' Architecture (CLEFClassifierSED)
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
    def __init__(self, in_chans, p=0.5, num_class=NUM_CLASSES, hidden_dim=512):
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
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES, drop_path_rate=DROP_PATH):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)

    def forward(self, wav, return_framewise=False):
        spec = self.mel_spectr_generator(wav)
        spec3 = torch.stack([spec, spec, spec], 1)
        feat = self.backbone(spec3)[-1]
        head_output = self.head(feat)
        framewise_logit = head_output["framewise_logit"]
        clip_logit = framewise_logit.max(dim=-1).values
        if return_framewise:
            return clip_logit, framewise_logit.permute(0, 2, 1)
        return clip_logit


print("OK model defs")
""")

# =============================================================
code("""# Cell 5: Load model from best_pth
DEVICE = torch.device("cpu")

model = CLEFClassifierSED().to(DEVICE)
ckpt = torch.load(str(best_pth), map_location="cpu", weights_only=False)
state = ckpt.get("model_state", ckpt)
msg = model.load_state_dict(state, strict=False)
print(f"  Load state: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
if msg.missing_keys:
    print(f"    missing samples: {msg.missing_keys[:3]}")
if msg.unexpected_keys:
    print(f"    unexpected samples: {msg.unexpected_keys[:3]}")
val_ns22 = ckpt.get("val_ns22", -1)
val_macro = ckpt.get("val_macro", -1)
print(f"  ckpt val_ns22={val_ns22:.4f} val_macro={val_macro:.4f}")
model.eval()
print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
""")

# =============================================================
code("""# Cell 6: Test file enumeration + empty handling
test_files = sorted(TEST_DIR.glob("*.ogg"))
print(f"Test soundscape files: {len(test_files)}")

HAS_TEST_FILES = len(test_files) > 0
if not HAS_TEST_FILES:
    print("[INFO] No test files (local commit run) — placeholder submission")
    placeholder = sample_sub.copy()
    for col in PRIMARY_LABELS:
        placeholder[col] = 0.5
    placeholder.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"  Placeholder saved")
""")

# =============================================================
code("""# Cell 7: Audio + inference helpers
def load_audio_60s(path):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return np.zeros(SR * 60, dtype=np.float32)


def build_20s_chunks(wav_60s):
    \"\"\"Generate 12 chunks of 20s windows centered at each 5s position.

    Test row_id format: {fname}_5, {fname}_10, ..., {fname}_60
    For each chunk_idx in 0..11:
      center = (chunk_idx + 0.5) * 5  (= 2.5, 7.5, ..., 57.5)
      20s window: center ± 10s
    \"\"\"
    chunks = np.zeros((N_WINDOWS, CHUNK_SAMPLES), dtype=np.float32)
    target_len = SR * 60
    if len(wav_60s) < target_len:
        wav_60s = np.pad(wav_60s, (0, target_len - len(wav_60s)))
    elif len(wav_60s) > target_len:
        wav_60s = wav_60s[:target_len]
    for ci in range(N_WINDOWS):
        center_sec = (ci + 0.5) * WINDOW_SEC_INFER
        start_sec = max(0, center_sec - WINDOW_SEC / 2)
        # If center near end, shift window left
        end_sec = start_sec + WINDOW_SEC
        if end_sec > 60:
            start_sec = 60 - WINDOW_SEC
        start = int(start_sec * SR)
        end = start + CHUNK_SAMPLES
        ch = wav_60s[start:end]
        if len(ch) < CHUNK_SAMPLES:
            ch = np.pad(ch, (0, CHUNK_SAMPLES - len(ch)))
        # Normalize chunk amplitude
        m = np.abs(ch).max()
        if m > 0: ch = ch / m
        chunks[ci] = ch
    return chunks


print("OK helpers")
""")

# =============================================================
code("""# Cell 8: Inference loop (PyTorch CPU, 1-fold)
if HAS_TEST_FILES:
    t0 = time.time()
    print(f"Inferring on {len(test_files)} test files with fold {BEST_FOLD}...")

    all_preds = []
    file_ids = []

    with torch.no_grad():
        for fi, f in enumerate(test_files):
            wav = load_audio_60s(f)
            chunks_20s = build_20s_chunks(wav)  # (12, 640000)
            chunks_t = torch.from_numpy(chunks_20s).to(DEVICE)
            # Run in mini-batches (CPU memory friendly)
            file_preds = np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)
            for s in range(0, N_WINDOWS, BATCH_SIZE):
                batch = chunks_t[s:s+BATCH_SIZE]
                clip_logit, framewise = model(batch, return_framewise=True)
                # framewise is (B, T, num_classes), take frame max
                frame_max = framewise.max(dim=1).values
                p_clip = torch.sigmoid(clip_logit).numpy()
                p_fmax = torch.sigmoid(frame_max).numpy()
                p_blend = 0.5 * p_clip + 0.5 * p_fmax
                file_preds[s:s+len(p_blend)] = p_blend
            all_preds.append(file_preds)
            fname = f.stem
            for c in range(N_WINDOWS):
                end_sec = (c + 1) * 5
                file_ids.append(f"{fname}_{end_sec}")
            if (fi + 1) % 50 == 0:
                print(f"  {fi+1}/{len(test_files)} ({(time.time()-t0)/60:.1f}min)")

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"\\nAll preds: {all_preds.shape}")
    print(f"  stats: mean={all_preds.mean():.4f}, max={all_preds.max():.4f}, min={all_preds.min():.4f}")
    print(f"  Inference time: {(time.time()-t0)/60:.1f}min")
else:
    all_preds = None
    file_ids = []
""")

# =============================================================
code("""# Cell 9: Build submission + save
if HAS_TEST_FILES:
    sub_df = pd.DataFrame(all_preds, columns=PRIMARY_LABELS)
    sub_df.insert(0, "row_id", file_ids)

    expected_rows = len(sample_sub)
    print(f"Submission rows: {len(sub_df)} (expected {expected_rows})")
    if len(sub_df) != expected_rows:
        print(f"[WARN] Row mismatch, aligning...")
        sub_map = sub_df.set_index("row_id")
        aligned = sample_sub[["row_id"]].copy()
        for lbl in PRIMARY_LABELS:
            if lbl in sub_map.columns:
                aligned[lbl] = aligned["row_id"].map(sub_map[lbl]).fillna(0.0)
            else:
                aligned[lbl] = 0.0
        sub_df = aligned

    out_path = Path("/kaggle/working/submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"\\nOK Submission saved: {out_path} ({out_path.stat().st_size/1e6:.1f}MB)")
else:
    print("[INFO] Placeholder already written")

print(f"\\nTotal time: {(time.time()-START)/60:.1f}min")
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_OUT}")
for i, c in enumerate(CELLS):
    src = "".join(c.get("source", []))
    head = src.split('\n')[0][:70] if src else "(empty)"
    print(f"  Cell {i:2d} ({c['cell_type']:8s}) {len(src):6d} chars | {head}")
