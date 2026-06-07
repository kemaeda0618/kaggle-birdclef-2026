"""Build M1 ONNX inference NB for Kaggle CPU sub (fast version).

Uses onnxruntime CPU instead of PyTorch — 2-3x speedup vs original.

Required Kaggle dataset_sources:
  - maekeso/birdclef2026-exp070-m1-onnx  (M1 5-fold ONNX files)
  - romantamrazov/onnxruntime-1-24-4  (onnxruntime wheel, offline install)
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_infer_m1_onnx.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# exp070 M1 ONNX inference (5-fold ensemble, CPU)

**ONNX runtime version — 2-3x faster than PyTorch CPU**

Workflow:
1. Install onnxruntime from public wheel dataset (offline)
2. Load 5 ONNX models from `birdclef2026-exp070-m1-onnx`
3. Compute mel spec with torchaudio
4. ONNX inference per fold
5. Rank-avg ensemble → submission

**Expected runtime**: 25-30min (vs 90min PyTorch CPU)
""")

# =============================================================
code("""# Cell 1: Install onnxruntime offline + imports
import subprocess, sys, os
import importlib.util

# Check if onnxruntime already installed
if importlib.util.find_spec("onnxruntime") is None:
    # Offline install from wheel dataset
    wheel_dir = None
    for p in ["/kaggle/input/onnxruntime-1-24-4",
              "/kaggle/input/datasets/romantamrazov/onnxruntime-1-24-4"]:
        from pathlib import Path
        if Path(p).exists():
            wheel_dir = Path(p); break
    if wheel_dir is not None:
        whls = list(wheel_dir.glob("*.whl"))
        cp312_whls = [w for w in whls if "cp312" in w.name or "py3" in w.name]
        target_whl = cp312_whls[0] if cp312_whls else (whls[0] if whls else None)
        if target_whl:
            print(f"Installing onnxruntime from {target_whl.name}...")
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(target_whl)],
                               capture_output=True, text=True)
            print(r.stdout[-300:] if r.stdout else "")
            if r.returncode != 0:
                print(f"Install failed: {r.stderr[-300:]}")
        else:
            print("No wheel found in wheel_dir")
    else:
        print("[WARN] onnxruntime wheel dir not found, trying pip install (needs internet)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], capture_output=True)

import onnxruntime as ort
print(f"onnxruntime version: {ort.__version__}")
""")

# =============================================================
code("""# Cell 2: Imports + Setup
import time, gc, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import soundfile as sf

torch.set_num_threads(4)
print(f"torch: {torch.__version__}, torchaudio: {torchaudio.__version__}")
START = time.time()
""")

# =============================================================
code("""# Cell 3: Config (M1 spec)
NUM_CLASSES = 234
SR = 32000
CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC
N_WINDOWS = 12
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000

BATCH_SIZE = 32

# Paths
DATA_PATHS = ["/kaggle/input/competitions/birdclef-2026",
              "/kaggle/input/birdclef-2026"]
DATA_PATH = None
for _p in DATA_PATHS:
    if Path(_p).exists():
        DATA_PATH = Path(_p); break
assert DATA_PATH is not None

TEST_DIR = DATA_PATH / "test_soundscapes"
SAMPLE_SUB_PATH = DATA_PATH / "sample_submission.csv"

# M1 ONNX dataset
M1_ONNX_DIR = None
for _p in ["/kaggle/input/birdclef2026-exp070-m1-onnx",
           "/kaggle/input/datasets/maekeso/birdclef2026-exp070-m1-onnx"]:
    if Path(_p).exists():
        M1_ONNX_DIR = Path(_p); break
assert M1_ONNX_DIR is not None, "M1 ONNX not attached"
print(f"M1 ONNX dir: {M1_ONNX_DIR}")

onnx_paths = sorted(M1_ONNX_DIR.glob("*.onnx"))
assert len(onnx_paths) > 0, "No ONNX files found"
print(f"Found {len(onnx_paths)} ONNX files:")
for f in onnx_paths:
    print(f"  {f.name} ({f.stat().st_size/1e6:.1f}MB)")
""")

# =============================================================
code("""# Cell 4: Load BC26 labels
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"BC26 labels: {len(PRIMARY_LABELS)}")
""")

# =============================================================
code("""# Cell 5: Test file enumeration + empty handling
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
code("""# Cell 6: Mel transform + helpers
class MelSpecTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


def load_audio_60s(path):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return np.zeros(SR * 60, dtype=np.float32)


def split_into_chunks(wav, chunk_samples=CHUNK_SAMPLES, n_chunks=N_WINDOWS):
    target = chunk_samples * n_chunks
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    return np.stack([wav[i*chunk_samples:(i+1)*chunk_samples] for i in range(n_chunks)])

print("OK helpers")
""")

# =============================================================
code("""# Cell 7: Pre-compute mel for all chunks (one-shot, faster)
if HAS_TEST_FILES:
    t0 = time.time()
    print(f"Loading {len(test_files)} test files + computing mel...")

    mel_transform = MelSpecTransform()
    mel_transform.eval()

    all_mels = []
    file_ids = []

    with torch.no_grad():
        for i, f in enumerate(test_files):
            wav = load_audio_60s(f)
            chunks = split_into_chunks(wav)  # (12, 160000)
            chunks_t = torch.from_numpy(chunks).unsqueeze(1)  # (12, 1, 160000)
            mel = mel_transform(chunks_t)  # (12, 1, 256, 314)
            # Z-score per chunk
            for c in range(mel.shape[0]):
                mel[c] = (mel[c] - mel[c].mean()) / (mel[c].std() + 1e-6)
            all_mels.append(mel.numpy().astype(np.float32))
            fname = f.stem
            for c in range(N_WINDOWS):
                end_sec = (c + 1) * 5
                file_ids.append(f"{fname}_{end_sec}")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_files)} ({(time.time()-t0)/60:.1f}min)")

    all_mels = np.concatenate(all_mels, axis=0)
    print(f"All mels shape: {all_mels.shape}")  # (N_files * 12, 1, 256, 314)
    print(f"Mel pre-compute time: {(time.time()-t0)/60:.1f}min")
else:
    all_mels = None
    file_ids = []
""")

# =============================================================
code("""# Cell 8: 5-fold ONNX inference
if HAS_TEST_FILES:
    n_chunks = len(all_mels)
    all_fold_preds = np.zeros((len(onnx_paths), n_chunks, NUM_CLASSES), dtype=np.float32)

    for fi, onnx_path in enumerate(onnx_paths):
        t_fold = time.time()
        print(f"\\n=== Fold {fi}: {onnx_path.name} ===")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        for s in range(0, n_chunks, BATCH_SIZE):
            batch = all_mels[s:s+BATCH_SIZE]  # (B, 1, 256, 314)
            outputs = sess.run(["clip_logit", "framewise"], {"mel": batch})
            clip_logit = outputs[0]
            framewise = outputs[1]
            frame_max = framewise.max(axis=1)
            p_clip = 1.0 / (1.0 + np.exp(-clip_logit))
            p_fmax = 1.0 / (1.0 + np.exp(-frame_max))
            p_blend = 0.5 * p_clip + 0.5 * p_fmax
            all_fold_preds[fi, s:s+len(p_blend)] = p_blend

        del sess; gc.collect()
        print(f"  Fold {fi} done in {(time.time()-t_fold)/60:.1f}min")

    print(f"\\nAll {len(onnx_paths)} folds done in {(time.time()-START)/60:.1f}min")
else:
    all_fold_preds = None
""")

# =============================================================
code("""# Cell 9: Rank-avg ensemble + submission
if HAS_TEST_FILES:
    print("=== 5-fold rank-avg ensemble ===")
    fold_ranks = np.zeros_like(all_fold_preds)
    for fi in range(len(onnx_paths)):
        flat = all_fold_preds[fi]
        rank = pd.DataFrame(flat).rank(axis=0, pct=True).to_numpy().astype(np.float32)
        fold_ranks[fi] = rank

    ensemble_pred = fold_ranks.mean(axis=0)
    print(f"Ensemble pred shape: {ensemble_pred.shape}")

    sub_df = pd.DataFrame(ensemble_pred, columns=PRIMARY_LABELS)
    sub_df.insert(0, "row_id", file_ids)

    expected_rows = len(sample_sub)
    if len(sub_df) != expected_rows:
        print(f"[WARN] Row mismatch {len(sub_df)} vs {expected_rows}, aligning...")
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
