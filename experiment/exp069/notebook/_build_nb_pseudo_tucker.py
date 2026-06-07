"""Build nb_pseudo_tucker.ipynb - Streaming Tucker SED 5-fold pseudo gen on BC2026 train_soundscapes.

Streaming: each file loaded → mel → 5-fold Tucker SED → save → discard
No audio_cache accumulation → no OOM on full 10,658 files
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_pseudo_tucker.ipynb")

cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ─────────── Cell 0: Title ───────────
cells.append(md_cell("""# exp069b-tucker: Tucker SED 5-fold pseudo gen (streaming, CPU)

**Pattern B-CPU parallel stream NB**: Tucker SED 5-fold ONNX のみ実行、streaming で full 10,658 files カバー。

**Output**: `/kaggle/working/tucker_raw_234.npz` (10658, 12, 234) float16

**並行 NB**:
  - nb_pseudo_nb4 (NB4 v11)
  - nb_pseudo_exp029 (exp029 R3)

**後段**: exp069c で 3 NPZ blend (rank-avg + sonotype mirror)
"""))


# ─────────── Cell 1: Setup ───────────
cells.append(code_cell("""# Setup
import sys, os, time, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import tqdm.auto as tqdm
from scipy.ndimage import gaussian_filter1d, convolve1d

print(f"Python: {sys.version}")
print(f"onnxruntime: {ort.__version__}")
print(f"librosa: {librosa.__version__}")
print(f"Providers: {ort.get_available_providers()}")

START = time.time()
"""))


# ─────────── Cell 2: CFG ───────────
cells.append(code_cell("""# CFG
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12
N_CLASSES = 234
WINDOW_SAMPLES = SR * WINDOW_SEC

# Tucker SED mel params
N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80

# Smoothing across 12 windows
GAUSS_SIGMA = 0.65

# Paths
_data_path_candidates = [
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
]
DATA_PATH = None
for _p in _data_path_candidates:
    if Path(_p).exists():
        DATA_PATH = _p; break
assert DATA_PATH is not None, f"BC2026 not found in {_data_path_candidates}"
TRAIN_SC_DIR = Path(DATA_PATH) / "train_soundscapes"

# Tucker SED ONNX (5-fold)
SED_BASE = None
for _p in ["/kaggle/input/bc2026-distilled-sed-public", "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public"]:
    if Path(_p).exists():
        SED_BASE = Path(_p); break
assert SED_BASE is not None, "Tucker SED not found"

# Output
OUT_DIR = Path("/kaggle/working")

# Discover test files
test_files = sorted(TRAIN_SC_DIR.glob("*.ogg"))
print(f"TRAIN_SC: {TRAIN_SC_DIR}")
print(f"SED_BASE: {SED_BASE}")
print(f"Test files: {len(test_files)}")
assert len(test_files) > 0
"""))


# ─────────── Cell 3: Load Tucker SED 5-fold ───────────
cells.append(code_cell("""# Load Tucker SED 5-fold ONNX sessions
import re

def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


# Find sed_fold*.onnx files
sed_fold_paths = sorted(SED_BASE.rglob("sed_fold*.onnx"),
                        key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1)))
assert len(sed_fold_paths) > 0, f"No sed_fold*.onnx under {SED_BASE}"
print(f"Tucker SED folds: {[p.name for p in sed_fold_paths]}")

sed_sessions = [make_sed_session(p) for p in sed_fold_paths]
print(f"Loaded {len(sed_sessions)} SED sessions ({time.time()-START:.0f}s)")
"""))


# ─────────── Cell 4: Mel + helper functions ───────────
cells.append(code_cell("""# Audio loading + mel + helpers
def load_one_60s(fp, sr=SR):
    y, _ = librosa.load(str(fp), sr=sr, mono=True)
    target_len = sr * 60
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)
"""))


# ─────────── Cell 5: Streaming inference loop ───────────
cells.append(code_cell("""# Streaming inference: each file load → mel → 5-fold Tucker → save → discard
N_FILES = len(test_files)

# Pre-allocate output (~125MB float16)
probs_sed = np.zeros((N_FILES, N_WINDOWS, N_CLASSES), dtype=np.float16)
file_ids = []

t0 = time.time()
for fi, fp in enumerate(tqdm.tqdm(test_files, desc="Tucker SED stream")):
    # 1. Load 60s audio
    y = load_one_60s(fp)
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

    # 2. Compute mel
    mel = audio_to_mel(chunks)   # (12, 1, n_mels, n_frames)

    # 3. 5-fold ensemble (averaged)
    p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid_np(clip_logits) + 0.5 * sigmoid_np(frame_max)
    p_mean = p_sum / len(sed_sessions)

    # 4. Gaussian smooth across 12 windows
    p_smooth = gaussian_filter1d(p_mean, sigma=GAUSS_SIGMA, axis=0, mode="nearest").astype(np.float32)

    # 5. Save to accumulator (small, ~120MB total for full)
    probs_sed[fi] = p_smooth.astype(np.float16)
    file_ids.append(fp.stem)

    # 6. Discard audio (no accumulation)
    del y, chunks, mel, p_sum, p_mean, p_smooth

    if (fi + 1) % 200 == 0 or fi == N_FILES - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / elapsed
        eta = (N_FILES - fi - 1) / rate / 60
        print(f"  [{fi+1}/{N_FILES}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"Tucker SED inference DONE in {(time.time()-t0)/60:.1f} min")
"""))


# ─────────── Cell 6: Load labels for column names ───────────
cells.append(code_cell("""# Load sample_submission for PRIMARY_LABELS column order
sample_sub = pd.read_csv(Path(DATA_PATH) / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES, f"Expected {N_CLASSES}, got {len(PRIMARY_LABELS)}"
print(f"PRIMARY_LABELS: {len(PRIMARY_LABELS)} species")
"""))


# ─────────── Cell 7: Save NPZ ───────────
cells.append(code_cell("""# Save outputs to /kaggle/working
np.savez_compressed(
    OUT_DIR / "tucker_raw_234.npz",
    probs=probs_sed,
    file_ids=np.array(file_ids),
)
print(f"Saved tucker_raw_234.npz: {(OUT_DIR / 'tucker_raw_234.npz').stat().st_size/1e6:.1f} MB")

with open(OUT_DIR / "primary_labels.json", "w") as f:
    json.dump(list(PRIMARY_LABELS), f, indent=2)
with open(OUT_DIR / "file_index.json", "w") as f:
    json.dump({fid: i for i, fid in enumerate(file_ids)}, f)
with open(OUT_DIR / "stream_meta.json", "w") as f:
    json.dump({"stream_id": "tucker", "n_files": len(file_ids), "shape": list(probs_sed.shape)}, f)

print(f"OK exp069b-tucker DONE: total {(time.time()-START)/60:.1f} min")
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
