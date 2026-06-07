"""Build exp080 inference NB (M_single standalone sub).

Loads trained b0 ONNX from exp080 training output.
For each test_soundscape file (60s), runs 12 × 5s chunks through ONNX.
Outputs submission.csv (row_id = filename_endsec).
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_infer.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md_cell("""# exp080 inference: M_single standalone sub

**Input**: trained b0 ONNX (`m_single_best.onnx`) from exp080 training NB output

**Process**:
  - Iterate test_soundscapes (each 60s)
  - Extract 12 × 5s chunks
  - ONNX inference (CPU)
  - Sigmoid → probability
  - Write submission CSV

**Output**: submission.csv (row_id × 234 sp)

**Note**: CPU only, 90 min limit. ONNX inference ~1-2x faster than PyTorch.
"""))

cells.append(code_cell("""!pip install -q onnxruntime
import sys, os, time, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import tqdm.auto as tqdm

print(f"Python: {sys.version[:50]}")
print(f"onnxruntime: {ort.__version__}, providers: {ort.get_available_providers()}")
START = time.time()
"""))

cells.append(code_cell("""# CFG
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12
N_CLASSES = 234
WINDOW_SAMPLES = SR * WINDOW_SEC

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
TEST_SC_DIR = Path(DATA_PATH) / "test_soundscapes"
SAMPLE_SUB = Path(DATA_PATH) / "sample_submission.csv"

# M_single ONNX from kernel_source
M_ONNX_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp080-train-b0-combined",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp080-train-b0-combined",
])
assert M_ONNX_DIR is not None
m_onnx_path = M_ONNX_DIR / "m_single_best.onnx"
assert m_onnx_path.exists(), f"Missing {m_onnx_path}"
print(f"M ONNX: {m_onnx_path} ({m_onnx_path.stat().st_size/1e6:.1f} MB)")

OUT_DIR = Path("/kaggle/working")
"""))

cells.append(code_cell("""# Load model
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(m_onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print(f"Input: {input_name}, shape: {sess.get_inputs()[0].shape}")
print(f"Output: {sess.get_outputs()[0].name}, shape: {sess.get_outputs()[0].shape}")

def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

# Load sample_submission for column ordering
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES
"""))

cells.append(code_cell("""# Inference loop
test_files = sorted(TEST_SC_DIR.glob("*.ogg"))
print(f"Test files: {len(test_files)}")
if len(test_files) == 0:
    print("WARN: no test files (Kaggle hidden test mode)")

rows = []
t0 = time.time()
for fi, fp in enumerate(tqdm.tqdm(test_files, desc="Infer")):
    try:
        y, _ = librosa.load(str(fp), sr=SR, mono=True)
    except Exception as e:
        print(f"  load fail {fp.name}: {e}")
        y = np.zeros(SR * 60, dtype=np.float32)

    # Pad/trim to 60s
    target = SR * 60
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
    # Normalize per chunk (matches training)
    for ci in range(N_WINDOWS):
        m = np.abs(chunks[ci]).max()
        if m > 0:
            chunks[ci] = chunks[ci] / m

    # ONNX inference (batch of 12 chunks)
    clip_logits = sess.run(None, {input_name: chunks})[0]  # (12, 234)
    probs = sigmoid_np(clip_logits)

    # Build rows
    file_stem = fp.stem
    for ci in range(N_WINDOWS):
        end_sec = (ci + 1) * 5
        row_id = f"{file_stem}_{end_sec}"
        rows.append([row_id] + probs[ci].tolist())

    if (fi + 1) % 50 == 0 or fi == len(test_files) - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / max(elapsed, 0.001)
        eta = (len(test_files) - fi - 1) / max(rate, 0.001) / 60
        print(f"  [{fi+1}/{len(test_files)}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"\\nInference DONE: {(time.time()-t0)/60:.1f} min, {len(rows)} rows")
"""))

cells.append(code_cell("""# Build submission
sub_df = pd.DataFrame(rows, columns=["row_id"] + PRIMARY_LABELS)
print(f"sub_df: {sub_df.shape}")

# Sanity
assert sub_df["row_id"].nunique() == len(sub_df), "Duplicate row_id"
print(f"  mean prob: {sub_df[PRIMARY_LABELS].mean().mean():.5f}")
print(f"  max prob:  {sub_df[PRIMARY_LABELS].max().max():.4f}")
print(f"  NaN: {sub_df[PRIMARY_LABELS].isna().sum().sum()}")

# Ensure row order matches sample_sub if test files exist
if len(test_files) > 0 and len(sample_sub) > 0:
    sub_df = sub_df.set_index("row_id").reindex(sample_sub["row_id"]).reset_index()
    sub_df[PRIMARY_LABELS] = sub_df[PRIMARY_LABELS].fillna(0.0)

sub_df.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"\\nSaved: {OUT_DIR / 'submission.csv'} ({(OUT_DIR / 'submission.csv').stat().st_size/1e6:.1f} MB)")
print(f"Total time: {(time.time()-START)/60:.1f} min")
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
