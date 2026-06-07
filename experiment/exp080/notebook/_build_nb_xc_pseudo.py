"""Build exp080b: Tucker SED 5-fold pseudo gen on XC audio (Part 1+2+3).

Inputs (dataset_sources):
  - tuckerarrants/bc2026-distilled-sed-public (5-fold ONNX)
  - maekeso/birdclef2026-xc-api-aves-audio (Part 1)
  - maekeso/birdclef2026-xc-api-aves-audio-part2 (Part 2)
  - maekeso/birdclef2026-xc-api-dl-part3 (non-Aves)

Process:
  1. Load Tucker SED 5-fold ONNX
  2. Discover all XC .mp3 files (rglob across 3 mount points)
  3. For each file:
     - librosa.load (mono, 32kHz)
     - Trim/pad to 60s (12 × 5s windows)
     - Tucker SED inference: 5-fold ensemble + Gaussian smoothing
     - Save pseudo (12, 234)
  4. Per-file metadata: scientific_name, duration, n_actual_chunks
  5. Save NPZ + metadata JSON

Output → /kaggle/working/:
  - xc_pseudo_tucker.npz (probs (N, 12, 234) float16, file_ids, durations, n_actual_chunks)
  - xc_meta.json (per-file metadata + species mapping to BC2026 primary_label)
  - primary_labels.json
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_xc_pseudo.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# Cell 0: Title
cells.append(md_cell("""# exp080b: Tucker SED pseudo gen on XC audio

**Inputs**:
  - Tucker SED 5-fold ONNX (`tuckerarrants/bc2026-distilled-sed-public`)
  - XC Part 1: 76 Aves species
  - XC Part 2: 83 Aves species
  - XC Part 3: non-Aves (small, mostly Amphibia)

**Process**: Tucker 5-fold inference on each XC audio file, save per-chunk (12 × 5s) pseudo

**Output**: `xc_pseudo_tucker.npz` + metadata

**Notes**:
- XC files vary 5-120s. Pad short files to 60s with zeros (predictions on padded regions ~0).
- `n_actual_chunks` saved per file for downstream sampling
- BC2026 primary_label mapping via scientific_name (with synonym fallback)
"""))

# Cell 1: Setup
cells.append(code_cell("""!pip install onnxruntime-gpu --quiet 2>&1 | tail -3

# Setup
import sys, os, time, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import tqdm.auto as tqdm
from scipy.ndimage import gaussian_filter1d

print(f"Python: {sys.version[:50]}")
print(f"onnxruntime: {ort.__version__}, providers: {ort.get_available_providers()}")
print(f"librosa: {librosa.__version__}")
START = time.time()

# CFG (Tucker SED spec, must match training)
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12
N_CLASSES = 234
WINDOW_SAMPLES = SR * WINDOW_SEC
TARGET_LEN = SR * 60  # 60s padded

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80
GAUSS_SIGMA = 0.65

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
TAXONOMY_CSV = DATA_PATH / "taxonomy.csv"
SAMPLE_SUB = DATA_PATH / "sample_submission.csv"

SED_BASE = find_dir([
    "/kaggle/input/bc2026-distilled-sed-public",
    "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public",
])
assert SED_BASE is not None, "Tucker SED not mounted"

XC_PART1 = find_dir([
    "/kaggle/input/birdclef2026-exp047-xc-api-dl-part1",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp047-xc-api-dl-part1",
])
XC_PART2 = find_dir([
    "/kaggle/input/birdclef2026-exp047-xc-api-dl-part2",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp047-xc-api-dl-part2",
])
XC_PART3 = find_dir([
    "/kaggle/input/birdclef2026-xc-api-dl-part3",
    "/kaggle/input/datasets/maekeso/birdclef2026-xc-api-dl-part3",
])

print(f"\\nSED_BASE: {SED_BASE}")
print(f"XC Part 1: {XC_PART1}")
print(f"XC Part 2: {XC_PART2}")
print(f"XC Part 3: {XC_PART3}")
assert XC_PART1 or XC_PART2 or XC_PART3, "No XC dataset mounted"

OUT_DIR = Path("/kaggle/working")
"""))

# Cell 2: Load Tucker SED 5-fold
cells.append(code_cell("""# Load Tucker SED 5-fold ONNX
import re

def make_sed_session(path, use_gpu=True):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)

USE_GPU = "CUDAExecutionProvider" in ort.get_available_providers()
print(f"USE_GPU: {USE_GPU}")

sed_fold_paths = sorted(SED_BASE.rglob("sed_fold*.onnx"),
                        key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1)))
assert len(sed_fold_paths) == 5, f"Expected 5 folds, got {len(sed_fold_paths)}"
print(f"Tucker SED folds: {[p.name for p in sed_fold_paths]}")

sed_sessions = [make_sed_session(p, USE_GPU) for p in sed_fold_paths]
print(f"Loaded {len(sed_sessions)} sessions ({time.time()-START:.0f}s)")

# Verify input/output
sess0 = sed_sessions[0]
print(f"  input name: {sess0.get_inputs()[0].name}, shape: {sess0.get_inputs()[0].shape}")
for o in sess0.get_outputs():
    print(f"  output: {o.name}, shape: {o.shape}")
"""))

# Cell 3: Discover XC files + species mapping
cells.append(code_cell("""# Discover XC files across 3 mount points
all_xc_files = []
for xc_dir in [XC_PART1, XC_PART2, XC_PART3]:
    if xc_dir is None:
        continue
    mp3_files = list(xc_dir.rglob("*.mp3"))
    print(f"  {xc_dir.name}: {len(mp3_files)} mp3 files")
    all_xc_files.extend(mp3_files)

print(f"\\nTotal XC files: {len(all_xc_files)}")
assert len(all_xc_files) > 0, "No XC files found"

# Build species mapping: dir name (e.g., Hylophilus_pectoralis) → primary_label (e.g., ashgre1)
# XC dir names are scientific_name with underscores
# BC2026 taxonomy has primary_label + scientific_name columns
taxo = pd.read_csv(TAXONOMY_CSV)
print(f"\\nBC2026 taxonomy: {len(taxo)} species")

sci_to_label = {}
for _, r in taxo.iterrows():
    sci = str(r["scientific_name"])
    # Normalize: space → underscore
    sci_normalized = sci.replace(" ", "_").replace("/", "_").replace("'", "")
    sci_to_label[sci_normalized] = str(r["primary_label"])
print(f"  sci_name → primary_label mapping: {len(sci_to_label)}")

# Per-file mapping
file_to_label = {}
unmapped_dirs = set()
for f in all_xc_files:
    species_dir = f.parent.name  # e.g., 'Hylophilus_pectoralis'
    if species_dir in sci_to_label:
        file_to_label[str(f)] = sci_to_label[species_dir]
    else:
        unmapped_dirs.add(species_dir)

print(f"\\nMapped: {len(file_to_label)} / {len(all_xc_files)} files")
if unmapped_dirs:
    print(f"Unmapped species dirs ({len(unmapped_dirs)}): {sorted(unmapped_dirs)[:20]}")

# Filter to mapped files only
xc_files = [f for f in all_xc_files if str(f) in file_to_label]
print(f"\\nFinal XC files to process: {len(xc_files)}")
"""))

# Cell 4: Audio loading + mel + inference helpers
cells.append(code_cell("""# Helpers
def load_audio_padded(fp, sr=SR, target_len=TARGET_LEN):
    \"\"\"Load MP3, mono, pad/trim to target_len. Returns audio + actual_dur_sec.\"\"\"
    try:
        y, _ = librosa.load(str(fp), sr=sr, mono=True)
        actual_len = len(y)
        if actual_len < target_len:
            y = np.pad(y, (0, target_len - actual_len))
        else:
            y = y[:target_len]
        return y.astype(np.float32), actual_len / sr
    except Exception as e:
        print(f"  WARN load failed {fp}: {e}")
        return None, 0.0


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


def tucker_5fold_inference(mel):
    \"\"\"mel: (N_WIN, 1, N_MELS, T_frames) → (N_WIN, 234) probabilities\"\"\"
    p_sum = np.zeros((mel.shape[0], N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid_np(clip_logits) + 0.5 * sigmoid_np(frame_max)
    p_mean = p_sum / len(sed_sessions)
    # Gaussian smooth across 12 windows (consistent with Tucker spec)
    p_smooth = gaussian_filter1d(p_mean, sigma=GAUSS_SIGMA, axis=0, mode="nearest").astype(np.float32)
    return p_smooth
"""))

# Cell 5: Run inference
cells.append(code_cell("""# Inference loop
N_FILES = len(xc_files)
print(f"Processing {N_FILES} files...")

# Pre-allocate output
probs_out = np.zeros((N_FILES, N_WINDOWS, N_CLASSES), dtype=np.float16)
file_ids = []          # XC file stem (e.g., XC12345)
sci_names = []         # species dir name
primary_labels = []    # mapped BC2026 primary_label
durations_sec = []     # actual audio duration (before pad)
n_actual_chunks_arr = []  # ceil(min(60, dur) / 5), capped at 12
sources = []           # which part dataset
failed_files = []

t0 = time.time()
for fi, fp in enumerate(tqdm.tqdm(xc_files, desc="Tucker on XC")):
    y, actual_dur = load_audio_padded(fp)
    if y is None:
        failed_files.append(str(fp))
        # Save zeros (already initialized)
        file_ids.append(fp.stem)
        sci_names.append(fp.parent.name)
        primary_labels.append(file_to_label[str(fp)])
        durations_sec.append(0.0)
        n_actual_chunks_arr.append(0)
        sources.append(fp.parent.parent.parent.name if fp.parent.parent else "?")
        continue

    # Extract 12 chunks of 5s
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

    # Mel + inference
    mel = audio_to_mel(chunks)
    p_smooth = tucker_5fold_inference(mel)

    probs_out[fi] = p_smooth.astype(np.float16)
    file_ids.append(fp.stem)
    sci_names.append(fp.parent.name)
    primary_labels.append(file_to_label[str(fp)])
    durations_sec.append(actual_dur)
    # Actual chunks based on duration (capped at 12)
    n_act = min(N_WINDOWS, int(np.ceil(actual_dur / WINDOW_SEC)))
    n_actual_chunks_arr.append(max(1, n_act))
    sources.append(fp.parent.parent.parent.name if fp.parent.parent else "?")

    # Cleanup
    del y, chunks, mel, p_smooth

    if (fi + 1) % 200 == 0 or fi == N_FILES - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / elapsed
        eta = (N_FILES - fi - 1) / rate / 60
        print(f"  [{fi+1}/{N_FILES}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"\\nInference DONE: {(time.time()-t0)/60:.1f} min")
print(f"Failed: {len(failed_files)}")
if failed_files:
    print(f"  Sample fails: {failed_files[:5]}")
"""))

# Cell 6: Verify + save
cells.append(code_cell("""# Verify output
print(f"=== Output verification ===")
print(f"  probs shape: {probs_out.shape}, dtype: {probs_out.dtype}")
print(f"  mean = {probs_out.mean():.5f}, max = {probs_out.max():.5f}")
print(f"  NaN: {int(np.isnan(probs_out).sum())}, Inf: {int(np.isinf(probs_out).sum())}")
print(f"  >0.5 frac = {(probs_out > 0.5).sum()/probs_out.size:.5f}")

dur_arr = np.array(durations_sec)
print(f"\\n  Duration stats: mean={dur_arr.mean():.1f}s, median={np.median(dur_arr):.1f}s, max={dur_arr.max():.1f}s")
print(f"  Files with duration > 0: {(dur_arr > 0).sum()} / {N_FILES}")

n_chunks_arr = np.array(n_actual_chunks_arr)
print(f"  n_actual_chunks: mean={n_chunks_arr.mean():.1f}, min={n_chunks_arr.min()}, max={n_chunks_arr.max()}")
total_valid_chunks = int(n_chunks_arr.sum())
print(f"  Total valid chunks (sum n_actual): {total_valid_chunks}")

# Save NPZ
print(f"\\n=== Save outputs ===")
np.savez_compressed(
    OUT_DIR / "xc_pseudo_tucker.npz",
    probs=probs_out,
    file_ids=np.array(file_ids),
    sci_names=np.array(sci_names),
    primary_labels=np.array(primary_labels),
    durations_sec=dur_arr.astype(np.float32),
    n_actual_chunks=n_chunks_arr.astype(np.int8),
    sources=np.array(sources),
)
print(f"  xc_pseudo_tucker.npz: {(OUT_DIR / 'xc_pseudo_tucker.npz').stat().st_size/1e6:.1f} MB")

# Load sample_sub for PRIMARY_LABELS ordering
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == 234
with open(OUT_DIR / "primary_labels.json", "w") as f:
    json.dump(list(PRIMARY_LABELS), f, indent=2)

# Meta
meta = {
    "exp": "exp080b",
    "description": "Tucker SED 5-fold pseudo on XC audio (Part 1+2+3)",
    "n_files": int(N_FILES),
    "n_windows": int(N_WINDOWS),
    "n_classes": int(N_CLASSES),
    "n_failed": len(failed_files),
    "total_valid_chunks": total_valid_chunks,
    "tucker_config": {
        "sr": SR, "n_mels": N_MELS_SED, "n_fft": N_FFT_SED, "hop": HOP_SED,
        "fmin": FMIN_SED, "fmax": FMAX_SED, "top_db": TOP_DB_SED,
        "gauss_sigma": GAUSS_SIGMA,
    },
    "output_stats": {
        "mean": float(probs_out.mean()),
        "max": float(probs_out.max()),
        "frac_gt_0.5": float((probs_out > 0.5).sum() / probs_out.size),
        "frac_gt_0.9": float((probs_out > 0.9).sum() / probs_out.size),
    },
    "total_time_min": (time.time() - START) / 60,
}
with open(OUT_DIR / "xc_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\\n=== exp080b DONE ===")
print(f"Total time: {(time.time() - START)/60:.1f} min")
print(f"\\nMeta:")
for k, v in meta.items():
    print(f"  {k}: {v}")
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
for c in cells:
    n_lines = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} L={n_lines}")
