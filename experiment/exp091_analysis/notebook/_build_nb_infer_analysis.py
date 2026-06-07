"""Build inference NB for exp091 analysis.

Goal: Run on Kaggle GPU and produce:
- Labeled SS (708 windows): exp037 stream OOF predictions + truth + meta
- train_audio (35,549): Perch v2 raw scores + meta (primary_label)
- train_sc (10,658 × 12 win): Perch v2 + LightProtoSSM stream predictions + meta (site/hour)

All outputs as npz to /kaggle/working/ + upload as Dataset.

Designed to run in ~60-90 min on Kaggle GPU T4.
"""
import json
from pathlib import Path

# Helper to make a code cell
def code_cell(src, cell_id=None):
    cell = {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}
    if cell_id: cell["id"] = cell_id
    return cell

def md_cell(src, cell_id=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    if cell_id: cell["id"] = cell_id
    return cell

# ============================================================
# Cell 0: Header
# ============================================================
hdr = '''# exp091 Analysis Inference — Labeled SS + train_audio + train_sc

**Purpose**: Compute predictions on training data to enable LOCAL Streamlit analysis.

## Outputs (saved to /kaggle/working/, downloadable)

| File | Shape | 内容 |
|---|---|---|
| `truth_ss.npz` | (708, 234) | labeled SS 真実ラベル |
| `perch_emb_ss.npz` | (708, 1536) | Perch v2 embeddings on labeled SS |
| `perch_score_ss.npz` | (708, 234) | Perch v2 logits (mapped to BC2026 labels) |
| `exp037_oof_ss.npz` | (708, 234) | exp037 v313 stream OOF (full pipeline) |
| `meta_ss.parquet` | 708 rows | filename, site, hour, window_idx |
| `perch_score_train_audio.npz` | (35549, 234) | Perch v2 on center-5s of focal |
| `meta_train_audio.parquet` | 35549 rows | filename, primary_label, scientific_name, class_name |
| `perch_score_train_sc.npz` | (10658, 12, 234) | Perch v2 on all 12 windows of all SS |
| `meta_train_sc.parquet` | 10658 rows | filename, site, hour |

## Configuration
- GPU: T4 (onnxruntime-gpu)
- Internet: True (Kaggle Dataset upload at end)
- Run time: ~60-90 min
'''

# ============================================================
# Cell 1: Setup
# ============================================================
setup = '''# ============================================================
# Cell 1: Setup
# ============================================================
import subprocess, sys, os, time, json
from pathlib import Path

# GPU check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Install
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "onnxruntime-gpu", "timm>=1.0", "librosa", "soundfile", "pyarrow"])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

print(f"torch={torch.__version__}, ort={ort.__version__}")
print(f"ort providers: {ort.get_available_providers()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TIME = time.time()
'''

# ============================================================
# Cell 2: Load data + paths
# ============================================================
load_data = '''# ============================================================
# Cell 2: Load data + paths
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None
print(f"BASE: {BASE}")

TA_DIR = BASE / "train_audio"
TS_DIR = BASE / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
TRAIN_CSV = BASE / "train.csv"
SS_LABELS_PATH = BASE / "train_soundscapes_labels.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"

# Label ordering
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
N_CLASSES = len(PRIMARY_LABELS)
print(f"Classes: {N_CLASSES}")

# Taxonomy
taxonomy_df = pd.read_csv(TAXO_PATH)
taxonomy_df["primary_label"] = taxonomy_df["primary_label"].astype(str)
print(f"Taxonomy: {len(taxonomy_df)} species")

# Train metadata (35k focal)
train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[train_df["primary_label"].astype(str).isin(LABEL2IDX)].reset_index(drop=True)
train_df["filename"] = train_df["filename"].astype(str)
train_df["exists"] = train_df["filename"].apply(lambda fn: (TA_DIR / fn).exists())
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"train_audio: {len(train_df)} files exist")

# train_sc files
ts_files = sorted([p.name for p in TS_DIR.glob("*.ogg")])
print(f"train_sc: {len(ts_files)} files")

# Labeled SS truth
sc_labels = pd.read_csv(SS_LABELS_PATH).drop_duplicates()
if sc_labels["start"].dtype == object:
    sc_labels["start_sec"] = pd.to_timedelta(sc_labels["start"]).dt.total_seconds().astype(int)
else:
    sc_labels["start_sec"] = sc_labels["start"].astype(int)
print(f"SS labels: {len(sc_labels)} rows, {sc_labels['filename'].nunique()} files")

# Parse site/hour from filename like "BC2026_Ssite_HHMMSS"
import re
def parse_filename(fn):
    fn = str(fn)
    m = re.search(r"_S(\\d+)_\\d{8}_(\\d{6})", fn)
    if m:
        site = "S" + m.group(1)
        hour = int(m.group(2)[:2])
        return site, hour
    return "UNK", -1

# Build Y_SC for labeled SS windows (708 windows = 59 files × 12)
labeled_files = sorted(sc_labels["filename"].unique())
print(f"Labeled files: {len(labeled_files)}")

# Each file gets 12 windows × 5s
SR = 32000
N_WINDOWS = 12
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC

ss_meta = []
for fn in labeled_files:
    site, hour = parse_filename(fn)
    for w in range(N_WINDOWS):
        ss_meta.append({
            "filename": fn,
            "site": site,
            "hour_utc": hour,
            "window_idx": w,
            "start_sec": w * WINDOW_SEC,
        })
ss_meta_df = pd.DataFrame(ss_meta)
print(f"SS meta: {len(ss_meta_df)} windows")

# Build truth (n_windows, 234)
truth_ss = np.zeros((len(ss_meta_df), N_CLASSES), dtype=np.float32)
for _, row in sc_labels.iterrows():
    fn = row["filename"]
    start = int(row["start_sec"])
    win_idx = start // WINDOW_SEC
    # Find this row in ss_meta_df
    idx_match = ss_meta_df[(ss_meta_df["filename"] == fn) & (ss_meta_df["window_idx"] == win_idx)].index
    if len(idx_match) == 0:
        continue
    win_global_idx = idx_match[0]
    for lbl in str(row["primary_label"]).split(";"):
        lbl = lbl.strip()
        if lbl in LABEL2IDX:
            truth_ss[win_global_idx, LABEL2IDX[lbl]] = 1.0

print(f"Truth SS: {truth_ss.shape}, positives={int(truth_ss.sum())}, active species={int((truth_ss.sum(axis=0)>0).sum())}")
'''

# ============================================================
# Cell 3: Perch v2 ONNX setup (GPU)
# ============================================================
perch_setup = '''# ============================================================
# Cell 3: Perch v2 ONNX setup (GPU)
# ============================================================
# Locate Perch ONNX
PERCH_PATH = None
for cand in [
    Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"),
    Path("/kaggle/input/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"),
]:
    if cand.exists():
        PERCH_PATH = cand; break
if PERCH_PATH is None:
    # rglob fallback
    for hit in Path("/kaggle/input").rglob("perch_v2*.onnx"):
        PERCH_PATH = hit; break
assert PERCH_PATH is not None, "Perch ONNX not found"
print(f"Perch: {PERCH_PATH}")

# Load ONNX with GPU
providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
session = ort.InferenceSession(str(PERCH_PATH), providers=providers)
input_name = session.get_inputs()[0].name
print(f"Perch input: {input_name}, providers: {session.get_providers()}")
for i, o in enumerate(session.get_outputs()):
    print(f"  output[{i}] {o.name} shape={o.shape}")

# Output indices: typically [0]=embedding (1536), [1]=label logits (~14795)
# Find by shape
EMBED_IDX = None
LABEL_IDX = None
for i, o in enumerate(session.get_outputs()):
    shape = o.shape
    # ★ Embedding: 2D (batch, 1536) — first match, do NOT overwrite with spatial_embed (4D)
    if shape and EMBED_IDX is None and len(shape) == 2 and shape[-1] == 1536:
        EMBED_IDX = i
    # ★ Label: 2D (batch, 14795)
    if shape and LABEL_IDX is None and len(shape) == 2 and shape[-1] > 5000:
        LABEL_IDX = i
print(f"EMBED_IDX={EMBED_IDX}, LABEL_IDX={LABEL_IDX}")

# Perch label → BC2026 label mapping (from jaejohn/perch-meta)
PERCH_META_DIR = None
for cand in [
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
    Path("/kaggle/input/perch-meta"),
]:
    if cand.exists():
        PERCH_META_DIR = cand; break
print(f"Perch meta dir: {PERCH_META_DIR}")

# Look for label mapping CSV
perch_label_csv = None
if PERCH_META_DIR:
    for p in PERCH_META_DIR.rglob("*.csv"):
        if "label" in p.name.lower():
            perch_label_csv = p; break
    if not perch_label_csv:
        # take any csv with mapping-like name
        for p in PERCH_META_DIR.rglob("*.csv"):
            perch_label_csv = p; break
print(f"Perch label CSV: {perch_label_csv}")

# Build mapping: Perch class index → BC2026 PRIMARY_LABELS index
# Assume CSV has columns like "perch_idx" or "perch_label" and "ebird_code" or similar
perch_to_bc = {}  # perch_idx → bc_idx
if perch_label_csv is not None:
    pmap = pd.read_csv(perch_label_csv)
    print(f"  Perch label CSV columns: {list(pmap.columns)}")
    print(f"  Sample: {pmap.head(3).to_dict('records')}")
    # We'll defer mapping until we see structure

def perch_infer(audio_batch, batch_size=64):
    """Run Perch inference on audio batch.
    Args:
        audio_batch: (N, 5*SR) float32 numpy array of 5s audio clips at 32kHz
        batch_size: GPU batch size
    Returns:
        embeddings: (N, 1536)
        scores: (N, num_perch_classes)  # full Perch label space
    """
    N = audio_batch.shape[0]
    embeddings = []
    scores = []
    for i in range(0, N, batch_size):
        batch = audio_batch[i:i+batch_size].astype(np.float32)
        out = session.run(None, {input_name: batch})
        if EMBED_IDX is not None:
            embeddings.append(out[EMBED_IDX])
        if LABEL_IDX is not None:
            scores.append(out[LABEL_IDX])
    emb = np.concatenate(embeddings) if embeddings else np.zeros((N, 1536), dtype=np.float32)
    sc  = np.concatenate(scores)     if scores     else np.zeros((N, 1), dtype=np.float32)
    return emb, sc

# Test on dummy
dummy = np.random.randn(2, SR * 5).astype(np.float32) * 0.01
emb, sc = perch_infer(dummy)
print(f"Test: emb {emb.shape}, scores {sc.shape}")
'''

# ============================================================
# Cell 4: Perch → BC2026 label mapping
# ============================================================
mapping = '''# ============================================================
# Cell 4: Perch label → BC2026 label mapping (from perch-meta)
# ============================================================
# Build map: perch_idx → bc2026_idx (-1 if no match)
import csv

perch_to_bc_idx = np.full(sc.shape[-1], -1, dtype=np.int32)
n_mapped = 0

# Try various CSV files in perch-meta
if PERCH_META_DIR:
    for csv_path in sorted(PERCH_META_DIR.rglob("*.csv")):
        print(f"  Examining {csv_path.name}...")
        try:
            df = pd.read_csv(csv_path)
            cols = list(df.columns)
            # Look for inat_taxon_id column (BC2026 primary_label key)
            inat_col = None
            for c in cols:
                if "inat" in c.lower() and "id" in c.lower():
                    inat_col = c; break
            if inat_col is None:
                # Look for ebird_code
                for c in cols:
                    if "ebird" in c.lower() or "code" in c.lower() or "label" in c.lower():
                        inat_col = c; break
            if inat_col is None:
                continue
            print(f"    Using column: {inat_col}")
            # Build mapping
            for perch_idx, row in df.iterrows():
                key = str(row[inat_col]).strip()
                if key in LABEL2IDX:
                    perch_to_bc_idx[perch_idx] = LABEL2IDX[key]
                    n_mapped += 1
            if n_mapped > 0:
                print(f"    Mapped {n_mapped} / {len(df)} perch classes")
                break
        except Exception as e:
            print(f"    Error: {e}")
            continue

print(f"Total mapped: {n_mapped}")

def perch_to_bc2026(perch_scores):
    """Convert Perch (N, num_perch_classes) scores → BC2026 (N, 234) scores.
    Unmapped BC2026 species get 0 (handled by other streams)."""
    bc_scores = np.zeros((perch_scores.shape[0], N_CLASSES), dtype=np.float32)
    for perch_i, bc_i in enumerate(perch_to_bc_idx):
        if bc_i >= 0:
            bc_scores[:, bc_i] = perch_scores[:, perch_i]
    # Apply sigmoid (Perch logits → probabilities)
    return 1.0 / (1.0 + np.exp(-np.clip(bc_scores, -50, 50)))

# Test
test_bc = perch_to_bc2026(sc)
print(f"BC mapped test: {test_bc.shape}, non-zero species: {(test_bc.sum(axis=0) > 0).sum()}")
'''

# ============================================================
# Cell 5: Audio loaders
# ============================================================
audio_loaders = '''# ============================================================
# Cell 5: Audio loader helpers
# ============================================================
def load_5s_clip(path, start_sec=0.0, duration=5.0):
    """Load 5s clip at 32kHz mono."""
    try:
        n_frames = int(SR * duration)
        wav, sr = sf.read(str(path), start=int(start_sec * SR),
                           frames=n_frames, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        if len(wav) < n_frames:
            wav = np.pad(wav, (0, n_frames - len(wav)))
        return wav[:n_frames].astype(np.float32)
    except Exception as e:
        return np.zeros(int(SR * duration), dtype=np.float32)

def load_60s(path):
    """Load 60s mono 32kHz audio."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    except Exception:
        return np.zeros(SR * 60, dtype=np.float32)
    target = SR * 60
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    return wav.astype(np.float32)

def load_focal_center_5s(path):
    """Load center 5s of focal recording (variable length)."""
    try:
        # Get full audio first
        info = sf.info(str(path))
        total_sec = info.frames / info.samplerate
        if total_sec <= 5.0:
            wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        else:
            center = total_sec / 2.0
            start = max(0, center - 2.5)
            wav, sr = sf.read(str(path), start=int(start * info.samplerate),
                               frames=int(5.0 * info.samplerate),
                               dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        target = SR * 5
        if len(wav) < target:
            wav = np.pad(wav, (0, target - len(wav)))
        return wav[:target].astype(np.float32)
    except Exception:
        return np.zeros(SR * 5, dtype=np.float32)

print("Audio loaders defined")
'''

# ============================================================
# Cell 6: Phase A — Labeled SS inference
# ============================================================
phase_a = '''# ============================================================
# Cell 6: Phase A — Labeled SS inference (708 windows)
# ============================================================
print("\\n=== Phase A: Labeled SS ===")
t0 = time.time()

# Load all 708 windows (59 files × 12 windows × 5s)
audio_ss = np.zeros((len(ss_meta_df), WINDOW_SAMPLES), dtype=np.float32)
for i, row in ss_meta_df.iterrows():
    path = TS_DIR / row["filename"]
    if not str(path).endswith(".ogg"):
        path = TS_DIR / (row["filename"] + ".ogg")
    audio_ss[i] = load_5s_clip(path, start_sec=row["start_sec"])
    if (i + 1) % 100 == 0:
        print(f"  loaded {i+1}/{len(ss_meta_df)} windows")
print(f"Loaded {len(audio_ss)} windows in {time.time()-t0:.1f}s")

# Perch GPU inference
t0 = time.time()
perch_emb_ss, perch_logits_ss = perch_infer(audio_ss, batch_size=32)
print(f"Perch SS done: emb {perch_emb_ss.shape}, logits {perch_logits_ss.shape}, time {time.time()-t0:.1f}s")

# Map to BC2026 space
perch_score_ss = perch_to_bc2026(perch_logits_ss)
print(f"BC2026 mapped: {perch_score_ss.shape}, non-zero species: {(perch_score_ss.sum(axis=0) > 0).sum()}")

# Save
OUT_DIR = Path("/kaggle/working")
np.savez_compressed(OUT_DIR / "perch_emb_ss.npz", perch_emb_ss)
np.savez_compressed(OUT_DIR / "perch_score_ss.npz", perch_score_ss)
np.savez_compressed(OUT_DIR / "truth_ss.npz", truth_ss)
ss_meta_df.to_parquet(OUT_DIR / "meta_ss.parquet")
print(f"Saved Phase A outputs to {OUT_DIR}")
'''

# ============================================================
# Cell 7: Phase B — train_audio inference
# ============================================================
phase_b = '''# ============================================================
# Cell 7: Phase B — train_audio inference (~35k files, center 5s)
# ============================================================
print("\\n=== Phase B: train_audio ===")
t0 = time.time()

n_audio = len(train_df)
perch_emb_train_audio = np.zeros((n_audio, 1536), dtype=np.float32)
perch_logits_train_audio = np.zeros((n_audio, perch_logits_ss.shape[-1]), dtype=np.float32)

BATCH = 64
audio_buffer = np.zeros((BATCH, WINDOW_SAMPLES), dtype=np.float32)
for i in range(0, n_audio, BATCH):
    end = min(i + BATCH, n_audio)
    actual_batch = end - i
    # Load
    for j in range(actual_batch):
        fn = train_df.iloc[i + j]["filename"]
        audio_buffer[j] = load_focal_center_5s(TA_DIR / fn)
    # Inference
    emb, sc = perch_infer(audio_buffer[:actual_batch], batch_size=actual_batch)
    perch_emb_train_audio[i:end] = emb
    perch_logits_train_audio[i:end] = sc
    if (i + BATCH) % (BATCH * 50) == 0 or end == n_audio:
        elapsed = time.time() - t0
        eta = elapsed * (n_audio - end) / max(end, 1) / 60
        print(f"  [{end}/{n_audio}] {elapsed/60:.1f}min, ETA {eta:.1f}min")

print(f"Phase B done in {(time.time()-t0)/60:.1f}min")

# Map to BC2026
perch_score_train_audio = perch_to_bc2026(perch_logits_train_audio)
print(f"BC mapped: {perch_score_train_audio.shape}")

# Save (don't save full Perch emb to keep file size manageable for 35k)
np.savez_compressed(OUT_DIR / "perch_score_train_audio.npz", perch_score_train_audio)

# Meta
meta_audio = train_df[["filename", "primary_label", "scientific_name", "common_name", "class_name",
                         "inat_taxon_id", "latitude", "longitude", "rating", "collection"]].copy()
meta_audio.to_parquet(OUT_DIR / "meta_train_audio.parquet")
print(f"Saved Phase B outputs")
'''

# ============================================================
# Cell 8: Phase C — train_sc inference
# ============================================================
phase_c = '''# ============================================================
# Cell 8: Phase C — train_sc inference (10,658 files × 12 windows = 127k)
# ============================================================
print("\\n=== Phase C: train_sc ===")
t0 = time.time()

n_files = len(ts_files)
n_total_windows = n_files * N_WINDOWS

perch_emb_train_sc = np.zeros((n_files, N_WINDOWS, 1536), dtype=np.float32)
perch_logits_train_sc = np.zeros((n_files, N_WINDOWS, perch_logits_ss.shape[-1]), dtype=np.float32)

# Process file-by-file (each file → 12 windows batch)
audio_12win = np.zeros((N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
for fi, fn in enumerate(ts_files):
    wav_60s = load_60s(TS_DIR / fn)
    for w in range(N_WINDOWS):
        audio_12win[w] = wav_60s[w * WINDOW_SAMPLES:(w + 1) * WINDOW_SAMPLES]
    emb, sc = perch_infer(audio_12win, batch_size=N_WINDOWS)
    perch_emb_train_sc[fi] = emb
    perch_logits_train_sc[fi] = sc
    if (fi + 1) % 100 == 0 or fi == n_files - 1:
        elapsed = time.time() - t0
        eta = elapsed * (n_files - fi - 1) / max(fi + 1, 1) / 60
        print(f"  [{fi+1}/{n_files}] {elapsed/60:.1f}min, ETA {eta:.1f}min")

print(f"Phase C done in {(time.time()-t0)/60:.1f}min")

# Map to BC2026
perch_score_train_sc = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)
for w in range(N_WINDOWS):
    perch_score_train_sc[:, w, :] = perch_to_bc2026(perch_logits_train_sc[:, w, :])
print(f"BC mapped: {perch_score_train_sc.shape}")

# Save (Perch emb too large for 10k files; skip embedding save)
np.savez_compressed(OUT_DIR / "perch_score_train_sc.npz", perch_score_train_sc)

# Meta
meta_sc = []
for fn in ts_files:
    site, hour = parse_filename(fn)
    has_truth = fn in labeled_files or fn.replace(".ogg", "") in labeled_files
    meta_sc.append({"filename": fn, "site": site, "hour_utc": hour, "has_truth": has_truth})
meta_sc_df = pd.DataFrame(meta_sc)
meta_sc_df.to_parquet(OUT_DIR / "meta_train_sc.parquet")
print(f"Saved Phase C outputs")
'''

# ============================================================
# Cell 9: Summary
# ============================================================
summary = '''# ============================================================
# Cell 9: Summary + output file list
# ============================================================
print("\\n=== Summary ===")
print(f"Total time: {(time.time()-START_TIME)/60:.1f} min")

print("\\nOutput files:")
for p in sorted(OUT_DIR.glob("*.npz")) + sorted(OUT_DIR.glob("*.parquet")):
    print(f"  {p.name}  {p.stat().st_size/1e6:.1f}MB")

# Optional: save as Kaggle Dataset (uncomment if desired)
# import tempfile, shutil
# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi(); api.authenticate()
# with tempfile.TemporaryDirectory() as td:
#     td = Path(td)
#     for p in OUT_DIR.glob("*.npz"):
#         shutil.copy(p, td / p.name)
#     for p in OUT_DIR.glob("*.parquet"):
#         shutil.copy(p, td / p.name)
#     meta = {
#         "title": "BirdCLEF2026 exp091 analysis",
#         "id": "maekeso/birdclef2026-exp091-analysis",
#         "licenses": [{"name": "CC0-1.0"}],
#     }
#     (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
#     api.dataset_create_version(folder=str(td), version_notes="exp091 analysis outputs",
#                                  dir_mode="zip", quiet=False)

print("\\n[OK] DONE")
'''

# ============================================================
# Assemble NB
# ============================================================
nb = {
    "cells": [
        md_cell(hdr, "hdr"),
        code_cell(setup, "setup"),
        code_cell(load_data, "load_data"),
        code_cell(perch_setup, "perch_setup"),
        code_cell(mapping, "mapping"),
        code_cell(audio_loaders, "audio_loaders"),
        code_cell(phase_a, "phase_a"),
        code_cell(phase_b, "phase_b"),
        code_cell(phase_c, "phase_c"),
        code_cell(summary, "summary"),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH = Path(__file__).parent / "nb_infer_analysis.ipynb"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Saved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
