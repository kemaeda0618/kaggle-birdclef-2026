"""Generate Perch v2 embedding pre-compute NB.

Pre-computes Perch v2 embeddings for every 5s chunk of train_audio + train_soundscapes.
Output:
- emb.npy: (N_chunks, 1536) float16
- meta.csv: filename, source ('focal'|'ss'), chunk_idx, row_idx

Used by exp014/015/016 R2 (and later) train NBs to skip Perch ONNX forward during
training, replacing it with a fast np.memmap lookup. Recipe change: focal random crop
becomes 5s chunk-aligned; mixup applied at feature level for distill target only.

Run: python _gen_nb_perch_precompute.py  ->  writes nb_perch_precompute.ipynb
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

# =============================================================================
# Cell 0: Header
# =============================================================================
cells.append(md_cell(r"""# Perch v2 embedding pre-compute (Kaggle T4)

Pre-computes Perch v2 1536-d embeddings for every 5s chunk of BC2026
train_audio + train_soundscapes. Used by exp014/015/016 R2 (and later) train NBs.

## Output
- `emb.npy` — `(N_chunks, 1536) float16` — Perch embeddings, indexed by row_idx
- `meta.csv` — `filename, source, chunk_idx, row_idx` lookup table

## Required inputs
- `birdclef-2026` competition data
- `tuckerarrants/perch-v2-no-dft-onnx` Perch v2 ONNX

## Output dataset
`maekeso/birdclef2026-perch-emb-cache` (~1.3 GB)

## Runtime
~100-120 min on T4 GPU. Resumable via own-kernel input pattern (re-run continues from
last saved batch).
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "onnxruntime-gpu", "librosa", "soundfile"])

import os, time, json, gc, math, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort

import warnings
warnings.filterwarnings("ignore")

print(f"ONNX Runtime: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

TRAIN_START = time.time()
MAX_RUNTIME_SEC = 11.0 * 3600   # 11h safety margin
print(f"Train start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(TRAIN_START))}")
""", "setup"))

# =============================================================================
# Cell 2: Paths
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"
print(f"Competition: {BASE}")

TA_DIR = BASE / "train_audio"
TS_DIR_SRC = BASE / "train_soundscapes"
print(f"  train_audio: {TA_DIR.exists()}")
print(f"  train_soundscapes (FUSE source): {TS_DIR_SRC.exists()}")

# Perch v2 ONNX
_perch_cands = list(Path("/kaggle/input").rglob("perch_v2*.onnx"))
assert _perch_cands, "Perch v2 ONNX not found under /kaggle/input"
PERCH_PATH = _perch_cands[0]
print(f"Perch ONNX: {PERCH_PATH}")

# Pre-copy train_soundscapes to local SSD (FUSE I/O is slow at ~50-200ms/open)
TS_DIR_LOCAL = Path("/kaggle/working/train_soundscapes_local")
TS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
_n_existing = sum(1 for _ in TS_DIR_LOCAL.glob("*.ogg"))
_n_src = sum(1 for _ in TS_DIR_SRC.glob("*.ogg")) if TS_DIR_SRC.exists() else 0
print(f"Pre-copy plan: local={_n_existing} / source={_n_src}")
if _n_src > 0 and _n_existing < _n_src:
    _t0 = time.time()
    _copied = 0
    for _f in TS_DIR_SRC.glob("*.ogg"):
        _dst = TS_DIR_LOCAL / _f.name
        if not _dst.exists():
            shutil.copy(str(_f), str(_dst))
            _copied += 1
    print(f"  Copied {_copied} new files in {time.time()-_t0:.1f}s")
TS_DIR = TS_DIR_LOCAL
print(f"  TS_DIR (override): {TS_DIR}")

# Resume input — read previous version's outputs to skip already-processed files
RESUME_DIR = None
RESUME_CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp014-perch-precompute"),
    Path("/kaggle/input/birdclef2026-exp014-perch-precompute"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-perch-emb-cache"),
    Path("/kaggle/input/birdclef2026-perch-emb-cache"),
]
for p in RESUME_CANDIDATES:
    if p.exists():
        hits = list(p.rglob("emb.npy"))
        if hits:
            RESUME_DIR = hits[0].parent; break
print(f"Resume dir: {RESUME_DIR if RESUME_DIR else '(none, fresh start)'}")

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
""", "paths"))

# =============================================================================
# Cell 3: Config
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Config — chunk size, Perch input shape, batch size
# ============================================================
SR = 32000
CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC   # 160000
PERCH_EMBED_DIM = 1536

# Perch batch size (Perch v2 ONNX accepts batched inputs)
# Note: v2 has 4 outputs (embedding/spatial_embedding/spectrogram/label[14795]).
# Running all outputs OOMs at batch=32 because label[14795] is huge.
# We request only 'embedding' below, but keep batch conservative.
PERCH_BATCH = 16

# Save partial progress every N files (~ to enable resume in case of session timeout)
SAVE_EVERY_N_FILES = 2000

print(f"SR: {SR}, chunk: {CHUNK_SEC}s ({CHUNK_SAMPLES} samples)")
print(f"Perch embed dim: {PERCH_EMBED_DIM}")
print(f"Perch batch: {PERCH_BATCH}")
print(f"Save every: {SAVE_EVERY_N_FILES} files")
""", "config"))

# =============================================================================
# Cell 4: Build file list and chunk plan
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Enumerate files, compute chunks per file
# ============================================================
# Focal: recurse train_audio/{taxon_id}/*.ogg
print("Enumerating focal files...")
_t0 = time.time()
focal_files = sorted(TA_DIR.rglob("*.ogg"))
focal_rel = [str(p.relative_to(TA_DIR)) for p in focal_files]
print(f"  {len(focal_files)} focal files ({time.time()-_t0:.1f}s)")

# SS: train_soundscapes/*.ogg
print("Enumerating soundscape files...")
_t0 = time.time()
ss_files = sorted(TS_DIR.glob("*.ogg"))
ss_rel = [p.name for p in ss_files]
print(f"  {len(ss_files)} ss files ({time.time()-_t0:.1f}s)")

# Build chunk plan: for each file, count number of complete 5s chunks
# We compute chunk_count = file_duration_samples // CHUNK_SAMPLES
# Use soundfile.info() which doesn't decode (fast)
print("Computing chunk counts (focal)...")
_t0 = time.time()
focal_plan = []  # list of (rel_path, source, n_chunks)
for i, p in enumerate(focal_files):
    try:
        info = sf.info(str(p))
        n_frames = info.frames
        sr_file = info.samplerate
        # convert to 32kHz-equivalent sample count
        n_samples_32k = int(n_frames * SR / sr_file)
        n_chunks = max(1, n_samples_32k // CHUNK_SAMPLES)
    except Exception:
        n_chunks = 1
    focal_plan.append((focal_rel[i], "focal", n_chunks))
    if (i + 1) % 5000 == 0:
        print(f"  focal {i+1}/{len(focal_files)}  ({time.time()-_t0:.1f}s)")
print(f"  focal plan done ({time.time()-_t0:.1f}s)")

print("Computing chunk counts (ss)...")
_t0 = time.time()
ss_plan = []
for i, p in enumerate(ss_files):
    try:
        info = sf.info(str(p))
        n_frames = info.frames
        sr_file = info.samplerate
        n_samples_32k = int(n_frames * SR / sr_file)
        n_chunks = max(1, n_samples_32k // CHUNK_SAMPLES)
    except Exception:
        n_chunks = 12  # default for 60s ss file
    ss_plan.append((ss_rel[i], "ss", n_chunks))
print(f"  ss plan done ({time.time()-_t0:.1f}s)")

# Build full chunk list (filename, source, chunk_idx, row_idx)
chunks = []
row_idx = 0
for rel, src, nc in focal_plan + ss_plan:
    for ci in range(nc):
        chunks.append((rel, src, ci, row_idx))
        row_idx += 1
N_TOTAL = row_idx
print(f"\nTotal chunks: {N_TOTAL}")
print(f"  focal chunks: {sum(nc for _, _, nc in focal_plan)}")
print(f"  ss chunks: {sum(nc for _, _, nc in ss_plan)}")
print(f"  est. emb size: {N_TOTAL * PERCH_EMBED_DIM * 2 / 1e9:.2f} GB (float16)")

meta_df = pd.DataFrame(chunks, columns=["filename", "source", "chunk_idx", "row_idx"])
meta_df.to_csv(OUT_DIR / "meta.csv", index=False)
print(f"Wrote meta.csv ({len(meta_df)} rows)")
""", "plan"))

# =============================================================================
# Cell 5: Perch teacher init + inference loop
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Perch teacher init
# ============================================================
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(str(PERCH_PATH), providers=providers)
print(f"Providers active: {session.get_providers()}")

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input: name={input_name}, shape={input_shape}")
for i, out in enumerate(session.get_outputs()):
    print(f"Output {i}: name={out.name}, shape={out.shape}")

# Find embedding output (1536-d)
embed_idx = None
for i, out in enumerate(session.get_outputs()):
    if out.shape and out.shape[-1] == PERCH_EMBED_DIM:
        embed_idx = i; break
if embed_idx is None:
    embed_idx = 1  # convention: 0=logits, 1=embed
print(f"Embed output idx: {embed_idx}")


# Get embedding output name (avoid computing other outputs to save GPU memory)
embed_out_name = session.get_outputs()[embed_idx].name
print(f"Will run only output: {embed_out_name!r}")


def perch_forward(wav_batch):
    '''wav_batch: np.float32 (B, CHUNK_SAMPLES) -> np.float16 (B, 1536).
    Only requests 'embedding' output to skip protopnet head (14795-dim) and avoid OOM.'''
    results = session.run([embed_out_name], {input_name: wav_batch.astype(np.float32)})
    return results[0].astype(np.float16)
""", "perch_init"))

# =============================================================================
# Cell 6: Resume — load prior emb.npy if available
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Resume — load prior emb buffer + done file index
# ============================================================
# Strategy: emb buffer is fully indexed by row_idx. Track which files are done
# (all chunks computed) and skip them on resume.

# Pre-allocate emb buffer (memory-mapped to /kaggle/working/emb.npy)
EMB_PATH = OUT_DIR / "emb.npy"
DONE_FILES_PATH = OUT_DIR / "done_files.txt"

if RESUME_DIR is not None and (RESUME_DIR / "emb.npy").exists():
    print(f"Resuming from {RESUME_DIR}")
    prior_emb_path = RESUME_DIR / "emb.npy"
    prior_meta_path = RESUME_DIR / "meta.csv"
    prior_done_path = RESUME_DIR / "done_files.txt"

    # Need to verify schema match: if N_TOTAL or meta differs, can't resume
    prior_meta = pd.read_csv(prior_meta_path) if prior_meta_path.exists() else None
    schema_ok = (prior_meta is not None and len(prior_meta) == N_TOTAL and
                 (prior_meta["filename"].values == meta_df["filename"].values).all() and
                 (prior_meta["chunk_idx"].values == meta_df["chunk_idx"].values).all())
    if schema_ok:
        print(f"  schema matches, copying prior emb.npy ({prior_emb_path.stat().st_size/1e9:.2f} GB)")
        shutil.copy(str(prior_emb_path), str(EMB_PATH))
        if prior_done_path.exists():
            shutil.copy(str(prior_done_path), str(DONE_FILES_PATH))
            done_files = set(DONE_FILES_PATH.read_text().splitlines())
        else:
            done_files = set()
        print(f"  resumed {len(done_files)} files done")
    else:
        print(f"  schema MISMATCH, starting fresh")
        done_files = set()
else:
    print("Fresh start: allocating emb.npy")
    done_files = set()

# Allocate (or extend) emb.npy as memmap
if not EMB_PATH.exists():
    emb_mmap = np.lib.format.open_memmap(
        str(EMB_PATH), mode="w+", dtype=np.float16, shape=(N_TOTAL, PERCH_EMBED_DIM))
    emb_mmap[:] = 0
    emb_mmap.flush()
    del emb_mmap

# Open for read+write
emb_mmap = np.lib.format.open_memmap(str(EMB_PATH), mode="r+")
print(f"emb.npy shape: {emb_mmap.shape}, dtype: {emb_mmap.dtype}")
print(f"Already done: {len(done_files)} files")
""", "resume"))

# =============================================================================
# Cell 7: Inference loop
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Inference loop — process file-by-file, batch chunks per file
# ============================================================
# Build (filename → list of (chunk_idx, row_idx)) lookup
file_to_chunks = {}
for fn, src, ci, ri in chunks:
    key = (fn, src)
    file_to_chunks.setdefault(key, []).append((ci, ri))

print(f"Total files to process: {len(file_to_chunks)}")
remaining_keys = [k for k in file_to_chunks.keys() if f"{k[1]}::{k[0]}" not in done_files]
print(f"Remaining: {len(remaining_keys)}")


def load_audio_full(path, sr_target=SR):
    '''Read full audio, resample to 32kHz mono float32.'''
    try:
        wav, sr_file = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr_file != sr_target:
            wav = librosa.resample(wav, orig_sr=sr_file, target_sr=sr_target)
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  failed to read {path}: {e}")
        return None


def chunkify(wav, chunk_samples):
    '''Split waveform into non-overlapping 5s chunks. Pad last if needed.'''
    n_chunks = max(1, len(wav) // chunk_samples)
    out = np.zeros((n_chunks, chunk_samples), dtype=np.float32)
    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        seg = wav[start:end]
        if len(seg) < chunk_samples:
            out[i, :len(seg)] = seg  # pad with zeros
        else:
            out[i] = seg
    return out


# Process files in order, batching chunks across files when possible
n_processed = 0
n_chunks_done = 0
session_start = time.time()
batch_wavs = []      # list of np.float32 (CHUNK_SAMPLES,)
batch_rows = []      # list of int row_idx
last_save = time.time()

def flush_batch():
    global batch_wavs, batch_rows
    if not batch_wavs:
        return
    arr = np.stack(batch_wavs, axis=0)
    emb = perch_forward(arr)
    for i, ri in enumerate(batch_rows):
        emb_mmap[ri] = emb[i]
    batch_wavs = []
    batch_rows = []


for fi, key in enumerate(remaining_keys):
    fn, src = key
    if src == "focal":
        path = TA_DIR / fn
    else:
        path = TS_DIR / fn

    wav = load_audio_full(path)
    if wav is None:
        # mark done with zero emb (already initialized)
        done_files.add(f"{src}::{fn}")
        continue

    chunk_arr = chunkify(wav, CHUNK_SAMPLES)
    chunk_meta = file_to_chunks[key]
    # chunk_meta length should match chunk_arr first dim; clip to min
    n_actual = min(len(chunk_arr), len(chunk_meta))
    for ci_idx in range(n_actual):
        batch_wavs.append(chunk_arr[ci_idx])
        batch_rows.append(chunk_meta[ci_idx][1])  # row_idx
        n_chunks_done += 1
        if len(batch_wavs) >= PERCH_BATCH:
            flush_batch()

    done_files.add(f"{src}::{fn}")
    n_processed += 1

    if (n_processed % 500) == 0:
        flush_batch()
        elapsed = time.time() - session_start
        rate = n_processed / max(elapsed, 1)
        remaining = len(remaining_keys) - n_processed
        eta_sec = remaining / max(rate, 0.01)
        total_elapsed = time.time() - TRAIN_START
        print(f"  [{n_processed}/{len(remaining_keys)}] files, "
              f"{n_chunks_done} chunks, "
              f"{rate:.1f} f/s, ETA {eta_sec/60:.1f}min, "
              f"total {total_elapsed/60:.1f}min",
              flush=True)

    # Periodic save
    if time.time() - last_save > 300:  # every 5 min
        flush_batch()
        emb_mmap.flush()
        DONE_FILES_PATH.write_text("\n".join(sorted(done_files)))
        last_save = time.time()
        print(f"    [save] flushed mmap + done_files ({len(done_files)} files)")

    # Time budget check
    total_elapsed = time.time() - TRAIN_START
    if total_elapsed > MAX_RUNTIME_SEC:
        print(f"[stop] time budget exhausted at {total_elapsed/60:.1f}min")
        break

flush_batch()
emb_mmap.flush()
DONE_FILES_PATH.write_text("\n".join(sorted(done_files)))
print(f"\nProcessed: {n_processed} new files, {n_chunks_done} new chunks")
print(f"Total done: {len(done_files)} / {len(file_to_chunks)} files")
""", "infer_loop"))

# =============================================================================
# Cell 8: Stats + save
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: Sanity stats
# ============================================================
emb_view = np.array(emb_mmap[:min(1000, N_TOTAL)])
print(f"emb stats (first 1000 rows):")
print(f"  min: {emb_view.min():.4f}, max: {emb_view.max():.4f}")
print(f"  mean: {emb_view.mean():.4f}, std: {emb_view.std():.4f}")
print(f"  zero rows: {(np.abs(emb_view).sum(axis=1) == 0).sum()} / 1000")
print(f"\nemb.npy size: {EMB_PATH.stat().st_size/1e9:.2f} GB")
print(f"meta.csv rows: {len(meta_df)}")

# Done file count vs total
n_done_files = len(done_files)
n_total_files = len(file_to_chunks)
done_frac = n_done_files / n_total_files if n_total_files > 0 else 0
print(f"\nFile completion: {n_done_files}/{n_total_files} ({100*done_frac:.1f}%)")

# Zero-row count over full emb
print("Counting zero rows over full emb...")
emb_chunk_size = 100000
n_zero_total = 0
for s in range(0, N_TOTAL, emb_chunk_size):
    e = min(s + emb_chunk_size, N_TOTAL)
    chunk_view = np.array(emb_mmap[s:e])
    n_zero_total += int((np.abs(chunk_view).sum(axis=1) == 0).sum())
print(f"  zero rows total: {n_zero_total} / {N_TOTAL} "
      f"({100*n_zero_total/N_TOTAL:.1f}%)")
""", "stats"))

# =============================================================================
# Cell 9: Upload to Kaggle Dataset
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 9: Upload emb.npy + meta.csv + done_files.txt to Kaggle Dataset
# ============================================================
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

DATASET_USER = "maekeso"
DATASET_SLUG = "birdclef2026-perch-emb-cache"
DATASET_TITLE = "BirdCLEF 2026 Perch v2 embedding cache (5s chunks)"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_copied = 0
    for f in [EMB_PATH, OUT_DIR / "meta.csv", DONE_FILES_PATH]:
        if f.exists():
            shutil.copy(str(f), str(td / f.name))
            n_copied += 1
            print(f"  staged {f.name}: {f.stat().st_size/1e9:.3f} GB")
    print(f"Staged {n_copied} files")

    meta = {
        "title": DATASET_TITLE,
        "id": f"{DATASET_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_done = len(done_files)
    version_notes = f"emb {n_done} files done"
    uploaded = False
    try:
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"OK Uploaded {DATASET_USER}/{DATASET_SLUG} ({version_notes})")
        uploaded = True
    except Exception as e:
        msg = str(e)
        print(f"  dataset_create_version error: {msg[:300]}")
        if "not found" in msg.lower() or "404" in msg or "Could not find dataset" in msg:
            try:
                api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
                print(f"OK Created {DATASET_USER}/{DATASET_SLUG} (first time)")
                uploaded = True
            except Exception as e2:
                print(f"  dataset_create_new error: {str(e2)[:300]}")

    if not uploaded:
        print("\nUpload failed — files remain in /kaggle/working/ for manual upload")
        print("Files:")
        for p in sorted(Path('/kaggle/working').iterdir()):
            if p.is_file():
                print(f"  {p.name}  {p.stat().st_size/1e6:.2f} MB")
""", "upload"))

# =============================================================================
# Cell 10: Summary
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 10: Session summary
# ============================================================
total_time = time.time() - TRAIN_START
print(f"\n{'='*60}")
print(f"Perch pre-compute summary")
print(f"{'='*60}")
print(f"  Total chunks:   {N_TOTAL}")
print(f"  Files done:     {len(done_files)} / {len(file_to_chunks)}")
print(f"  Session time:   {total_time/60:.1f} min")
print(f"  Output:         emb.npy ({EMB_PATH.stat().st_size/1e9:.2f} GB)")
if len(done_files) >= len(file_to_chunks):
    print(f"\n  >>> Pre-compute COMPLETE <<<")
else:
    rem = len(file_to_chunks) - len(done_files)
    print(f"\n  ... {rem} files remaining; Save & Run All again to continue")
""", "summary"))

# =============================================================================
# Assemble notebook
# =============================================================================
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
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb_perch_precompute.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
