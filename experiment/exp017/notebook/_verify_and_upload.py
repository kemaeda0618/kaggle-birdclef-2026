"""Verify already-downloaded emb.npy is complete, then upload as Kaggle Dataset.

Skips re-download; uses files already in _perch_cache_work/kernel_output/.
"""
import json, os, sys, io, shutil, time
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

import numpy as np
import pandas as pd

WORK = Path.cwd() / "_perch_cache_work"
DL_DIR = WORK / "kernel_output"

# Verify emb.npy
print("=== Verify emb.npy ===")
emb_path = DL_DIR / "emb.npy"
assert emb_path.exists(), f"missing: {emb_path}"
print(f"  file size: {emb_path.stat().st_size / 1e9:.3f} GB")

try:
    emb = np.load(str(emb_path), mmap_mode="r")
    print(f"  shape: {emb.shape}")
    print(f"  dtype: {emb.dtype}")
    print(f"  first row stats: min={emb[0].min():.4f}, max={emb[0].max():.4f}")
    print(f"  random row stats: min={emb[len(emb)//2].min():.4f}, max={emb[len(emb)//2].max():.4f}")
    # Check zero rows
    n_zero_sample = 100000
    sample = np.array(emb[:n_zero_sample])
    n_zero = (np.abs(sample).sum(axis=1) == 0).sum()
    print(f"  zero rows in first {n_zero_sample}: {n_zero} ({100*n_zero/n_zero_sample:.1f}%)")
except Exception as e:
    print(f"  ERROR loading: {type(e).__name__}: {e}")
    sys.exit(1)

# Verify meta.csv
print("\n=== Verify meta.csv ===")
meta_path = DL_DIR / "meta.csv"
meta = pd.read_csv(meta_path)
print(f"  rows: {len(meta)}")
print(f"  cols: {list(meta.columns)}")
print(f"  sources: {meta['source'].value_counts().to_dict()}")
print(f"  row_idx max: {meta['row_idx'].max()}")
assert len(meta) == emb.shape[0], f"meta rows ({len(meta)}) != emb rows ({emb.shape[0]})"
print(f"  ✓ meta rows == emb rows")

# Verify done_files.txt
print("\n=== Verify done_files.txt ===")
done_path = DL_DIR / "done_files.txt"
done = done_path.read_text().splitlines()
print(f"  done files: {len(done)}")
print(f"  unique files in meta: {meta['filename'].nunique()}")

# Stage for upload
print("\n=== Stage files ===")
STAGE = WORK / "stage"
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir()
for fn in ["emb.npy", "meta.csv", "done_files.txt"]:
    src = DL_DIR / fn
    shutil.copy2(str(src), str(STAGE / fn))
    print(f"  staged: {fn} ({src.stat().st_size/1e6:.2f} MB)")

meta_json = {
    "title": "BirdCLEF 2026 Perch v2 embedding cache (5s chunks)",
    "id": "maekeso/birdclef2026-perch-emb-cache",
    "licenses": [{"name": "CC0-1.0"}],
}
(STAGE / "dataset-metadata.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
print(f"  staged: dataset-metadata.json")

# Upload
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("\n=== Check existence ===")
try:
    files_res = api.dataset_list_files("maekeso/birdclef2026-perch-emb-cache")
    dataset_exists = True
    print(f"  EXISTS → version_up")
except Exception:
    dataset_exists = False
    print(f"  NOT FOUND → create_new")

print(f"\n=== Upload (this may take 5-15 min for 1.1 GB) ===")
t0 = time.time()
if dataset_exists:
    api.dataset_create_version(folder=str(STAGE),
                                version_notes="perch v2 emb cache (~350k chunks, float16)",
                                dir_mode="zip", quiet=False)
    print(f"\nUploaded new version in {time.time()-t0:.1f}s")
else:
    api.dataset_create_new(folder=str(STAGE), public=False,
                            dir_mode="zip", quiet=False)
    print(f"\nCreated new dataset in {time.time()-t0:.1f}s")

print(f"\nURL: https://www.kaggle.com/datasets/maekeso/birdclef2026-perch-emb-cache")
