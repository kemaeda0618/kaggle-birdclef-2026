"""Convert exp014 perch-precompute kernel output → maekeso/birdclef2026-perch-emb-cache Dataset.

選択 DL: emb.npy, meta.csv, done_files.txt のみ。train_soundscapes_local の .ogg (25 GB) は skip。
"""
import json, os, sys, io, shutil, tempfile, time
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

KERNEL_USER  = "maekeso"
KERNEL_SLUG  = "birdclef2026-exp014-perch-precompute"
DATASET_USER  = "maekeso"
DATASET_SLUG  = "birdclef2026-perch-emb-cache"
DATASET_TITLE = "BirdCLEF 2026 Perch v2 embedding cache (5s chunks)"

WORK = Path.cwd() / "_perch_cache_work"
WORK.mkdir(exist_ok=True)
print(f"Working dir: {WORK}")

# Clear stale soundscape copies from prior run (~ tens of MB to ~25GB)
ss_dir = WORK / "kernel_output" / "train_soundscapes_local"
if ss_dir.exists():
    print(f"Clearing stale soundscape copies in {ss_dir} ...")
    shutil.rmtree(ss_dir)

STAGE = WORK / "stage"
if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir()

TARGETS = ["emb.npy", "meta.csv", "done_files.txt"]
print(f"\n[1/2] Downloading selected files from kernel '{KERNEL_USER}/{KERNEL_SLUG}'...")
for fn in TARGETS:
    print(f"  fetching {fn} ...")
    t0 = time.time()
    try:
        api.kernels_output_download_file(KERNEL_USER, KERNEL_SLUG, fn, str(STAGE))
        sz = (STAGE / fn).stat().st_size
        print(f"    done in {time.time()-t0:.1f}s, size={sz/1e6:.2f} MB")
    except Exception as e:
        print(f"    error: {type(e).__name__}: {str(e)[:300]}")

print(f"\nStaged files:")
for f in sorted(STAGE.iterdir()):
    if f.is_file():
        print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB")

assert (STAGE / "emb.npy").exists(), "emb.npy missing — kernel output may not have completed"

# dataset metadata
meta = {
    "title": DATASET_TITLE,
    "id": f"{DATASET_USER}/{DATASET_SLUG}",
    "licenses": [{"name": "CC0-1.0"}],
}
(STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"  + dataset-metadata.json")

# existence check
print(f"\n[2/2] Checking if {DATASET_USER}/{DATASET_SLUG} exists...")
try:
    files_res = api.dataset_list_files(f"{DATASET_USER}/{DATASET_SLUG}")
    dataset_exists = True
    print(f"  EXISTS, will create new VERSION")
except Exception as e:
    dataset_exists = False
    print(f"  NOT FOUND, will CREATE new dataset")

print(f"\nUploading from {STAGE}...")
t0 = time.time()
if dataset_exists:
    api.dataset_create_version(folder=str(STAGE),
                                version_notes="perch v2 emb cache from exp014 precompute",
                                dir_mode="zip", quiet=False)
    print(f"  OK new version uploaded in {time.time()-t0:.1f}s")
else:
    api.dataset_create_new(folder=str(STAGE), public=False,
                            dir_mode="zip", quiet=False)
    print(f"  OK new dataset created in {time.time()-t0:.1f}s")

print(f"\nDataset URL: https://www.kaggle.com/datasets/{DATASET_USER}/{DATASET_SLUG}")
