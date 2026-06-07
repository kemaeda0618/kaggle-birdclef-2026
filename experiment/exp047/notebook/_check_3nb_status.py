"""Check status of 3 NBs and resulting Datasets."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NBS = [
    ("maekeso/birdclef2026-exp047-bc-extract", "maekeso/birdclef2026-exp047-bc-extracted"),
    ("maekeso/birdclef2026-exp047-inat-extract", "maekeso/birdclef2026-exp047-inat-extracted"),
    ("maekeso/birdclef2026-exp047-anuraset-extract", "maekeso/birdclef2026-exp047-anuraset-extracted"),
]

for nb_slug, ds_slug in NBS:
    print(f"\n=== {nb_slug} ===")
    try:
        s = api.kernels_status(nb_slug)
        print(f"  NB status: {s}")
    except Exception as e:
        print(f"  NB err: {str(e)[:120]}")
    try:
        files = api.dataset_list_files(ds_slug)
        print(f"  Dataset: {ds_slug}")
        n_files = len(files.files) if hasattr(files, 'files') else 0
        total_mb = sum(getattr(f, 'totalBytes', 0) for f in files.files) / 1e6 if hasattr(files, 'files') else 0
        print(f"    files: {n_files}, size: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        # Sample files
        if hasattr(files, 'files') and files.files:
            print(f"    Sample files (first 5):")
            for f in files.files[:5]:
                print(f"      {f.name}  ({getattr(f, 'totalBytes', 0)/1e6:.2f} MB)")
    except Exception as e:
        print(f"  Dataset CHECK ERR: {str(e)[:120]}")
