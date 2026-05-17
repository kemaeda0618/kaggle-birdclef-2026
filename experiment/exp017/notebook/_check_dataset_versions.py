"""Check exp017 dataset versions and contents."""
import json, os
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp017-weights"

# Try with explicit version numbers
for v in [1, 2, 3, 4, 5]:
    try:
        files_res = api.dataset_list_files(SLUG, dataset_version_number=v)
        print(f"\nVersion {v}:")
        for f in files_res.files:
            print(f"  {f.name}")
    except Exception as e:
        err = str(e)[:200]
        if "not found" in err.lower() or "404" in err:
            print(f"\nVersion {v}: NOT FOUND")
            break
        print(f"\nVersion {v}: error: {type(e).__name__}: {err}")
        break

# Latest (default)
try:
    files_res = api.dataset_list_files(SLUG)
    print(f"\nLatest (default):")
    for f in files_res.files:
        print(f"  {f.name}")
except Exception as e:
    print(f"\nLatest: error: {str(e)[:200]}")
