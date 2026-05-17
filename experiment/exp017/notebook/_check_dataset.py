"""Quick check of maekeso/birdclef2026-exp017-weights versions and files."""
import json, os
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

SLUG = "maekeso/birdclef2026-exp017-weights"

# Try version listing via meta endpoint
try:
    meta_res = api.metadata_get("maekeso", "birdclef2026-exp017-weights")
    print("Metadata:")
    print(json.dumps(meta_res if isinstance(meta_res, dict) else meta_res.__dict__,
                     indent=2, default=str)[:1500])
except Exception as e:
    print(f"metadata_get error: {type(e).__name__}: {str(e)[:300]}")
print()

# Files (default = latest version)
try:
    files_res = api.dataset_list_files(SLUG)
    print(f"Files in {SLUG} (latest version):")
    for f in files_res.files:
        print(f"  {f.name}")
except Exception as e:
    print(f"list_files error: {type(e).__name__}: {str(e)[:300]}")
