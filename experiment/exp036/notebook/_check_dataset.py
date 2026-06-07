import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
slug = "maekeso/birdclef2026-exp036-softauc-weights"
# Try dataset_list_files (most direct check)
try:
    files = api.dataset_list_files(slug).files
    print(f"OK dataset exists: {slug}")
    print(f"  files: {len(files)}")
    for f in files[:10]:
        size = getattr(f, "totalBytes", None) or getattr(f, "total_bytes", None) or 0
        print(f"    {f.name}  {size/1e6:.2f} MB")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
