"""Verify new pseudo Dataset content."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp029-pseudo-r3"
print(f"=== {slug} ===")
try:
    files = api.dataset_list_files(slug)
    if hasattr(files, "files"):
        for f in files.files:
            size = None
            for attr in ["totalBytes", "total_bytes", "size"]:
                if hasattr(f, attr):
                    size = getattr(f, attr); break
            print(f"  {f.name}  ({size/1e6:.1f}MB)" if size else f"  {f.name}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")
