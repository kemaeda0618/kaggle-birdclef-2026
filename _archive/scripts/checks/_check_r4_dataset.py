"""Check if birdclef2026-exp029-r4-l1 Kaggle Dataset already exists."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp029-r4-l1"
print(f"=== {slug} ===")
try:
    files = api.dataset_list_files(slug)
    print(f"  EXISTS")
    if hasattr(files, "files"):
        for f in files.files:
            print(f"  - {f.name}  {f.totalBytes/1e6:.1f}MB")
    else:
        print(f"  files: {files}")
except Exception as e:
    msg = str(e)
    if "Not Found" in msg or "404" in msg or "404" in str(type(e)):
        print(f"  NOT FOUND (Dataset does not exist yet)")
    else:
        print(f"  ERR: {type(e).__name__}: {msg[:300]}")
