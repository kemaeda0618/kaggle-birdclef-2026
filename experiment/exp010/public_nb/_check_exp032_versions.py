import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Check dataset versions / metadata
slug = "maekeso/birdclef2026-exp032-weights"
print(f"=== {slug} ===")
try:
    info = api.dataset_view(slug)
    print(f"  title: {info.title}")
except AttributeError:
    pass

# list files (latest version)
files = api.dataset_list_files(slug).files
print(f"  latest version files: {len(files)}")
for f in files:
    size = getattr(f, "totalBytes", None) or getattr(f, "total_bytes", None) or 0
    print(f"    {f.name}  {size/1e6:.2f} MB")

# Try fetching version metadata via dataset_status or related
print(f"\n  attempting status...")
try:
    status = api.dataset_status(slug)
    print(f"  status: {status}")
except Exception as e:
    print(f"  err: {type(e).__name__}: {str(e)[:200]}")
