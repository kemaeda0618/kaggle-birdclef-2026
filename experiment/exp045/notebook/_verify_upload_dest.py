"""Check which Kaggle Dataset received exp045 upload."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Check both
for slug in ["maekeso/birdclef2026-exp045-l1-filtered", "maekeso/birdclef2026-exp029-l1-single"]:
    print(f"\n=== {slug} ===")
    try:
        files = api.dataset_list_files(slug).files
        print(f"  files ({len(files)}):")
        for f in sorted(files, key=lambda x: getattr(x, 'creation_date', '')):
            cd = getattr(f, 'creation_date', '?')
            sz = f.total_bytes / 1e6
            print(f"    {f.name:60s} {sz:8.1f} MB  {cd}")
    except Exception as e:
        print(f"  ERR: {e}")
