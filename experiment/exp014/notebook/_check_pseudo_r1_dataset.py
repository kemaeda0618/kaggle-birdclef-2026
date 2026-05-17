"""Check maekeso/birdclef2026-exp014-pseudo-r1 dataset."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ["maekeso/birdclef2026-exp014-pseudo-r1",
             "maekeso/exp014-r1-pseudo"]:
    print(f"=== {slug} ===")
    try:
        files = api.dataset_list_files(slug).files
        for f in files[:15]:
            print(f"  {f.name:50s} {f.total_bytes/1024/1024:.2f} MB")
        print(f"  Total: {len(files)} files")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {str(e)[:120]}")
    print()
