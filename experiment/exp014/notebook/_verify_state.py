"""Verify maekeso/exp014-state dataset exists and is attached to infer NB."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# 1. Does the Dataset exist?
print("=== maekeso/exp014-state files ===")
try:
    files = api.dataset_list_files("maekeso/exp014-state").files
    for f in files[:30]:
        print(f"  {f.name:50s} {f.total_bytes/1024/1024:.2f} MB")
    print(f"  Total: {len(files)} files")
except Exception as e:
    print(f"  ✗ Dataset not found: {e}")

# 2. Try downloading the latest version to verify
print("\n=== Verifying latest version ===")
try:
    info = api.dataset_view("maekeso/exp014-state")
    print(f"  ID: {info.id}")
    print(f"  Title: {info.title}")
    print(f"  Last updated: {info.last_updated}")
except Exception as e:
    print(f"  view failed: {e}")
