"""Check the Kaggle Dataset upload status for R1 student."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Files in maekeso/birdclef2026-exp013-r1-student-sed ===")
files = api.dataset_list_files("maekeso/birdclef2026-exp013-r1-student-sed").files
for f in files:
    print(f"  {f.name:40s} {f.total_bytes/1024/1024:.2f} MB")
