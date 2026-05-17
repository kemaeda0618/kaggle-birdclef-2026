"""Check maekeso/nb1g-birdnet-embed dataset contents."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Files in maekeso/nb1g-birdnet-embed ===")
files = api.dataset_list_files("maekeso/nb1g-birdnet-embed").files
for f in files:
    print(f"  {f.name:50s} {f.total_bytes/1024/1024:.2f} MB")
