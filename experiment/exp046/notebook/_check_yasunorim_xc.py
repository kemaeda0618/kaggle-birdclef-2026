"""Check yasunorim/xc-birdclef-2026-target-urls + whitecaphat BC2025 XC."""
import json, os, tempfile, zipfile
from pathlib import Path
import pandas as pd

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

bc26_tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
bc26_sci_lower = set(bc26_tax["scientific_name"].astype(str).str.lower())
bc26_labels = set(bc26_tax["primary_label"].astype(str))

for slug in [
    "yasunorim/xc-birdclef-2026-target-urls",
    "whitecaphat/birdclef2025-xc-audio",
    "atsunorifujita/birdclef-2023-additional",
    "ludovick/birdclef-merge-1",
    "mariotsaberlin/xeno-canto-extended-metadata-for-birdclef2023",
]:
    print(f"\n{'='*78}\n=== {slug} ===\n{'='*78}")
    try:
        meta_path = "/tmp"
        api.dataset_metadata(slug, meta_path)
        meta_json = json.loads((Path("/tmp/dataset-metadata.json")).read_text())
        info = meta_json.get("info", {})
        print(f"  title: {info.get('title')}")
        print(f"  subtitle: {info.get('subtitle')}")
        print(f"  description: {info.get('description', '')[:600]}")
        print(f"  totalDownloads: {info.get('totalDownloads')}")

        # List files
        files = api.dataset_list_files(slug).files
        sum_bytes = sum(f.total_bytes for f in files)
        print(f"  files: {len(files)}, total size: {sum_bytes/1e9:.2f} GB")
        for f in sorted(files, key=lambda x: -x.total_bytes)[:8]:
            print(f"    {f.name}  {f.total_bytes/1e6:.1f} MB")
    except Exception as e:
        print(f"  err: {e}")
