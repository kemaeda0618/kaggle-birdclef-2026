"""Verify Datasets exist via multiple API methods."""
import json, os, requests
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Method 1: dataset_metadata (may require write)
for slug in ["maekeso/birdclef2026-xc-api-aves-audio",
             "maekeso/birdclef2026-xc-api-aves-audio-part1",
             "maekeso/birdclef2026-xc-api-aves-audio-part2"]:
    print(f"\n--- {slug} ---")
    try:
        meta = api.dataset_metadata(slug, "/tmp")
        meta_json = json.loads((Path("/tmp") / "dataset-metadata.json").read_text())
        info = meta_json.get("info", {})
        print(f"  EXISTS")
        print(f"  title: {info.get('title', 'n/a')}")
        print(f"  size: {info.get('totalBytes', 'n/a')}")
        # also get version info
        versions = info.get('versions', [])
        if versions:
            latest = versions[-1] if versions else {}
            print(f"  versions: {len(versions)}")
    except Exception as e:
        print(f"  NOT FOUND: {str(e)[:200]}")

# Method 2: List own datasets
print("\n\n=== Listing own datasets ===")
try:
    datasets = api.dataset_list(user="maekeso", search="xc-api")
    for d in datasets[:10]:
        print(f"  {d.ref}")
except Exception as e:
    print(f"  err: {str(e)[:200]}")
