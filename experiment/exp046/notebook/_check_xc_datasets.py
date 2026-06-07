"""Check rohanrao/xeno-canto-bird-recordings-extended datasets."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slugs = [
    "rohanrao/xeno-canto-bird-recordings-extended-a-m",
    "rohanrao/xeno-canto-bird-recordings-extended-n-z",
]
for slug in slugs:
    print(f"\n=== {slug} ===")
    try:
        meta = api.dataset_metadata(slug, "/tmp")
        meta_json = json.loads((Path("/tmp/dataset-metadata.json")).read_text())
        info = meta_json.get("info", {})
        print(f"  title: {info.get('title')}")
        print(f"  subtitle: {info.get('subtitle')}")
        # Description first 500 chars
        desc = info.get('description', '')
        print(f"  description preview: {desc[:1000]}")
        print(f"  usability: {info.get('usabilityRating')}")
        print(f"  totalDownloads: {info.get('totalDownloads')}")
        print(f"  totalViews: {info.get('totalViews')}")
        print(f"  keywords: {info.get('keywords')}")
        print(f"  licenses: {info.get('licenses')}")
    except Exception as e:
        print(f"  meta err: {e}")

    try:
        files = api.dataset_list_files(slug).files
        print(f"  total files: {len(files)}")
        total_bytes = sum(f.total_bytes for f in files)
        print(f"  total size: {total_bytes/1e9:.2f} GB")
        # Show first 10 file names + sizes
        for f in sorted(files, key=lambda x: -x.total_bytes)[:10]:
            print(f"    {f.name:60s}  {f.total_bytes/1e6:8.1f} MB")
    except Exception as e:
        print(f"  files err: {e}")
