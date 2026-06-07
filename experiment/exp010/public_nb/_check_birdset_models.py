"""Check files / metadata in BirdSet pretrained model datasets."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

candidates = [
    "denden12/birdset-convnext-base-xcl",
    "dacquaviva/birdset-effnet-b1-xcl",
    "ttyn4519/perch-distill-birdset-sda-v1-models",
    "xinzenglovecat/birdset",
    "yasunorim/xc-birdclef-2026-target-urls",
    "whitecaphat/birdclef2025-xc-audio",
    "nomorevotch/xeno-canto-bird-recordings-extended-a-m-32khz-ogg",
]
for slug in candidates:
    print(f"\n=== {slug} ===")
    try:
        files = api.dataset_list_files(slug).files
        total_size = sum(getattr(f, "totalBytes", 0) or getattr(f, "total_bytes", 0) or 0 for f in files)
        print(f"  files: {len(files)}, total: {total_size/1e9:.2f} GB")
        for f in files[:15]:
            size = getattr(f, "totalBytes", None) or getattr(f, "total_bytes", None) or 0
            print(f"    {f.name}  {size/1e6:.1f} MB")
        if len(files) > 15:
            print(f"    ... +{len(files)-15} more")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
