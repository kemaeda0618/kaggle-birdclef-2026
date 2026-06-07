"""Check status of Part 1 and Part 2 NBs + their output Datasets."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Kernel Status ===")
kernel_slugs = [
    "maekeso/birdclef2026-exp047-xc-api-dl",        # original
    "maekeso/birdclef2026-exp047-xc-api-dl-part1",  # new (user mentioned)
    "maekeso/birdclef2026-exp047-xc-api-dl-part2",
]
for slug in kernel_slugs:
    print(f"\n--- {slug} ---")
    try:
        status = api.kernels_status(slug)
        print(f"  status: {status}")
    except Exception as e:
        print(f"  err: {str(e)[:200]}")

print("\n\n=== Dataset Status ===")
dataset_slugs = [
    "maekeso/birdclef2026-xc-api-aves-audio",
    "maekeso/birdclef2026-xc-api-aves-audio-part2",
]
for slug in dataset_slugs:
    print(f"\n--- {slug} ---")
    try:
        # Try get_dataset_info or similar
        files = api.dataset_list_files(slug).files
        sum_bytes = sum(f.total_bytes for f in files)
        print(f"  EXISTS, files: {len(files)}, total: {sum_bytes/1e9:.2f} GB")
        for f in sorted(files, key=lambda x: -x.total_bytes)[:5]:
            print(f"    {f.name[:60]:60s} {f.total_bytes/1e6:.1f} MB")
    except Exception as e:
        print(f"  NOT FOUND or err: {str(e)[:200]}")
