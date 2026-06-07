"""Check XC api dl NB status + try to fetch output for resume planning."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ["maekeso/birdclef2026-exp047-xc-api-dl",
             "maekeso/birdclef2026-exp047-extract-pretrain",
             "maekeso/birdclef2026-exp047-xc-audio-dl"]:
    print(f"\n=== {slug} ===")
    try:
        status = api.kernels_status(slug)
        print(f"  status: {status}")
    except Exception as e:
        print(f"  err: {e}")
