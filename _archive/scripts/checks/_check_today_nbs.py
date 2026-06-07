import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in [
    "birdclef2026-exp081-infer-r1",
    "birdclef2026-ensemble-e14-17-r2-infer",
]:
    print(f"=== {slug} ===")
    try:
        st = api.kernels_status(f"maekeso/{slug}")
        print(f"  {st}")
        files = api.kernels_list_files(f"maekeso/{slug}")
        for f in files.files[:5]:
            print(f"  file: {f.name} ({f.size})")
    except Exception as e:
        print(f"  ERR: {e}")
    print()
