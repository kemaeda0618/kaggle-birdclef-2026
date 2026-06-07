import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ["birdclef2026-exp080-infer-pth", "birdclef2026-exp080a-adaptive-blend", "birdclef2026-exp079-blend-m7-overlay"]:
    print(f"=== maekeso/{slug} ===")
    try:
        st = api.kernels_status(f"maekeso/{slug}")
        print(f"  status: {st}")
        files = api.kernels_list_files(f"maekeso/{slug}")
        for f in files.files[:10]:
            print(f"  file: {f.name} ({f.size})")
    except Exception as e:
        print(f"  ERR: {e}")
    print()
