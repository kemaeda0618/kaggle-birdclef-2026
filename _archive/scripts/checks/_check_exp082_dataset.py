import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ["birdclef2026-exp082-weights", "birdclef2026-exp081-weights", "birdclef2026-exp015-weights"]:
    print(f"\n=== {slug} ===")
    try:
        files = api.dataset_list_files(f"maekeso/{slug}")
        for f in files.files[:20]:
            sz = getattr(f, "total_bytes", None) or getattr(f, "totalBytes", "?")
            print(f"  {f.name} ({sz})")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")
