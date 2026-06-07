import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

with tempfile.TemporaryDirectory() as td:
    api.dataset_download_file("maekeso/birdclef2026-exp081-weights", "history.json", path=td)
    p = Path(td) / "history.json"
    # Kaggle may return .json or .json.zip
    if not p.exists():
        zp = Path(td) / "history.json.zip"
        if zp.exists():
            import zipfile
            with zipfile.ZipFile(zp) as z:
                z.extractall(td)
    if p.exists():
        hist = json.loads(p.read_text(encoding="utf-8"))
        print(f"history.json keys: {list(hist.keys()) if isinstance(hist, dict) else 'list'}")
        if isinstance(hist, list):
            print(f"  entries: {len(hist)}")
            if hist:
                print(f"  first: {hist[0]}")
                print(f"  last: {hist[-1]}")
        elif isinstance(hist, dict):
            for k, v in hist.items():
                if isinstance(v, list):
                    print(f"  {k}: list[{len(v)}], first={v[0] if v else 'N/A'}, last={v[-1] if v else 'N/A'}")
                else:
                    print(f"  {k}: {v}")
    else:
        print("history.json not found")
