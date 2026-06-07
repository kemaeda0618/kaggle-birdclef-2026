import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== cooolz/openvino-package files ===")
try:
    files = api.dataset_list_files("cooolz/openvino-package")
    for f in files.files[:30]:
        sz = getattr(f, "total_bytes", None) or getattr(f, "totalBytes", "?")
        print(f"  {f.name} ({sz})")
except Exception as e:
    print(f"  ERR: {e}")
