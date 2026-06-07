import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Look for exp014 dataset
print("=== exp014 datasets ===")
ds = api.dataset_list(user="maekeso", search="exp014")
for d in ds[:5]:
    print(f"  {d.ref}")

# Or kernel output for exp014 R2
print("\n=== Kernel files for exp014-train-r2 ===")
try:
    files = api.kernels_list_files("maekeso/birdclef2026-exp014-train-r2")
    for f in files.files[:15]:
        print(f"  {f.name}")
except Exception as e:
    print(f"  ERR: {e}")
