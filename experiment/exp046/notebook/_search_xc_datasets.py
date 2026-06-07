"""Search Kaggle for XC datasets covering BC2026 species (South American / Pantanal)."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

queries = [
    "xeno canto south america",
    "xeno canto brazil pantanal",
    "birdclef xeno canto",
    "xeno canto neotropical",
    "xeno canto extended birdclef 2024 2025",
    "birdset",
]

for q in queries:
    print(f"\n=== {q} ===")
    try:
        results = api.dataset_list(search=q, sort_by="hottest", max_size=None)
        for r in results[:8]:
            try:
                size_gb = r.totalBytes / 1e9 if hasattr(r, 'totalBytes') else 'n/a'
            except: size_gb = 'n/a'
            print(f"  {r.ref}  size={size_gb}  downloads={getattr(r, 'downloadCount', 'n/a')}")
    except Exception as e:
        print(f"  err: {e}")
