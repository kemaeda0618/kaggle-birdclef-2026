"""Targeted search: SED-focused / CNN backbone-focused NBs in BC2026."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

def show(rows, label):
    print(f"\n=== {label} ({len(rows)} hits) ===")
    for k in rows:
        ref = getattr(k, "ref", "")
        title = getattr(k, "title", "")
        print(f"  {ref:60s} | {title}")

# Most relevant CNN queries
queries = [
    "sed standalone",
    "efficientnet",
    "efficientnetv2",
    "nfnet",
    "sed train",
    "single model",
    "0.93",  # search by score in title
    "0.94",
    "convnext",
]
for q in queries:
    rows = api.kernels_list(
        competition="birdclef-2026", search=q,
        sort_by="scoreDescending", page_size=20,
    )
    show(rows, f"search='{q}'")
