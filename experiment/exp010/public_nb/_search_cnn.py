"""Search Kaggle for CNN-based BC2026 NBs (efficientnet, nfnet, sed, hgnet, convnext)."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

def show(rows, label):
    print(f"\n=== {label} ({len(rows)} hits) ===")
    for k in rows:
        score = getattr(k, "totalScore", None) or "-"
        ref = getattr(k, "ref", "")
        title = getattr(k, "title", "")
        print(f"  {str(score):>8} | {ref:60s} | {title}")

queries = [
    "efficientnet sed", "nfnet", "sed only", "hgnet", "convnext", "cnn",
    "iter pseudo sed", "v2s", "asymmetric", "babych", "noisy student",
    "5fold", "ensemble sed", "tucker",
]
for q in queries:
    rows = api.kernels_list(
        competition="birdclef-2026", search=q,
        sort_by="scoreDescending", page_size=15,
    )
    show(rows, f"search='{q}'")
