"""Search Kaggle for Xeno-Canto / BirdSet datasets."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()


def show(rows, label):
    print(f"\n=== {label} ({len(rows)} hits) ===")
    for d in rows[:25]:
        ref = getattr(d, "ref", "")
        title = getattr(d, "title", "")[:80]
        size = getattr(d, "size", "?")
        votes = getattr(d, "voteCount", "?")
        print(f"  {ref:60s} | size={str(size):>12} | votes={str(votes):>4} | {title}")


queries = [
    "xeno canto",
    "xeno-canto",
    "xenocanto",
    "birdset",
    "bird vocalization",
    "birdclef xeno",
    "bird audio pretrain",
    "all bird species",
]
for q in queries:
    rows = api.dataset_list(search=q, sort_by="hottest", page=1)
    show(rows, f"search='{q}'")
