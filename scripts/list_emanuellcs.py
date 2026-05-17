"""List all emanuellcs notebooks."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("All NBs by emanuellcs:")
nbs = api.kernels_list(user="emanuellcs", page_size=30)
for n in nbs:
    ref = getattr(n, "ref", "?")
    title = getattr(n, "title", "?")
    score = getattr(n, "totalScore", None)
    score_s = f"{score:.4f}" if score else "—"
    print(f"  {ref:60s} | LB={score_s} | {title}")
