"""Search Kaggle BirdCLEF 2026 NBs that mention BirdNET."""
import json, os
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("=== Search: 'birdnet' ===")
nbs = api.kernels_list(
    competition="birdclef-2026", search="birdnet",
    sort_by="scoreDescending", page_size=20,
)
for k in nbs:
    score = getattr(k, "totalScore", None)
    score_s = f"{score:.4f}" if score is not None else "—"
    title = getattr(k, "title", "")
    print(f"  score={score_s}  ref={k.ref}  title={title!r}")

print("\n=== Search: 'bird-net' ===")
nbs = api.kernels_list(
    competition="birdclef-2026", search="bird-net",
    sort_by="scoreDescending", page_size=10,
)
for k in nbs:
    score = getattr(k, "totalScore", None)
    score_s = f"{score:.4f}" if score is not None else "—"
    title = getattr(k, "title", "")
    print(f"  score={score_s}  ref={k.ref}  title={title!r}")
