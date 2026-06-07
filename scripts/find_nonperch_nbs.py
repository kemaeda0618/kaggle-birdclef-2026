"""Find public BC2026 NBs sorted by score descending, then filter for non-Perch ones.

We pull metadata + NB content to detect Perch usage (looking for "perch" / "Perch" /
"google/bird-vocalization-classifier" / "perch_v2" / "embedding_v2" etc).
"""
import json, os, sys, time
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("Listing BC2026 public NBs (sort: by score desc)...")
results = api.kernels_list(
    competition="birdclef-2026",
    sort_by="scoreDescending",
    page_size=50,
    output_type="all",
)
print(f"  {len(results)} returned")
print()

# Sometimes the API objects don't have a score directly. Let me inspect first one.
if results:
    sample = results[0]
    print("=== sample kernel attrs ===")
    for k in sorted(dir(sample)):
        if not k.startswith("_") and not callable(getattr(sample, k)):
            v = getattr(sample, k)
            if v is not None and v != "":
                print(f"  {k}: {str(v)[:120]}")
    print()

# Extract slugs
print("=== top results ===")
for i, r in enumerate(results[:30]):
    title = getattr(r, "title", "?")
    ref = getattr(r, "ref", "?")
    author = getattr(r, "author", "?")
    score = getattr(r, "totalVotes", None) or getattr(r, "score", "?")
    lang = getattr(r, "language", "?")
    print(f"[{i:>2d}] {ref:<60s}  score={score}  {title[:50]}")
