"""Search top public notebooks for BirdCLEF+ 2026."""
import json, os
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Search public NBs by competition ===")
print("\n--- Sort by votes (top voted) ---")
try:
    nbs = api.kernels_list(competition="birdclef-2026", sort_by="voteCount",
                            page_size=20, page=1)
    for n in nbs[:20]:
        ref = getattr(n, "ref", "?")
        title = getattr(n, "title", "?")
        votes = getattr(n, "totalVotes", 0)
        author = getattr(n, "author", "?")
        last_run = getattr(n, "lastRunTime", "?")
        print(f"  votes={votes:3d} | {author:25s} | {title[:60]}")
        print(f"    ref: {ref}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {e}")

print("\n--- Sort by relevance (popular) ---")
try:
    nbs = api.kernels_list(competition="birdclef-2026", sort_by="relevance",
                            page_size=15, page=1)
    for n in nbs[:15]:
        ref = getattr(n, "ref", "?")
        title = getattr(n, "title", "?")
        votes = getattr(n, "totalVotes", 0)
        print(f"  votes={votes:3d} | {title[:80]}")
        print(f"    ref: {ref}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {e}")

print("\n--- Sort by hotness (recent activity) ---")
try:
    nbs = api.kernels_list(competition="birdclef-2026", sort_by="hotness",
                            page_size=15, page=1)
    for n in nbs[:15]:
        ref = getattr(n, "ref", "?")
        title = getattr(n, "title", "?")
        votes = getattr(n, "totalVotes", 0)
        print(f"  votes={votes:3d} | {title[:80]}")
        print(f"    ref: {ref}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {e}")

print("\n--- Sort by date created (newest) ---")
try:
    nbs = api.kernels_list(competition="birdclef-2026", sort_by="dateCreated",
                            page_size=15, page=1)
    for n in nbs[:15]:
        ref = getattr(n, "ref", "?")
        title = getattr(n, "title", "?")
        votes = getattr(n, "totalVotes", 0)
        print(f"  votes={votes:3d} | {title[:80]}")
        print(f"    ref: {ref}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {e}")
