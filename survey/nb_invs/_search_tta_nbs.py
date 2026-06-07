"""Search BC2026 NBs with TTA, sorted by score desc."""
import json, os, sys, io
from pathlib import Path
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Top scoring NBs for BC2026
print("=" * 60)
print("Top scoring BC2026 NBs")
print("=" * 60)
try:
    nbs = api.kernels_list(competition="birdclef-2026", sort_by="scoreDescending", page_size=30)
    for n in nbs[:30]:
        title = getattr(n, 'title', '?')
        score = getattr(n, 'totalScore', None) or getattr(n, 'score', None)
        ref = getattr(n, 'ref', '?')
        votes = getattr(n, 'totalVotes', None)
        print(f"  [{score}] {title[:50]}")
        print(f"    ref={ref}, votes={votes}")
except Exception as e:
    print(f"err: {str(e)[:300]}")

print()
print("=" * 60)
print("Search BC2026 NBs containing 'TTA'")
print("=" * 60)
try:
    nbs2 = api.kernels_list(competition="birdclef-2026", search="TTA", sort_by="scoreDescending", page_size=20)
    for n in nbs2:
        title = getattr(n, 'title', '?')
        score = getattr(n, 'totalScore', None) or getattr(n, 'score', None)
        ref = getattr(n, 'ref', '?')
        print(f"  [{score}] {title[:60]} | {ref}")
except Exception as e:
    print(f"err: {str(e)[:300]}")
