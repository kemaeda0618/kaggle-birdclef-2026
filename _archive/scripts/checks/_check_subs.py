import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

subs = api.competition_submissions("birdclef-2026")
print(f"{'Date':<22} {'PublicScore':<12} {'Description':<60}")
print("-" * 100)
for s in subs[:20]:
    date = str(s.date)[:19] if s.date else "-"
    score = str(s.publicScore)[:8] if s.publicScore else "?"
    desc = (s.description or "")[:58]
    print(f"{date:<22} {score:<12} {desc:<60}")
