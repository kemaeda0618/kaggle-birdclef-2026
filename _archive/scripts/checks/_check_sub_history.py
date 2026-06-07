import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

subs = api.competition_submissions("birdclef-2026")
print(f"{'Date':<22} {'Score':<10} {'Description':<70}")
print("-" * 105)
for s in subs[:15]:
    date = str(s.date)[:19] if s.date else "-"
    score = str(s.public_score)[:8] if s.public_score else "?"
    desc = (s.description or "")[:68]
    print(f"{date:<22} {score:<10} {desc:<70}")
