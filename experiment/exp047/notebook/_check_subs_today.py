"""Check today (2026-05-22 UTC) submission count."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions("birdclef-2026")

print("=== Submissions (last 7 days) ===")
for s in subs[:15]:
    d = str(s.date)[:19]
    pub = s.public_score or "NA"
    url = s.url[:75] if s.url else ""
    print(f"  {d} | pub={pub} | {url}")

# Count today (UTC 2026-05-22)
today_utc = "2026-05-22"
n_today = sum(1 for s in subs if today_utc in str(s.date))
print(f"\nToday ({today_utc}) submits: {n_today}/5")
print(f"Remaining: {5 - n_today}")
