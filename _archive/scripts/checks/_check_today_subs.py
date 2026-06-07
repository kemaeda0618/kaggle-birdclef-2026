"""Check today's BC2026 submission status + ready kernels."""
import json, os, tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Get recent submissions
print("=== Recent submissions ===")
try:
    subs = api.competition_submissions("birdclef-2026")
    today_jst = (datetime.now(timezone.utc) + timedelta(hours=9)).strftime("%Y-%m-%d")
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_today = 0
    for s in subs[:20]:
        date_str = str(s.date)[:19] if hasattr(s, 'date') else "?"
        score = getattr(s, 'public_score', '?')
        desc = getattr(s, 'description', '?')[:60]
        is_today = today_utc in date_str or today_jst in date_str
        marker = "★TODAY" if is_today else "      "
        if is_today: n_today += 1
        print(f"  {marker} {date_str}  score={score}  | {desc}")
    print(f"\n  Today subs: {n_today}/5")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")
