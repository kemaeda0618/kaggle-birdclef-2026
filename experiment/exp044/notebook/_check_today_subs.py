"""Today's submission count check."""
import json, os
from pathlib import Path
from datetime import datetime
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions('birdclef-2026')
today_jst_date = "2026-05-21"  # JST today
# Kaggle stores UTC. JST = UTC+9. JST 2026-05-21 00:00 = UTC 2026-05-20 15:00
# Daily reset is at midnight UTC (Kaggle uses UTC for daily quotas)
today_utc_date = "2026-05-21"
print("=== Today (UTC 2026-05-21) submissions ===")
for s in subs:
    d = str(s.date)
    if today_utc_date in d:
        pub = s.public_score or 'NA'
        print(f"  {d} | pub={pub} | url={s.url[:80]}")
print()
n_today = sum(1 for s in subs if today_utc_date in str(s.date))
print(f"Today total: {n_today}/5")
print(f"Remaining: {5-n_today}")
