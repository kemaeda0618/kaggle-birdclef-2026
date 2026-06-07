"""List recent Kaggle submissions."""
import json, os
from pathlib import Path
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions('birdclef-2026')
print(f'total submissions: {len(subs)}')
print('=== recent submissions (2026-05-18 onward) ===')
for s in subs:
    d = str(s.date)
    if '2026-05-18' in d or '2026-05-19' in d or '2026-05-20' in d or '2026-05-21' in d:
        desc = (s.description or '')[:80]
        pub = s.public_score if s.public_score else 'NA'
        priv = s.private_score if s.private_score else 'NA'
        print(f'  {d} | ver={s.ref} | pub={pub} | priv={priv} | desc={desc}')
