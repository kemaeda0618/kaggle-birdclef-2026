"""Find exact kernel for 0.923 submission and any exp029 submissions."""
import json, os
from pathlib import Path
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions('birdclef-2026')
print(f'all submissions ({len(subs)}):')
for s in subs:
    pub = s.public_score if s.public_score else 'NA'
    desc = (s.description or '')[:40]
    print(f'  {s.date} | ver={s.ref} | pub={pub} | url={s.url}')
