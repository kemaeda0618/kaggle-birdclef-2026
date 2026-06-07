"""Identify which kernel each submission came from."""
import json, os
from pathlib import Path
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions('birdclef-2026')
print(f'total submissions: {len(subs)}')
for s in subs:
    d = str(s.date)
    if '2026-05-19' in d or '2026-05-20' in d or '2026-05-21' in d:
        # Get all attrs
        attrs = [a for a in dir(s) if not a.startswith('_')]
        sub_attrs = {a: getattr(s, a) for a in attrs if not callable(getattr(s, a, None))}
        print(f'  ver={s.ref} pub={s.public_score}')
        for k, v in sub_attrs.items():
            if k not in ['ref', 'public_score', 'private_score', 'date', 'description']:
                print(f'    {k}: {v}')
        print()
