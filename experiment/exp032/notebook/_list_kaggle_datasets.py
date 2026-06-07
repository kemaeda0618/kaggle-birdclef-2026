"""List user's Kaggle datasets to confirm R1/R2 separation patterns."""
import json, os
from pathlib import Path
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
datasets = api.dataset_list(user="maekeso", search="exp", max_size=None)
print(f"total: {len(datasets)}")
for d in sorted(datasets, key=lambda x: str(x.ref)):
    print(f"  {d.ref}")
