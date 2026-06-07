"""Check actual state of birdclef2026-exp032-weights dataset on Kaggle."""
import json, os
from pathlib import Path
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

owner = "maekeso"
slug = "birdclef2026-exp032-weights"

# Try to get info
try:
    info = api.dataset_metadata(f"{owner}/{slug}", "/tmp")
    print(f"metadata fetched: {info}")
except Exception as e:
    print(f"metadata err: {e}")

# List files
try:
    files = api.dataset_list_files(f"{owner}/{slug}").files
    print(f"\n=== {slug} files ({len(files)}) ===")
    for f in files:
        attrs = {k: getattr(f, k, None) for k in dir(f) if not k.startswith('_') and not callable(getattr(f, k, None))}
        print(f"  {f.name}: {attrs}")
except Exception as e:
    print(f"list err: {e}")
