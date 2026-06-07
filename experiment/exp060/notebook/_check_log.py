import json, os, tempfile, sys, io
from pathlib import Path
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
with tempfile.TemporaryDirectory() as td:
    api.kernels_output('maekeso/birdclef2026-exp060-blend-exp048-exp020', path=td, quiet=True)
    print('files:', [f.name for f in Path(td).iterdir()])
    for f in Path(td).iterdir():
        if f.suffix == '.log':
            txt = f.read_text(encoding='utf-8', errors='ignore')
            print(f'--- log last 3000 chars ---')
            print(txt[-3000:])
