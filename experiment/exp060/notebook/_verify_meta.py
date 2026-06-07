import json, os, tempfile, sys, io
from pathlib import Path
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if not os.environ.get('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['key']
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ['birdclef2026-exp060-blend-exp048-exp020', 'birdclef2026-exp062-aves-exp032']:
    with tempfile.TemporaryDirectory() as td:
        try:
            api.kernels_pull(f'maekeso/{slug}', path=td, metadata=True, quiet=True)
            mf = next(Path(td).glob('kernel-metadata.json'), None)
            if mf:
                m = json.loads(mf.read_text())
                print(f'=== {slug} ===')
                ds = m.get("dataset_sources", [])
                ks = m.get("kernel_sources", [])
                cs = m.get("competition_sources", [])
                print(f'  datasets ({len(ds)}):')
                for d in ds: print(f'    - {d}')
                print(f'  kernels ({len(ks)}):')
                for k in ks: print(f'    - {k}')
                print(f'  competitions ({len(cs)}): {cs}')
                print()
        except Exception as e:
            print(f'{slug} err: {str(e)[:200]}')
