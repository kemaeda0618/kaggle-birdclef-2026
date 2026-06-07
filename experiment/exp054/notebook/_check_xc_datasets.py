"""Check XC Datasets actual contents."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for ds in ["maekeso/birdclef2026-xc-api-dl-part1",
           "maekeso/birdclef2026-xc-api-dl-part2",
           "maekeso/birdclef2026-xc-api-dl-part3"]:
    print(f"\n=== {ds} ===")
    try:
        files = api.dataset_list_files(ds)
        flist = files.files if hasattr(files, 'files') else []
        print(f"  total files: {len(flist)}")
        if flist:
            # sample by extension
            exts = {}
            for f in flist:
                ext = Path(f.name).suffix.lower()
                exts[ext] = exts.get(ext, 0) + 1
            print(f"  by ext: {exts}")
            for f in flist[:5]:
                print(f"    {f.name}  ({getattr(f, 'totalBytes', 0)/1e6:.2f} MB)")
    except Exception as e:
        print(f"  err: {str(e)[:150]}")
