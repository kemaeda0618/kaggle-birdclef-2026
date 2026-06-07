"""Download kernel output to inspect what was extracted/downloaded."""
import json, os, tempfile, zipfile
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ["maekeso/birdclef2026-exp047-xc-api-dl",
             "maekeso/birdclef2026-exp047-extract-pretrain"]:
    print(f"\n{'='*60}\n{slug}\n{'='*60}")
    out_dir = Path(f"/tmp/{slug.split('/')[1]}_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Try various API methods
        methods_to_try = []
        for attr in dir(api):
            if "output" in attr.lower() or "download" in attr.lower():
                if not attr.startswith("_"):
                    methods_to_try.append(attr)
        print(f"  Available output methods: {[m for m in methods_to_try if 'kernel' in m.lower() or 'notebook' in m.lower()]}")

        # Try kernels_output (new API)
        if hasattr(api, "kernels_output"):
            api.kernels_output(slug, str(out_dir))
            for f in out_dir.rglob("*"):
                if f.is_file():
                    sz = f.stat().st_size
                    print(f"  {f.name} ({sz/1e6:.1f} MB)" if sz > 1000 else f"  {f.name} ({sz} B)")
    except Exception as e:
        print(f"  download err: {str(e)[:300]}")
