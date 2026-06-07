"""Fetch latest run logs from exp049/050/051 NBs."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slugs = [
    "maekeso/birdclef2026-exp049-effv2s-r1",
    "maekeso/birdclef2026-exp050-convnext-pico-r1",
    "maekeso/birdclef2026-exp051-transformer-r1",
]

for slug in slugs:
    print(f"\n{'='*60}\n{slug}\n{'='*60}")
    try:
        status = api.kernels_status(slug)
        print(f"  status: {status}")
    except Exception as e:
        print(f"  status err: {str(e)[:200]}")

    # Try to get last run output
    try:
        with tempfile.TemporaryDirectory() as td:
            api.kernel_output(slug, td)
            for f in Path(td).rglob("*"):
                if f.is_file():
                    print(f"  output file: {f.name} ({f.stat().st_size} B)")
    except Exception as e:
        print(f"  output err: {str(e)[:200]}")
