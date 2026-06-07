"""Check v4 run output for iNat/AnuraSet structure."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp047-bc-inat-anura"
print(f"=== {slug} status ===")
print(api.kernels_status(slug))

# Download latest output
out_dir = Path("/tmp/bc_inat_anura_v4")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nDownloading latest run output...")
try:
    api.kernels_output(slug, str(out_dir))
    print("OK")
except Exception as e:
    print(f"err: {str(e)[:200]}")
