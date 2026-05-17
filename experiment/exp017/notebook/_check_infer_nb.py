"""Check exp017 infer NB push status and attached dataset version."""
import json, os
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp017-infer"

# Status
try:
    status = api.kernels_status(SLUG)
    print(f"Kernel status:")
    print(f"  {json.dumps(status, indent=2, default=str)[:1500]}")
except Exception as e:
    print(f"kernels_status error: {type(e).__name__}: {str(e)[:300]}")
print()

# List kernels
try:
    kernels = api.kernels_list(user="maekeso", search="exp017-infer")
    print(f"Kernels matching exp017-infer:")
    for k in kernels:
        ref = getattr(k, "ref", "?")
        last = getattr(k, "lastRunTime", "?")
        ttype = getattr(k, "totalVotes", "?")
        print(f"  {ref}  last_run={last}")
except Exception as e:
    print(f"kernels_list error: {type(e).__name__}: {str(e)[:300]}")
