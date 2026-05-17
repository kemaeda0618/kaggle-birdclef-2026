"""Check perch precompute NB status via SDK."""
import json, os, sys, io
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp014-perch-precompute"

# Get kernel status
try:
    status = api.kernels_status(f"{USER}/{SLUG}")
    print(f"Status: {status}")
except Exception as e:
    print(f"kernels_status failed: {e}")

# List recent kernels
try:
    kernels = api.kernels_list(user=USER, search="perch", sort_by="dateRun", page_size=5)
    for k in kernels:
        print(f"  {k.ref}: title={k.title!r}, lastRunTime={getattr(k, 'lastRunTime', 'N/A')}, totalVotes={getattr(k, 'totalVotes', 'N/A')}")
except Exception as e:
    print(f"kernels_list failed: {e}")
