"""Check train_r2 v4 status."""
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

ref = "maekeso/birdclef2026-exp014-train-r2"
print(f"Checking {ref} status...")
try:
    s = api.kernels_status(ref)
    print(f"  status: {getattr(s, 'status', '?')}")
    print(f"  failureMessage: {getattr(s, 'failureMessage', None)}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
