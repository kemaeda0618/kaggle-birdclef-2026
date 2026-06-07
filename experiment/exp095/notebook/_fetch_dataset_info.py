"""Fetch exp095 R3 dataset metadata + file list."""
import json, os, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

OWNER = "maekeso"
SLUG = "birdclef2026-exp095-l0-r2spec"

# List files
print(f"=== {OWNER}/{SLUG} files ===")
try:
    files = api.dataset_list_files(f"{OWNER}/{SLUG}")
    for f in files.files:
        size = getattr(f, "totalBytes", getattr(f, "size", "?"))
        print(f"  {f.name}  ({size} bytes)")
except Exception as e:
    print(f"list_files err: {e}")

# Status
print(f"\n=== status ===")
try:
    s = api.dataset_status(f"{OWNER}/{SLUG}")
    print(f"  {s}")
except Exception as e:
    print(f"status err: {e}")
