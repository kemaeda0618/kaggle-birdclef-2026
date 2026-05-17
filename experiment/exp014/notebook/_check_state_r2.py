"""Check maekeso/exp014-state-r2 dataset (Session 1 upload)."""
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

slug = "maekeso/exp014-state-r2"
print(f"=== {slug} ===")
try:
    files = api.dataset_list_files(slug).files
    for f in files[:20]:
        print(f"  {f.name:50s} {f.total_bytes/1024/1024:.2f} MB")
    print(f"  Total: {len(files)} files")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
