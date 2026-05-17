"""Verify exp014-state-r2 dataset via dataset_view metadata API."""
import json, os, sys, io, tempfile
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
print(f"Trying dataset_view: {slug}")
try:
    md = api.dataset_view(slug)
    print(f"  ref: {getattr(md, 'ref', '?')}")
    print(f"  title: {getattr(md, 'title', '?')}")
    print(f"  totalBytes: {getattr(md, 'totalBytes', '?')}")
    print(f"  versionNumber: {getattr(md, 'versionNumber', '?')}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")

print()
print("Trying dataset_metadata_prepare (download just metadata)...")
with tempfile.TemporaryDirectory() as td:
    try:
        api.dataset_metadata(slug, path=str(td))
        for p in Path(td).iterdir():
            print(f"  {p.name}  {p.stat().st_size} bytes")
            if p.name == "dataset-metadata.json":
                print(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
