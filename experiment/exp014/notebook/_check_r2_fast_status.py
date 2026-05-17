"""Check R2-fast NB status + verify emb cache dataset existence."""
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

print("=" * 60)
print("R2-fast NB status")
print("=" * 60)
try:
    st = api.kernels_status(f"{USER}/birdclef2026-exp014-train-r2-fast")
    print(f"Status: {st}")
except Exception as e:
    print(f"kernels_status error: {e}")

print("\n" + "=" * 60)
print("Pre-compute NB status (final)")
print("=" * 60)
try:
    st = api.kernels_status(f"{USER}/birdclef2026-exp014-perch-precompute")
    print(f"Status: {st}")
except Exception as e:
    print(f"kernels_status error: {e}")

print("\n" + "=" * 60)
print("emb cache dataset check via dataset_view")
print("=" * 60)
try:
    info = api.dataset_view(f"{USER}/birdclef2026-perch-emb-cache")
    print(f"Dataset exists:")
    print(f"  title: {info.title}")
    print(f"  size: {getattr(info, 'totalBytes', 'N/A')}")
    print(f"  lastUpdated: {getattr(info, 'lastUpdated', 'N/A')}")
    print(f"  versionNumber: {getattr(info, 'currentVersionNumber', 'N/A')}")
except Exception as e:
    print(f"dataset_view error: {e}")
