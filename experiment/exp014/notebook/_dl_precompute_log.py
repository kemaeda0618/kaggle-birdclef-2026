"""Download Perch pre-compute NB output/log to investigate ERROR."""
import json, os, sys, io, shutil
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
OUT = Path("C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp014/notebook/_precompute_log")
OUT.mkdir(parents=True, exist_ok=True)

print(f"Downloading {USER}/{SLUG} output to {OUT}")
try:
    api.kernels_output(f"{USER}/{SLUG}", path=str(OUT), quiet=False, force=True)
    print("\nFiles downloaded:")
    for p in sorted(OUT.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}  {size_kb:.1f} KB")
except Exception as e:
    print(f"kernels_output error: {e}")
