import json, os, sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
slug = "maekeso/birdclef2026-pseudo-diversity-analysis"

st = api.kernels_status(slug)
print("status:", getattr(st, "status", None))
print("failure:", getattr(st, "failure_message", None))
print()

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\pseudo_diversity\cache")
OUT.mkdir(parents=True, exist_ok=True)

# Wipe previous to avoid confusion
for f in OUT.iterdir():
    if f.is_file():
        f.unlink()

api.kernels_output(slug, path=str(OUT), quiet=True, force=True)
print(f"Downloaded to: {OUT}")
for f in sorted(OUT.iterdir()):
    sz = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name}  ({sz:,} bytes)")
