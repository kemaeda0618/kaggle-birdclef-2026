"""Download train_r2 Session 1 log to check Cell 12 upload status."""
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

USER = "maekeso"
SLUG = "birdclef2026-exp014-train-r2"
ref  = f"{USER}/{SLUG}"

with tempfile.TemporaryDirectory() as td:
    print(f"Downloading {ref} output to {td}")
    try:
        api.kernels_output(ref, path=td, force=True)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
    files = sorted(Path(td).iterdir())
    for f in files:
        sz = f.stat().st_size
        print(f"  {f.name:60s} {sz/1024:.1f} KB")
    # find log file
    for f in files:
        if "log" in f.name.lower() or f.suffix in [".log", ".txt"]:
            print(f"\n=== LAST 200 LINES of {f.name} ===")
            text = f.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            for line in lines[-200:]:
                print(line)
            break
