"""Check exp090 last run log to measure full dry-run / sub time."""
import json, io, os, sys, tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    kj = Path.home() / ".kaggle" / "kaggle.json"
    creds = json.loads(kj.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
# exp090 NB slug
SLUG = "birdclef2026-exp090-tuned-tax"

print(f"\n{'='*70}\n{USER}/{SLUG}\n{'='*70}")
try:
    st = api.kernels_status(f"{USER}/{SLUG}")
    print(f"status: {st}")
except Exception as e:
    print(f"[status err] {e}")

with tempfile.TemporaryDirectory() as td:
    try:
        api.kernels_output(f"{USER}/{SLUG}", path=td, force=True, quiet=True)
        for f in sorted(Path(td).rglob("*")):
            if f.is_file():
                sz = f.stat().st_size
                print(f"  {f.relative_to(td)}: {sz} bytes")
                if f.suffix == ".log" or "notebook" in f.name.lower():
                    if f.suffix == ".log":
                        log = f.read_text(encoding="utf-8", errors="replace")
                        # Get last 3000 chars
                        print(f"\n----- LOG (last 3000 chars) -----\n{log[-3000:]}\n----- /LOG -----")
    except Exception as e:
        print(f"[output err] {e}")
