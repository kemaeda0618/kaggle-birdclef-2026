"""Check Kaggle kernel status and pull log/output for both OV NBs."""
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
SLUGS = ["birdclef2026-exp029-r3-ov", "birdclef2026-tucker-sed-ov"]

for slug in SLUGS:
    print(f"\n{'='*70}\n{USER}/{slug}\n{'='*70}")

    # status
    try:
        st = api.kernels_status(f"{USER}/{slug}")
        # st can be JSON string or object — handle both
        if isinstance(st, str):
            try:
                st_d = json.loads(st)
                print(f"status: {st_d.get('status')}, failureMessage: {st_d.get('failureMessage')}")
            except Exception:
                print(f"status raw: {st}")
        else:
            print(f"status object: {st}")
    except Exception as e:
        print(f"[status err] {type(e).__name__}: {e}")

    # output (log + files)
    with tempfile.TemporaryDirectory() as td:
        try:
            api.kernels_output(f"{USER}/{slug}", path=td, force=True, quiet=True)
            files = sorted(Path(td).rglob("*"))
            print(f"\noutput files in {td}:")
            for f in files:
                if f.is_file():
                    sz = f.stat().st_size
                    print(f"  {f.relative_to(td)}: {sz} bytes")
                    # show log content
                    if f.name in ("__notebook__.log", f"{slug}.log") or f.suffix == ".log":
                        print(f"  ---- LOG ({f.name}) ----")
                        try:
                            print(f.read_text(encoding="utf-8", errors="replace")[-8000:])
                        except Exception as e:
                            print(f"  [read err] {e}")
                        print("  ---- /LOG ----")
        except Exception as e:
            print(f"[output err] {type(e).__name__}: {str(e)[:300]}")
