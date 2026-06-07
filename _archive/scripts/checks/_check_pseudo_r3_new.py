"""Check exp029 pseudo R3 train_sc NB (new clean version)."""
import json, os, tempfile, re
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp029-pseudo-r3-train-sc"

print(f"=== {slug} ===")
try:
    st = api.kernels_status(slug)
    if hasattr(st, "to_dict"): d = st.to_dict()
    elif isinstance(st, dict): d = st
    else: d = vars(st) if hasattr(st, "__dict__") else {"raw": str(st)}
    print(f"  status: {d}")
except Exception as e:
    print(f"  ERR: {e}")

# Output files
print(f"\n=== Output files ===")
with tempfile.TemporaryDirectory() as td:
    try:
        api.kernels_output(slug, path=td, quiet=True)
        for f in sorted(Path(td).rglob("*")):
            if f.is_file():
                print(f"  {f.name}  {f.stat().st_size/1e6:.1f}MB")
        # Log last lines
        log = next(Path(td).rglob("*.log"), None)
        if log:
            content = log.read_text(encoding="utf-8", errors="replace")
            print(f"\n=== Log last 40 lines ===")
            try:
                entries = json.loads(content)
                lines = [e.get("data", "").rstrip("\n") for e in entries if isinstance(e, dict) and "data" in e]
                lines = [l for l in lines if l.strip()]
                for l in lines[-40:]:
                    print(l)
            except Exception:
                print(content[-3000:])
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")
