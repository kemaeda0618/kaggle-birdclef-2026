"""Check exp028 pseudo_exp029 kernel status. If complete, download pseudo CSV."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029"

print(f"=== {slug} ===")
try:
    st = api.kernels_status(slug)
    if hasattr(st, "to_dict"): d = st.to_dict()
    elif isinstance(st, dict): d = st
    else: d = vars(st) if hasattr(st, "__dict__") else {"raw": str(st)}
    print(f"  status: {d}")
except Exception as e:
    print(f"  status ERR: {type(e).__name__}: {str(e)[:200]}")

# Check what files are in output
print(f"\n=== Output files ===")
with tempfile.TemporaryDirectory() as td:
    try:
        api.kernels_output(slug, path=td, quiet=True)
        for f in sorted(Path(td).rglob("*")):
            if f.is_file():
                print(f"  {f.name}  {f.stat().st_size/1e6:.1f}MB")
    except Exception as e:
        print(f"  output ERR: {type(e).__name__}: {str(e)[:200]}")
