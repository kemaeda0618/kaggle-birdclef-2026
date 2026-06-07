"""Check V3 + V1 status, download outputs if complete."""
import json, os, time
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

DATA_DIR = Path("experiment/streamlit_analyze/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for slug in [
    "maekeso/birdclef2026-exp091-v3-exp090-oof",   # V3 exp090 OOF
    "maekeso/birdclef2026-exp091-analysis-infer",  # V1 Perch raw
]:
    print(f"\n=== {slug} ===")
    try:
        st = api.kernels_status(slug)
        # status is a dict-like object
        if hasattr(st, "to_dict"):
            d = st.to_dict()
        elif isinstance(st, dict):
            d = st
        else:
            d = vars(st) if hasattr(st, "__dict__") else {"raw": str(st)}
        print(f"  status: {d}")
    except Exception as e:
        print(f"  status ERR: {type(e).__name__}: {str(e)[:200]}")

    # Try download
    try:
        print(f"  downloading to {DATA_DIR}/...")
        api.kernels_output(slug, path=str(DATA_DIR), quiet=False)
    except Exception as e:
        print(f"  download ERR: {type(e).__name__}: {str(e)[:200]}")

print(f"\n=== Files in {DATA_DIR} ===")
for p in sorted(DATA_DIR.iterdir()):
    if p.is_file():
        print(f"  {p.name}  {p.stat().st_size/1e6:.2f}MB")
