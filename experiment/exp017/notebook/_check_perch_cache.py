"""Try download (small probe) to confirm dataset existence + check kernel status."""
import json, os, tempfile
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Check precompute kernel status
print("=== precompute kernel status ===")
try:
    s = api.kernels_status("maekeso/birdclef2026-exp014-perch-precompute")
    print(f"  {s}")
except Exception as e:
    print(f"  error: {type(e).__name__}: {str(e)[:200]}")

# Try dataset list_files for each
for slug in [
    "maekeso/birdclef2026-perch-emb-cache",
]:
    print(f"\n=== {slug} ===")
    try:
        files_res = api.dataset_list_files(slug)
        print(f"EXISTS, files:")
        for f in files_res.files:
            print(f"  {f.name}")
    except Exception as e:
        msg = str(e)
        if "403" in msg or "404" in msg:
            print(f"NOT FOUND or NO ACCESS (status: {msg[:100]})")
        else:
            print(f"error: {type(e).__name__}: {msg[:200]}")

# Try kernel output (precompute kernel output = the embedding files)
print("\n=== precompute kernel output ===")
try:
    out = api.kernels_output("maekeso/birdclef2026-exp014-perch-precompute", path="/tmp/_probe", force=False, quiet=True)
    files = list(Path("/tmp/_probe").rglob("*"))
    for f in files[:20]:
        if f.is_file():
            print(f"  {f.relative_to('/tmp/_probe')}  size={f.stat().st_size}")
except Exception as e:
    print(f"  error: {type(e).__name__}: {str(e)[:300]}")
