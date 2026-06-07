"""Check bc-inat-anura kernel run output."""
import json, os, tempfile
from pathlib import Path
import pandas as pd
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp047-bc-inat-anura"
print(f"=== {slug} status ===")
print(api.kernels_status(slug))

# Download output
out_dir = Path("/tmp/bc_inat_anura_out")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nDownloading output...")
try:
    api.kernels_output(slug, str(out_dir))
    print("OK")
except Exception as e:
    print(f"err: {str(e)[:200]}")

print("\n=== Output files ===")
for f in out_dir.rglob("*"):
    if f.is_file():
        try:
            sz = f.stat().st_size
            print(f"  {f.relative_to(out_dir)}  ({sz/1024:.1f} KB)")
        except: pass

# Check per_source.csv
for f in out_dir.rglob("per_source.csv"):
    print(f"\n=== {f.name} ===")
    df = pd.read_csv(f)
    print(df.to_string(index=False))

# Check per_species.csv top
for f in out_dir.rglob("per_species.csv"):
    print(f"\n=== {f.name} (top 20) ===")
    df = pd.read_csv(f)
    print(df.head(20).to_string(index=False))
    print(f"Total: {len(df)} species")

# Check log
for f in out_dir.rglob("*.log"):
    print(f"\n=== {f.name} ===")
    text = f.read_text(encoding="utf-8", errors="ignore")
    # Find iNat / AnuraSet related lines
    for line in text.split("\n"):
        if any(k in line for k in ["iNat", "AnuraSet", "anura", "inat", "mounted", "skip", "matched"]):
            print(f"  {line}")
