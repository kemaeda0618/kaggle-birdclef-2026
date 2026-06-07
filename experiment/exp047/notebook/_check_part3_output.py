"""Check Part 3 NB output."""
import json, os, tempfile, zipfile
from pathlib import Path
import pandas as pd

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp047-xc-api-dl-part3"
print(f"=== {slug} status ===")
print(api.kernels_status(slug))

# Download kernel output
out_dir = Path("/tmp/exp047_part3_out")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nDownloading output to {out_dir}...")
try:
    api.kernels_output(slug, str(out_dir))
    print("OK")
except Exception as e:
    print(f"err: {str(e)[:200]}")

# Look for CSVs
print("\n=== Output files ===")
for f in out_dir.rglob("*.csv"):
    print(f"  {f.relative_to(out_dir)}  ({f.stat().st_size/1024:.1f} KB)")

# Read species_summary
for f in out_dir.rglob("_species_summary.csv"):
    df = pd.read_csv(f)
    print(f"\n=== {f.name} ===")
    print(f"Total species queried: {len(df)}")
    if "completion_pct" in df.columns:
        completed = df[df["completion_pct"] >= 100]
        partial = df[(df["completion_pct"] > 0) & (df["completion_pct"] < 100)]
        nothing = df[df["completion_pct"] == 0]
        print(f"  Fully (100%): {len(completed)}")
        print(f"  Partial: {len(partial)}")
        print(f"  Nothing (0%): {len(nothing)}")
        print(f"  total downloaded: {df['downloaded'].sum()} files")
        print(f"  total size: {df['total_mb'].sum()/1024:.2f} GB")
        print(f"\n  Top 10 species:")
        print(df.sort_values("downloaded", ascending=False).head(10).to_string(index=False))
        print(f"\n  No data species:")
        print(df[df["completion_pct"] == 0].head(20).to_string(index=False))

# Read metadata
for f in out_dir.rglob("_metadata.csv"):
    df = pd.read_csv(f)
    print(f"\n=== {f.name} ===")
    print(f"Total files: {len(df)}")
    if "scientific_name" in df.columns:
        print(f"  Unique species: {df['scientific_name'].nunique()}")
