"""Verify (2) exp058 filtered_metadata.csv columns + (5) BC2026 secondary_labels format."""
import json, os, sys, tempfile, io
from pathlib import Path
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
api = KaggleApi(); api.authenticate()

# #2: exp058 filtered_metadata.csv
print("=" * 60)
print("#2: exp058 filtered_metadata.csv columns + sample")
print("=" * 60)
with tempfile.TemporaryDirectory() as td:
    try:
        api.dataset_download_file("maekeso/birdclef2026-exp058-xc-pseudo", "filtered_metadata.csv", path=td)
        p = next(Path(td).glob("*"))
        if p.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(p) as zf: zf.extractall(td)
            p = next(Path(td).glob("filtered_metadata.csv"))
        df = pd.read_csv(p)
        print(f"rows: {len(df)}")
        print(f"columns: {df.columns.tolist()}")
        print(f"\\nfirst 2 rows:")
        print(df.head(2).T)
        # Check key columns
        for k in ["primary_label", "filepath", "scientific_name", "recordist", "quality"]:
            present = k in df.columns
            print(f"  {k}: {'✓' if present else '✗ MISSING'}")
    except Exception as e:
        print(f"err: {str(e)[:300]}")

# #5: BC2026 train.csv secondary_labels format
print("\n" + "=" * 60)
print("#5: BC2026 train.csv secondary_labels format")
print("=" * 60)
local_train = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\train.csv")
if local_train.exists():
    df = pd.read_csv(local_train)
    print(f"rows: {len(df)}")
    print(f"columns: {df.columns.tolist()}")
    if "secondary_labels" in df.columns:
        unique_formats = df["secondary_labels"].astype(str).head(20).tolist()
        print(f"\\nfirst 20 secondary_labels values:")
        for v in unique_formats:
            print(f"  {repr(v)}")
        # Count types
        empty_str = (df["secondary_labels"].astype(str) == "[]").sum()
        nan_count = df["secondary_labels"].isna().sum()
        non_empty = (df["secondary_labels"].astype(str) != "[]").sum()
        print(f"\\nempty '[]': {empty_str}")
        print(f"NaN: {nan_count}")
        print(f"non-empty (with [...] format?): {non_empty}")
    else:
        print(f"  ✗ secondary_labels column MISSING")
else:
    print(f"  ✗ {local_train} not found")
