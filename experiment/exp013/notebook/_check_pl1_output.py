"""Check the output of birdclef2026-exp013-pl1-pseudo-gen."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp013-pl1-pseudo-gen"

# Status
print("=== Notebook info ===")
info = api.kernels_status(f"{USER}/{SLUG}")
print(info)

# Download outputs to a tmp dir and inspect
import tempfile
with tempfile.TemporaryDirectory() as td:
    print(f"\n=== Downloading outputs to {td} ===")
    api.kernels_output(f"{USER}/{SLUG}", path=td)
    for f in sorted(Path(td).rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(td)!s:40s} {size/1024/1024:.2f} MB")
    # Inspect CSV head
    csv = Path(td) / "pseudo_labels.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv, nrows=5)
        print(f"\n=== pseudo_labels.csv head (first 5 rows, {df.shape[1]} cols) ===")
        print(df.iloc[:, :6])
        # Full row count
        full_rows = sum(1 for _ in open(csv)) - 1
        print(f"\nTotal rows: {full_rows}")
