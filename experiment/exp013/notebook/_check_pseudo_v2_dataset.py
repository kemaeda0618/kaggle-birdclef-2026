"""Check pseudo-r1-v2 dataset contents."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Files in maekeso/birdclef2026-exp013-pseudo-r1-v2 ===")
files = api.dataset_list_files("maekeso/birdclef2026-exp013-pseudo-r1-v2").files
for f in files:
    print(f"  {f.name:40s} {f.total_bytes/1024/1024:.2f} MB")

# Also get NB1 v2 status + output stats
print("\n=== NB1 v2 status ===")
info = api.kernels_status("maekeso/birdclef2026-exp013-pl1-pseudo-gen-v2")
print(info)

# Download CSV and inspect
import tempfile
with tempfile.TemporaryDirectory() as td:
    print(f"\nDownloading pseudo_labels.csv to {td} ...")
    api.dataset_download_file(
        "maekeso/birdclef2026-exp013-pseudo-r1-v2",
        file_name="pseudo_labels.csv",
        path=td,
    )
    # unzip if .zip
    z = Path(td) / "pseudo_labels.csv.zip"
    if z.exists():
        import zipfile
        with zipfile.ZipFile(z) as zf:
            zf.extractall(td)
        z.unlink()
    csv = Path(td) / "pseudo_labels.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv, nrows=3)
        print(f"=== pseudo_labels.csv head ===")
        print(f"Shape preview: {df.shape}")
        print(df.iloc[:, :6])
        total = sum(1 for _ in open(csv)) - 1
        size_mb = csv.stat().st_size / 1024 / 1024
        print(f"\nTotal rows: {total}")
        print(f"File size: {size_mb:.1f} MB")
        # Compare to v1: 127,896 rows / 393 MB
        v1_rows = 127896
        keep_rate = total / v1_rows * 100
        print(f"vs NB1 v1 (127,896 rows, 393 MB): {keep_rate:.1f}% kept")
