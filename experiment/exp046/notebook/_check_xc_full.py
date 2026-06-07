"""Get full file list with pagination + check coverage."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try downloading metadata CSV from the dataset (usually small)
for slug in ["rohanrao/xeno-canto-bird-recordings-extended-a-m",
             "rohanrao/xeno-canto-bird-recordings-extended-n-z"]:
    print(f"\n=== {slug} ===")
    with tempfile.TemporaryDirectory() as td:
        try:
            # try to download just the CSV
            api.dataset_download_file(slug, "train_extended.csv", path=td)
            csv_path = next(Path(td).glob("*.csv*"), None) or next(Path(td).glob("*"), None)
            if csv_path:
                # uncompress if needed
                if csv_path.suffix == ".zip":
                    import zipfile
                    with zipfile.ZipFile(csv_path, "r") as zf:
                        zf.extractall(td)
                    csv_path = next(Path(td).glob("*.csv"), None)
                if csv_path and csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    print(f"  CSV rows: {len(df)}")
                    print(f"  CSV columns: {df.columns.tolist()[:15]}")
                    if "species" in df.columns:
                        species = df["species"].nunique()
                        print(f"  unique species: {species}")
                        print(f"  species sample: {df['species'].unique()[:5].tolist()}")
                    if "scientific_name" in df.columns:
                        sci = df["scientific_name"].nunique()
                        print(f"  unique scientific_name: {sci}")
                        print(f"  scientific_name sample: {df['scientific_name'].unique()[:5].tolist()}")
                    if "ebird_code" in df.columns:
                        print(f"  unique ebird_code: {df['ebird_code'].nunique()}")
                    if "duration" in df.columns:
                        total_h = df["duration"].sum() / 3600
                        print(f"  total duration: {total_h:.1f} hours")
        except Exception as e:
            print(f"  err: {e}")
