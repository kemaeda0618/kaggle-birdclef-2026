"""Download metadata CSVs to verify content."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
api = KaggleApi(); api.authenticate()

for slug, subdir in [("maekeso/birdclef2026-xc-api-dl-part1", "xc_api"),
                      ("maekeso/birdclef2026-xc-api-dl-part2", "xc_api_part2")]:
    print(f"\n{'='*60}\n{slug}\n{'='*60}")
    with tempfile.TemporaryDirectory() as td:
        for csv_name in ["_metadata.csv", "_species_summary.csv"]:
            try:
                api.dataset_download_file(slug, f"{subdir}/{csv_name}", path=td)
                p = next(Path(td).glob(f"*{csv_name}*"), None)
                if p and p.suffix == ".zip":
                    import zipfile
                    with zipfile.ZipFile(p) as zf:
                        zf.extractall(td)
                    p = next(Path(td).glob(f"*{csv_name}"), None)
                if p and p.exists():
                    df = pd.read_csv(p)
                    print(f"\n  {csv_name}: {len(df)} rows, columns: {df.columns.tolist()[:6]}")
                    if "scientific_name" in df.columns:
                        n_sp = df['scientific_name'].nunique()
                        print(f"    unique species: {n_sp}")
                    if "completion_pct" in df.columns:
                        completed_100 = (df["completion_pct"] >= 100).sum()
                        print(f"    completion 100%: {completed_100}/{len(df)}")
                        total_dl = df["downloaded"].sum()
                        total_url = df["total_urls"].sum()
                        print(f"    total dl: {total_dl}/{total_url} files")
                        if "total_mb" in df.columns:
                            total_gb = df["total_mb"].sum() / 1024
                            print(f"    total size: {total_gb:.2f} GB")
            except Exception as e:
                print(f"  err {csv_name}: {str(e)[:200]}")
