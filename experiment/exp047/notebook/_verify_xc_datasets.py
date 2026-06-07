"""Verify Datasets exist + check contents."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

for slug in ["maekeso/birdclef2026-xc-api-dl-part1",
             "maekeso/birdclef2026-xc-api-dl-part2"]:
    print(f"\n{'='*60}\n{slug}\n{'='*60}")
    try:
        files = api.dataset_list_files(slug).files
        total_size = sum(f.total_bytes for f in files)
        print(f"  files: {len(files)}")
        print(f"  total: {total_size/1e9:.2f} GB")
        # Show top dirs
        dirs = set()
        species_set = set()
        for f in files:
            parts = f.name.split("/")
            if len(parts) >= 2:
                dirs.add(parts[0])
            if len(parts) >= 3 and parts[0] == "audio":
                species_set.add(parts[1])
            elif len(parts) >= 2 and "audio/" in f.name:
                species_set.add(parts[-2])
        print(f"  top dirs: {sorted(dirs)[:5]}")
        print(f"  species: {len(species_set)}")
        # Sample files
        mp3s = [f for f in files if f.name.endswith(".mp3")]
        csvs = [f for f in files if f.name.endswith(".csv")]
        print(f"  mp3 files: {len(mp3s)}")
        print(f"  csv files: {len(csvs)}")
        for c in csvs[:5]:
            print(f"    {c.name}")
    except Exception as e:
        print(f"  err: {str(e)[:200]}")
