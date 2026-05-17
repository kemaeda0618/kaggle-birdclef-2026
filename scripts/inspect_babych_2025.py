"""Inspect Nikita Babych 2025 1st place ensemble dataset."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

r = api.dataset_list_files("nikitababich/birdclef2025-1st-place-ensemble")
files = r.files
print(f"Total files: {len(files)}")
print()

total = 0
ext_count = {}
for f in files:
    info = json.loads(str(f))
    name = info["name"]
    sz = info["totalBytes"]
    total += sz
    ext = Path(name).suffix.lower() or "(no ext)"
    ext_count[ext] = ext_count.get(ext, 0) + 1

print(f"Total size: {total/1e9:.2f} GB")
print()
print("Extensions:")
for ext, cnt in sorted(ext_count.items(), key=lambda x: -x[1]):
    print(f"  {ext}: {cnt}")
print()
print("First 30 files:")
for f in files[:30]:
    info = json.loads(str(f))
    print(f"  {info['name']:80s} {info['totalBytes']/1e6:.2f} MB")
print()
print("Last 20 files:")
for f in files[-20:]:
    info = json.loads(str(f))
    print(f"  {info['name']:80s} {info['totalBytes']/1e6:.2f} MB")
