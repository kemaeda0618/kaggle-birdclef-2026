"""Try multiple methods to check if birdclef2026-exp029-r4-l1 exists."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp029-r4-l1"

# Method 1: dataset_view (metadata)
print("=== Method 1: dataset_view ===")
try:
    info = api.dataset_view(slug)
    print(f"  EXISTS")
    print(f"  title: {getattr(info, 'title', '?')}")
    print(f"  url:   {getattr(info, 'url', '?')}")
    print(f"  versionNumber: {getattr(info, 'currentVersionNumber', '?')}")
except Exception as e:
    msg = str(e)[:300]
    print(f"  ERR: {type(e).__name__}: {msg}")

# Method 2: search exact slug
print("\n=== Method 2: search by slug ===")
try:
    results = api.dataset_list(search="birdclef2026-exp029-r4-l1", user="maekeso")
    if results:
        for r in results:
            ref = getattr(r, "ref", "?")
            print(f"  found: {ref}")
    else:
        print(f"  NOT FOUND in search")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")

# Method 3: list all maekeso datasets and check
print("\n=== Method 3: list all maekeso datasets matching exp029 ===")
try:
    results = api.dataset_list(user="maekeso", search="exp029")
    for r in results[:20]:
        ref = getattr(r, "ref", "?")
        title = getattr(r, "title", "?")
        print(f"  {ref}: {title[:60]}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")

# Method 4: try download (will get 404 if not exists)
print("\n=== Method 4: dataset_download_files (probing) ===")
with tempfile.TemporaryDirectory() as td:
    try:
        api.dataset_download_files(slug, path=td, quiet=True)
        files = list(Path(td).rglob("*"))
        print(f"  EXISTS, downloaded files: {len(files)}")
        for f in files:
            if f.is_file():
                print(f"  - {f.name}  {f.stat().st_size/1e6:.1f}MB")
    except Exception as e:
        msg = str(e)[:300]
        if "404" in msg or "Not Found" in msg:
            print(f"  NOT FOUND (404)")
        else:
            print(f"  ERR: {type(e).__name__}: {msg}")
