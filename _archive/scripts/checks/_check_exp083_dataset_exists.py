import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try multiple slug variants
slugs = [
    "exp083-state",
    "birdclef2026-exp083-state",
    "birdclef2026-exp083-weights",
]
for slug in slugs:
    print(f"\n=== Try maekeso/{slug} ===")
    try:
        ds = api.dataset_view(f"maekeso/{slug}")
        print(f"  EXISTS: title={ds.title}, current_version={ds.current_version_number}, "
              f"total_files={ds.total_files if hasattr(ds, 'total_files') else '?'}")
    except Exception as e:
        print(f"  NOT FOUND or ERR: {type(e).__name__}: {str(e)[:200]}")

# Also list user's datasets
print(f"\n=== maekeso's datasets (last 10) ===")
try:
    dss = api.dataset_list(user="maekeso", sort_by="updated", max_size=10)
    for ds in dss[:10]:
        print(f"  {ds.ref}  (updated: {ds.last_updated})")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")
