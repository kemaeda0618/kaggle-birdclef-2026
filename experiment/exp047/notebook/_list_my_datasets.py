"""List user maekeso's all datasets to confirm AnuraSet status."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try paginated listing
all_datasets = []
for page in range(1, 10):
    try:
        ds = api.dataset_list(user="maekeso", page=page)
        ds_list = list(ds)
        if not ds_list: break
        all_datasets.extend(ds_list)
        print(f"Page {page}: {len(ds_list)} datasets")
    except Exception as e:
        print(f"  page {page} err: {str(e)[:100]}")
        break

print(f"\nTotal datasets found: {len(all_datasets)}")
print(f"\nAll refs:")
for d in all_datasets:
    print(f"  {d.ref}")

# Specifically check for anuraset
print(f"\n=== Anuraset matches ===")
anura_matches = [d for d in all_datasets if "anura" in d.ref.lower()]
print(f"  {len(anura_matches)} anuraset datasets")
for d in anura_matches:
    print(f"    {d.ref}")
