import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
kernels = api.kernels_list(user="maekeso", page_size=30, sort_by="dateRun")
print(f"{'Slug':<55} {'LastRun':<22}")
print("-" * 80)
for k in kernels[:25]:
    slug = k.ref
    lr = str(k.last_run_time)[:19] if k.last_run_time else "-"
    print(f"{slug:<55} {lr:<22}")
