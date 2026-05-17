"""Search for Babych's code/notebooks/datasets."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Notebooks
print("=== Notebooks by nikitababich ===")
nbs = api.kernels_list(user="nikitababich", page_size=20, sort_by="dateRun")
for n in nbs:
    title = getattr(n, "title", "?")
    ref = getattr(n, "ref", "?")
    print(f"  {ref} — {title}")

print()
print("=== Datasets by nikitababich ===")
ds = api.dataset_list(user="nikitababich", page=1, max_size=20)
for d in ds:
    ref = getattr(d, "ref", "?")
    title = getattr(d, "title", "?")
    print(f"  {ref} — {title}")

print()
print("=== Search 'babych' across notebooks ===")
nbs = api.kernels_list(search="babych", page_size=10, sort_by="dateRun")
for n in nbs:
    title = getattr(n, "title", "?")
    ref = getattr(n, "ref", "?")
    print(f"  {ref} — {title}")

print()
print("=== Search 'multi-iterative' across notebooks ===")
nbs = api.kernels_list(search="multi-iterative", page_size=10, sort_by="dateRun")
for n in nbs:
    title = getattr(n, "title", "?")
    ref = getattr(n, "ref", "?")
    print(f"  {ref} — {title}")
