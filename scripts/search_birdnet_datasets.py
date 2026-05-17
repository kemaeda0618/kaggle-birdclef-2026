"""Search Kaggle for BirdNET model/dataset assets."""
import json, os
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Datasets matching 'birdnet' ===")
for d in api.dataset_list(search="birdnet", page=1, max_size=20):
    title = getattr(d, "title", "?")
    ref = getattr(d, "ref", "?")
    last = getattr(d, "last_updated", "?")
    print(f"  {ref:<60} | {title}")

print()
print("=== Datasets matching 'birdnet onnx' ===")
for d in api.dataset_list(search="birdnet onnx", page=1, max_size=10):
    title = getattr(d, "title", "?")
    ref = getattr(d, "ref", "?")
    print(f"  {ref:<60} | {title}")

print()
print("=== Models matching 'birdnet' ===")
try:
    for m in api.model_list(search="birdnet", page_size=10):
        ref = getattr(m, "ref", "?") or getattr(m, "id", "?")
        title = getattr(m, "title", "?")
        print(f"  {ref}  {title}")
except Exception as e:
    print(f"  (model_list failed: {type(e).__name__})")
