"""Check dhruvpaidukle/birdnet-onnx dataset."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Note: it's a Kaggle Model, not Dataset
print("=== dhruvpaidukle/birdnet-onnx (model_sources) ===")
try:
    # Try as model
    info = api.model_get("dhruvpaidukle/birdnet-onnx")
    print(f"  Model: {info}")
except Exception as e:
    print(f"  model_get failed: {e}")

# Also try dataset
print("\n=== as Dataset ===")
try:
    files = api.dataset_list_files("dhruvpaidukle/birdnet-onnx").files
    for f in files:
        print(f"  {f.name:40s} {f.total_bytes/1024/1024:.2f} MB")
except Exception as e:
    print(f"  Not a dataset: {e}")

# Search for "birdnet" models
print("\n=== Models matching 'birdnet' ===")
try:
    models = api.model_list(search="birdnet", max_size=10)
    for m in models:
        ref = getattr(m, "ref", "?")
        title = getattr(m, "title", "?")
        print(f"  {ref:60s} | {title}")
except Exception as e:
    print(f"  search failed: {e}")
