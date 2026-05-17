"""Inspect BirdNET model assets on Kaggle."""
import json, os
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

REFS = [
    "dhruvpaidukle/birdnet-onnx",
    "adamfarrag/birdnet",
    "shadiakiki1/birdnetlib",
    "shadiakiki1/birdnet-analyzer",
]

for ref in REFS:
    print(f"=== {ref} ===")
    try:
        # Models
        instances = api.model_instance_list(ref).get("instances", [])
        for inst in instances[:3]:
            print(f"  instance: {inst}")
    except Exception as e:
        print(f"  model_instance_list err: {type(e).__name__}: {str(e)[:100]}")
    print()
