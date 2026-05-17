"""Inspect BirdNET model files via SDK."""
import json, os, tempfile
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

REFS = [
    "dhruvpaidukle/birdnet-onnx",
    "adamfarrag/birdnet",
]

for ref in REFS:
    print(f"=== {ref} ===")
    # Try model_get
    try:
        m = api.model_get(ref)
        print(f"  title: {getattr(m, 'title', '?')}")
        print(f"  description: {str(getattr(m, 'description', ''))[:200]}")
        print(f"  versions: {getattr(m, 'versions', '?')}")
    except Exception as e:
        print(f"  model_get err: {type(e).__name__}: {str(e)[:120]}")
    # Variations
    try:
        v = api.model_list_variations(ref)
        for var in v.modelVariations[:5]:
            print(f"  variation: {var}")
    except Exception as e:
        print(f"  list_variations err: {type(e).__name__}")
    print()
