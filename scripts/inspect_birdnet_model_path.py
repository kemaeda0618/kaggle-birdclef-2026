"""Try to find the right Kaggle Model variation path for BirdNET."""
import json, os
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try get_model + variations for several BirdNET refs
REFS = [
    "dhruvpaidukle/birdnet-onnx",
    "adamfarrag/birdnet",
]

for ref in REFS:
    print(f"=== {ref} ===")
    # try via REST: GET /v1/models/{owner}/{name}
    import requests
    token = os.environ["KAGGLE_API_TOKEN"]
    r = requests.get(f"https://www.kaggle.com/api/v1/models/{ref}/get",
                     headers={"Authorization": f"Bearer {token}"})
    if r.ok:
        data = r.json()
        print(json.dumps(data, indent=2)[:3000])
    else:
        print(f"  failed {r.status_code}: {r.text[:200]}")
    print()
