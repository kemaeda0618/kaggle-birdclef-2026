"""Check if Kaggle Dataset has v2 (bad R1 v2 mel-fix) or only v1."""
import json, os, requests
from pathlib import Path
creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
key = creds["key"]
headers = {"Authorization": f"Bearer {key}"} if key.startswith("KGAT_") else {}
auth = (creds["username"], key) if not key.startswith("KGAT_") else None

# Try downloading specific version 1 ckpt to check
print("=== Try downloading version 1 to verify ===")
url = "https://www.kaggle.com/api/v1/datasets/download/maekeso/birdclef2026-exp032-weights?datasetVersionNumber=1"
r = requests.head(url, headers=headers, auth=auth, timeout=30, allow_redirects=True)
print(f"v1 HEAD: status={r.status_code}, size hint={r.headers.get('Content-Length', '?')}")

# Try v2
url2 = "https://www.kaggle.com/api/v1/datasets/download/maekeso/birdclef2026-exp032-weights?datasetVersionNumber=2"
r2 = requests.head(url2, headers=headers, auth=auth, timeout=30, allow_redirects=True)
print(f"v2 HEAD: status={r2.status_code}, size hint={r2.headers.get('Content-Length', '?')}")

# Try v3
url3 = "https://www.kaggle.com/api/v1/datasets/download/maekeso/birdclef2026-exp032-weights?datasetVersionNumber=3"
r3 = requests.head(url3, headers=headers, auth=auth, timeout=30, allow_redirects=True)
print(f"v3 HEAD: status={r3.status_code}")

# Also try the dataset_view via SDK
print("\n=== SDK dataset info ===")
os.environ["KAGGLE_API_TOKEN"] = key if key.startswith("KGAT_") else ""
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
# list versions via dataset_status
try:
    status = api.dataset_status("maekeso/birdclef2026-exp032-weights")
    print(f"status: {status}")
except Exception as e:
    print(f"status err: {e}")
