"""Search BC2026 discussions for non-Perch high-score reports."""
import json, os
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try various discussion list endpoints
COMP = "birdclef-2026"

# Method 1: list_forum / list_topics — see what's available
try:
    forums = api.list_forums()  # may not exist as method
    print("forums:", forums)
except Exception as e:
    print(f"list_forums: {e}")

# Try kernels-style listing for "forum"
import requests
# Direct API call
headers = {"Authorization": f"Bearer {os.environ['KAGGLE_API_TOKEN']}"}
url = f"https://www.kaggle.com/api/v1/competitions/{COMP}/forum"
try:
    r = requests.get(url, headers=headers, timeout=20)
    print(f"GET {url}: {r.status_code}")
    print(r.text[:500])
except Exception as e:
    print(f"forum: {e}")

# Try forum topics list - using competition forum endpoint
for path in [
    "/competitions/list",
    f"/competitions/{COMP}/discussion",
    f"/discussions/competitions/{COMP}",
]:
    url = f"https://www.kaggle.com/api/v1{path}"
    try:
        r = requests.get(url, headers=headers, timeout=20)
        print(f"  GET {path}: {r.status_code}, body[:200]={r.text[:200]}")
    except Exception as e:
        print(f"  {path}: {e}")
