"""Push exp009 training notebook to Kaggle."""
import json, os, requests

# Load token
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
with open(kaggle_json) as f:
    creds = json.load(f)
token = creds.get("key") or os.environ.get("KAGGLE_API_TOKEN", "")

# Load notebook
nb_path = os.path.join(os.path.dirname(__file__), "train.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb_text = f.read()

payload = {
    "id": 115574455,
    "text": nb_text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableInternet": True,
    "machineShape": "NvidiaTeslaT4",
    "competitionDataSources": ["birdclef-2026"],
    "kernelDataSources": [
        "maekeso/birdclef2026-mel-cache-train-audio-256",
        "maekeso/birdclef2026-mel-cache-train-sc-256",
    ],
}

resp = requests.post(
    "https://www.kaggle.com/api/v1/kernels/push",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=payload,
)

print(f"Status: {resp.status_code}")
print(resp.json())
