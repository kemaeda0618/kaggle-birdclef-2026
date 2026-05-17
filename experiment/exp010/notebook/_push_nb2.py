"""Push exp010 NB2 (MLP Head submission) to Kaggle."""
import json, os, sys, io, requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Load token
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
with open(kaggle_json) as f:
    creds = json.load(f)
token = creds["key"]
owner = creds["username"]

# Load notebook
nb_path = os.path.join(os.path.dirname(__file__), "nb2_mlp.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb_text = f.read()

SLUG = "birdclef2026-exp010-nb2-mlp"
KERNEL_ID = 115856613  # NB2 v9 kernel ID (same kernel across versions)

payload = {
    "id": KERNEL_ID,
    "slug": SLUG,
    "newTitle": "birdclef2026-exp010-nb2-mlp",
    "text": nb_text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableInternet": False,
    "competitionDataSources": ["birdclef-2026"],
    "datasetDataSources": ["rishikeshjani/perch-onnx-for-birdclef-2026"],
    "kernelDataSources": ["maekeso/birdclef2026-exp010-nb1-embedding"],
}

resp = requests.post(
    "https://www.kaggle.com/api/v1/kernels/push",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json=payload,
)

print(f"Status: {resp.status_code}")
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
except Exception:
    print(resp.text)
