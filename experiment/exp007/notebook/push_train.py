"""Push exp007 train_and_pseudo.ipynb to Kaggle."""
import json, os, requests

TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")
if not TOKEN:
    # Read from kaggle.json
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.isfile(kaggle_json):
        with open(kaggle_json) as f:
            TOKEN = json.load(f)["key"]
if not TOKEN:
    raise RuntimeError("Set KAGGLE_API_TOKEN env var or configure ~/.kaggle/kaggle.json")

with open("C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp007/notebook/train_and_pseudo.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

nb_text = json.dumps(nb, ensure_ascii=False)

payload = {
    "newTitle": "birdclef2026-exp007-train",
    "text": nb_text,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "competition_data_sources": ["birdclef-2026"],
    "dataset_data_sources": [],
    "kernel_data_sources": [],
    "category_ids": ["audio"],
}

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

resp = requests.post(
    "https://www.kaggle.com/api/v1/kernels/push",
    headers=headers,
    json=payload,
    timeout=120,
)

print(f"Status: {resp.status_code}")
print(resp.text)
