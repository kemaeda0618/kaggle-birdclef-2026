"""exp029 R3 fold 0 の実 val_ns22 を Kaggle Dataset の history.json から確認。"""
import json, os, io, sys, tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp029-l1-single"

# List files
print(f"=== Files in {slug} ===")
files = api.dataset_list_files(slug).files
for f in files:
    print(f"  {f.name}: {f.total_bytes/1e6:.2f} MB")

# Download history files
import urllib.request
import shutil

key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
for f in files:
    if "history" not in f.name.lower() and "json" not in f.name.lower():
        continue
    print(f"\n=== Downloading {f.name} ===")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}/{f.name}"
    try:
        h = {"Authorization": f"Bearer {key}"} if key.startswith("KGAT_") else {}
        auth = None if key.startswith("KGAT_") else ("maekeso", key)
        req = urllib.request.Request(url, headers=h)
        if auth:
            import base64
            cred = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
            req.add_header("Authorization", f"Basic {cred}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        local = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook") / f.name
        local.write_bytes(data)
        print(f"  saved: {local}")
        if f.name.endswith(".json"):
            content = json.loads(data.decode("utf-8"))
            if isinstance(content, list):
                print(f"  Type: list with {len(content)} entries")
                if len(content) > 0:
                    print(f"  First: {content[0]}")
                if len(content) > 1:
                    print(f"  Last:  {content[-1]}")
                # Show val_ns22 trajectory
                vals = [e.get("val_ns22") for e in content if "val_ns22" in e]
                if vals:
                    print(f"  val_ns22 trajectory: {vals}")
                    print(f"  val_ns22 max: {max(vals):.4f}")
                    print(f"  val_ns22 final: {vals[-1]:.4f}")
            elif isinstance(content, dict):
                print(f"  Type: dict, keys = {list(content.keys())[:20]}")
                for k, v in list(content.items())[:15]:
                    print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ERR: {e}")
