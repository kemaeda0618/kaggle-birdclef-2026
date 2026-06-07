"""Fetch exp106 standalone log via REST."""
import json, os, requests, base64
from pathlib import Path

kj = Path.home() / ".kaggle" / "kaggle.json"
creds = json.loads(kj.read_text())
token = creds["key"]
url = "https://www.kaggle.com/api/v1/kernels/output?userName=maekeso&kernelSlug=birdclef2026-exp106-standalone-3fold"
if token.startswith("KGAT_"):
    headers = {"Authorization": f"Bearer {token}"}
else:
    basic = base64.b64encode(f"{creds['username']}:{token}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}"}

r = requests.get(url, headers=headers)
print("status:", r.status_code)
if r.ok:
    j = r.json()
    print("keys:", list(j.keys()))
    if "log" in j:
        print("\n---- LOG (last 6000) ----")
        print(j["log"][-6000:])
    if "files" in j:
        print("\n---- Files ----")
        for f in j["files"]:
            print(f"  {f}")
else:
    print(r.text[:500])
