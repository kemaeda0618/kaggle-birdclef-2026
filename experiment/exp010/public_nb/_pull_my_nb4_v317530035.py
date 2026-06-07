"""Pull specific version of user's NB4 NB (scriptVersionId=317530035) for verification."""
import json, os, requests
from pathlib import Path

creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
key = creds["key"]

scriptVersionId = 317530035

target = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\my_nb4_v317530035\nb4_v317530035.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)

url = f"https://www.kaggle.com/kernels/scriptcontent/{scriptVersionId}/download"
headers = {"Authorization": f"Bearer {key}"} if key.startswith("KGAT_") else {}
auth = (creds["username"], key) if not key.startswith("KGAT_") else None

print(f"Downloading scriptVersionId={scriptVersionId}...")
r = requests.get(url, headers=headers, auth=auth, timeout=60, allow_redirects=True)
print(f"  status: {r.status_code}, size: {len(r.content)} bytes")
if r.status_code == 200 and len(r.content) > 1000:
    target.write_bytes(r.content)
    print(f"saved to {target.name}")
    nb = json.loads(r.content)
    print(f"  cells: {len(nb['cells'])}")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        first = src.split("\n")[0][:130] if src else "(empty)"
        print(f"  [{i:02d}] {c['cell_type']:8s} {len(src):5d}c | {first}")
else:
    print(f"FAIL: status={r.status_code}, first 200 chars: {r.content[:200].decode('utf-8', errors='replace')}")
