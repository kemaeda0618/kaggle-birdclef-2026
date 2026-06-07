"""Direct download of specific scriptVersionId via Kaggle web URL."""
import json, os, requests
from pathlib import Path

# Get Kaggle credentials
creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
user = creds["username"]
key = creds["key"]

scriptVersionId = 313904345

# Try the scriptcontent download endpoint (authenticated via basic auth)
urls_to_try = [
    f"https://www.kaggle.com/kernels/scriptcontent/{scriptVersionId}/download",
    f"https://www.kaggle.com/api/v1/kernels/scriptcontent/{scriptVersionId}/download",
]

# Determine auth header
if key.startswith("KGAT_"):
    headers = {"Authorization": f"Bearer {key}"}
    print(f"Using Bearer token (KGAT_)")
else:
    auth = (user, key)
    headers = {}

target_path = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_v313904345\0-932-v313904345.ipynb")
target_path.parent.mkdir(parents=True, exist_ok=True)

for url in urls_to_try:
    print(f"\nTrying: {url}")
    try:
        if key.startswith("KGAT_"):
            r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        else:
            r = requests.get(url, auth=auth, timeout=30, allow_redirects=True)
        print(f"  status: {r.status_code}, content-type: {r.headers.get('Content-Type', '?')}, size: {len(r.content)} bytes")
        if r.status_code == 200 and len(r.content) > 1000:
            target_path.write_bytes(r.content)
            print(f"OK saved {len(r.content)} bytes to {target_path}")
            # Try parsing as JSON
            try:
                nb = json.loads(r.content)
                cells = nb.get("cells", [])
                print(f"  parsed as ipynb: {len(cells)} cells")
            except Exception as e:
                # Check if it's HTML (login page) or actual content
                first200 = r.content[:200].decode("utf-8", errors="replace")
                print(f"  not ipynb, first 200 chars: {first200}")
            break
        elif r.status_code in (301, 302, 303, 307, 308):
            print(f"  redirect: {r.headers.get('Location')}")
    except Exception as e:
        print(f"  err: {type(e).__name__}: {str(e)[:200]}")
