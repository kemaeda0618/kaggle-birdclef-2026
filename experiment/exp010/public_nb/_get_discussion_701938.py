"""Fetch BC2026 discussion 701938 (comment 3461621)."""
import json, os, requests
from pathlib import Path

creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
key = creds["key"]

# Try Kaggle API for forum topic
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# 1. Try via SDK
try:
    print("=== Kaggle SDK forum topic ===")
    res = api.get_forum_topic(topic_id=701938) if hasattr(api, 'get_forum_topic') else None
    if res:
        print(f"  type: {type(res).__name__}")
        for attr in dir(res):
            if not attr.startswith("_"):
                val = getattr(res, attr, None)
                if val and not callable(val):
                    print(f"  {attr}: {str(val)[:300]}")
except Exception as e:
    print(f"  SDK approach failed: {type(e).__name__}: {str(e)[:200]}")

# 2. Direct web fetch (markdown)
print("\n=== Direct web fetch ===")
url = "https://www.kaggle.com/competitions/birdclef-2026/discussion/701938.json"
headers = {"Authorization": f"Bearer {key}"} if key.startswith("KGAT_") else {}
auth = (creds["username"], key) if not key.startswith("KGAT_") else None

for url_try in [
    f"https://www.kaggle.com/discussions/general/701938.json",
    f"https://www.kaggle.com/api/v1/forums/topics/701938",
    f"https://www.kaggle.com/competitions/birdclef-2026/discussion/701938",
]:
    try:
        r = requests.get(url_try, headers=headers, auth=auth, timeout=30, allow_redirects=True)
        print(f"  {url_try[:80]}... → {r.status_code}, {len(r.content)} bytes")
        ct = r.headers.get("Content-Type", "")
        if r.status_code == 200 and ("json" in ct or r.content.lstrip().startswith(b"{")):
            try:
                data = r.json()
                # find relevant content
                print(f"  JSON keys: {list(data.keys())[:20] if isinstance(data, dict) else type(data)}")
                # Save for inspection
                out = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\discussion_701938.json")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  saved: {out}")
                break
            except Exception as e:
                pass
        elif r.status_code == 200:
            # Try HTML extract — look for discussion content
            txt = r.content.decode("utf-8", errors="replace")
            if "discussion" in txt.lower() or "topic" in txt.lower():
                # Save HTML for grep
                out = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\discussion_701938.html")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(txt, encoding="utf-8")
                print(f"  saved HTML: {out}")
                # extract from script tags
                import re
                m = re.search(r'window\["pageRequestConfig"\]\s*=\s*({.+?});\s*\n', txt, re.DOTALL)
                if m:
                    try:
                        cfg = json.loads(m.group(1))
                        print(f"  pageRequestConfig keys: {list(cfg.keys())[:10]}")
                    except Exception:
                        pass
                break
    except Exception as e:
        print(f"  err: {str(e)[:100]}")
