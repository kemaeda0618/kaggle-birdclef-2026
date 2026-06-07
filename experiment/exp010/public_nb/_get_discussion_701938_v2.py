"""Fetch discussion 701938 via Kaggle internal API endpoints."""
import json, os, requests
from pathlib import Path

creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
key = creds["key"]
headers = {"Authorization": f"Bearer {key}"} if key.startswith("KGAT_") else {}
auth = (creds["username"], key) if not key.startswith("KGAT_") else None

# Try multiple known Kaggle internal endpoints
endpoints = [
    f"https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopic?forumTopicId=701938",
    f"https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicComments?forumTopicId=701938",
    f"https://www.kaggle.com/discussions/forum-topic-data?forumTopicId=701938",
    f"https://www.kaggle.com/discussions/forum-topic-comments?forumTopicId=701938",
]

for url in endpoints:
    print(f"\n=== {url[:90]}... ===")
    for method in ["POST", "GET"]:
        try:
            if method == "POST":
                r = requests.post(url, headers={**headers, "Content-Type": "application/json", "x-xsrf-token": "n/a"},
                                  json={"forumTopicId": 701938}, auth=auth, timeout=30)
            else:
                r = requests.get(url, headers=headers, auth=auth, timeout=30)
            print(f"  {method}: {r.status_code}, {len(r.content)} bytes, {r.headers.get('Content-Type', '?')[:50]}")
            if r.status_code == 200 and len(r.content) > 200:
                try:
                    data = r.json()
                    print(f"  JSON top keys: {list(data.keys())[:15] if isinstance(data, dict) else type(data).__name__}")
                    out = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\discussion_701938.json")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"  saved: {out}")
                    break
                except Exception as e:
                    print(f"  not json: {str(e)[:80]}")
                    print(f"  first 200: {r.content[:200].decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"  {method} err: {type(e).__name__}: {str(e)[:100]}")
