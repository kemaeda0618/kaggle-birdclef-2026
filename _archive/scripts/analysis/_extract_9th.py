"""Extract BC25 9th place writeup + comments — search for window related insights."""
import json
from pathlib import Path

fp = Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/c81ad71d-a265-48b2-a04a-c72ac27c1573/tool-results/mcp-kaggle-get_forum_topic-1780053778414.txt")
d = json.loads(fp.read_text(encoding="utf-8"))["forum_topic"]

print(f"=== {d['name']} ===")
print(f"votes={d['total_votes']}\n")
print("--- WRITEUP (first 5000 chars) ---")
intro = d["first_message"]["raw_markdown"]
print(intro[:5000])
print("\n---\n")

# Recursive walk
def walk(comments, out):
    for c in comments:
        out.append(c)
        if "replies" in c and c["replies"]:
            walk(c["replies"], out)
    return out

all_c = walk(d.get("comments", []), [])
real = [c for c in all_c if c.get("post_date") and c.get("raw_markdown")]
real.sort(key=lambda x: x["post_date"])

# Filter for window/chunk/duration related comments
keys = ["window", "chunk", "sliding", "duration", "5s", "10s", "20s", "5 sec", "10 sec", "20 sec",
        "second"]
print(f"=== Comments mentioning window/chunk/duration ({len(real)} total comments) ===")
for c in real:
    body = c["raw_markdown"]
    if any(k in body.lower() for k in keys):
        author = c.get("author", {}).get("display_name", "?")
        date = c["post_date"][:10]
        snippet = body.strip().replace("\n", " ")[:500]
        print(f"\n[{date}] {author}:\n{snippet}\n{'-'*50}")
