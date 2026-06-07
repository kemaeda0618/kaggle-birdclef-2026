"""Extract Babych BC25 1st place writeup body + key comments mentioning post-process / pp."""
import json
from pathlib import Path
import re

fp = Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/c81ad71d-a265-48b2-a04a-c72ac27c1573/tool-results/mcp-kaggle-get_forum_topic-1779974838054.txt")

d = json.loads(fp.read_text(encoding="utf-8"))["forum_topic"]
print(f"=== {d['name']} ===")
print(f"votes={d['total_votes']}, msgs={d['total_messages']}")
print()

# === Full writeup body ===
intro = d["first_message"]["raw_markdown"]
print("=== FULL WRITEUP BODY ===")
print(intro)
print()
print(f"=== END WRITEUP ({len(intro)} chars) ===")
print()

# === All comments flattened, sorted by date, filtered to pp-relevant ===
def walk(comments, out):
    for c in comments:
        out.append(c)
        if "replies" in c and c["replies"]:
            walk(c["replies"], out)
    return out

all_c = walk(d.get("comments", []), [])
real = [c for c in all_c if c.get("post_date") and c.get("raw_markdown")]
real.sort(key=lambda x: x["post_date"])

# Filter to PP / post-process / smoothing / TTA / prior keyword
pp_keys = ["post-process", "postprocess", "post process", "smoothing", "TTA", "prior",
           "delta", "lambda", "weight", "filter", "threshold", "γ", "gamma",
           "frame_max", "Power Transform", "rank", "Babych", "pp ", " pp", "site"]
pp_comments = [c for c in real
               if any(k.lower() in c["raw_markdown"].lower() for k in pp_keys)]

print(f"=== PP-RELEVANT COMMENTS ({len(pp_comments)} of {len(real)}) ===\n")
for c in pp_comments:
    author = c.get("author", {}).get("display_name", "?")
    date = c["post_date"][:10]
    body = c["raw_markdown"]
    if len(body) > 800:
        body = body[:800] + "..."
    print(f"[{date}] {author}:\n{body}\n{'-'*60}")
