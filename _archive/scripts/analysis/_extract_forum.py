"""Extract topic info + recent comments from saved forum_topic JSON."""
import json
import sys
from pathlib import Path

files = [
    Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/c81ad71d-a265-48b2-a04a-c72ac27c1573/tool-results/mcp-kaggle-get_forum_topic-1779974094559.txt"),
    Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/c81ad71d-a265-48b2-a04a-c72ac27c1573/tool-results/mcp-kaggle-get_forum_topic-1779974096088.txt"),
]

def walk(comments, out):
    """Flatten nested comments with replies."""
    for c in comments:
        out.append(c)
        if "replies" in c and c["replies"]:
            walk(c["replies"], out)
    return out

for fp in files:
    d = json.loads(fp.read_text(encoding="utf-8"))["forum_topic"]
    print("=" * 70)
    print(f"#{d['id']} : {d['name']}")
    print(f"msgs={d['total_messages']}, votes={d['total_votes']}")
    print("--- INTRO ---")
    intro = d["first_message"]["raw_markdown"][:2000]
    print(intro)
    print()

    all_comments = walk(d.get("comments", []), [])
    # Filter only with date + content
    real = [c for c in all_comments if c.get("post_date") and c.get("raw_markdown")]
    real.sort(key=lambda x: x["post_date"], reverse=True)
    print(f"--- RECENT 15 COMMENTS (newest first, total={len(real)}) ---")
    for c in real[:15]:
        author = c.get("author", {}).get("display_name", "?")
        date = c["post_date"][:10]
        body = c["raw_markdown"].strip().replace("\n", " ")[:300]
        print(f"\n[{date}] {author}: {body}")
    print()
