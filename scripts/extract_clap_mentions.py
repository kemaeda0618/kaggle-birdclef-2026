"""Extract all CLAP-related comments from forum thread."""
import json, re
from pathlib import Path

p = Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/9568b6dd-7c0e-4e43-a2e2-22f2a2fe3498/tool-results/mcp-kaggle-get_forum_topic-1778329365543.txt")
data = json.loads(p.read_text(encoding="utf-8"))
inner = json.loads(data[0]["text"])
topic = inner["forum_topic"]

print(f"Title: {topic.get('name')}")
print(f"Author: {topic.get('author_user_display_name')}")
print(f"Total messages: {topic.get('total_messages')}")
print()

# First message
fm = topic["first_message"]
fm_md = fm.get("raw_markdown", "")
print("=" * 70)
print(f"FIRST MESSAGE (by {fm['author']['display_name']}):")
print("=" * 70)
print(fm_md[:8000])
print()

# Comments
comments = topic.get("comments", []) or []
print(f"\n{len(comments)} comments")

# Find CLAP-mentioning comments
print("\n" + "=" * 70)
print("CLAP-MENTIONING COMMENTS")
print("=" * 70)

def walk(c, depth=0):
    md = c.get("raw_markdown", "")
    if "CLAP" in md or "clap" in md or ".935" in md or "0.93" in md or "AUC" in md.upper():
        a = c["author"]["display_name"]
        date = c.get("post_date", "")
        prefix = "  " * depth
        print(f"\n{prefix}--- {a} ({date}) ---")
        print(prefix + md[:3000].replace("\n", "\n" + prefix))
    for r in (c.get("replies") or []):
        walk(r, depth + 1)

for c in comments:
    walk(c)
