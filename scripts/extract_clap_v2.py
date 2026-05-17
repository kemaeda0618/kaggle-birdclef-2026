"""Extract progressions from forum thread - look for date markers and LB scores."""
import json, re
from pathlib import Path

p = Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/9568b6dd-7c0e-4e43-a2e2-22f2a2fe3498/tool-results/mcp-kaggle-get_forum_topic-1778329365543.txt")
data = json.loads(p.read_text(encoding="utf-8"))
inner = json.loads(data[0]["text"])
topic = inner["forum_topic"]

# All comments (recursive)
all_comments = []
def collect(c, depth=0):
    all_comments.append((depth, c))
    for r in (c.get("replies") or []):
        collect(r, depth + 1)
for c in topic.get("comments") or []:
    collect(c)

print(f"Total comments collected: {len(all_comments)}")
print()

# Sort by date
all_comments.sort(key=lambda x: x[1].get("post_date", ""))

# Print every comment with date + author + first 600 chars
for depth, c in all_comments:
    md = c.get("raw_markdown", "")
    a = (c.get("author") or {}).get("display_name", "?")
    date = c.get("post_date", "")[:10]
    prefix = "  " * depth
    short = md[:600].replace("\n", " | ")
    print(f"{date} [{a}] {prefix}{short}")
    print()
