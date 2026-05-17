"""Extract discussion topic + comments from MCP-saved file for analysis."""
import json
from pathlib import Path

f = Path(r"C:\Users\maeke\.claude\projects\C--Users-maeke-work-kaggle-birdclef-2026\ac229504-285e-44e1-a368-aa714279d269\tool-results\mcp-kaggle-get_forum_topic-1777199964184.txt")
out = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\scripts\_discussion_dump")
out.mkdir(exist_ok=True)

outer = json.load(open(f, "r", encoding="utf-8"))
inner = json.loads(outer[0]["text"])

ft = inner["forum_topic"]
title = ft.get("name") or ft.get("title")
url = ft.get("url")

# Build a flat thread file with all messages
def msg_block(m, depth=0):
    a = m.get("author", {}) or {}
    name = a.get("display_name") or m.get("author_user_display_name") or "unknown"
    tier = a.get("tier", "")
    rank = m.get("competition_ranking", "")
    votes = (m.get("votes") or {}).get("total_upvotes", "")
    date = m.get("post_date", "")[:10]
    indent = "  " * depth
    head = f"{indent}--- {name} [{tier}] rank={rank} votes={votes} date={date}"
    body = m.get("raw_markdown", "")
    body_indented = "\n".join(indent + line for line in body.split("\n"))
    return f"{head}\n{body_indented}\n"

lines = []
lines.append(f"# {title}")
lines.append(f"URL: https://www.kaggle.com{url}")
lines.append(f"Total messages: {ft.get('total_messages')}")
lines.append("=" * 80)

# First message
fm = ft.get("first_message")
if fm:
    lines.append("\n## ORIGINAL POST\n")
    lines.append(msg_block(fm, depth=0))

# Comments + nested replies (recursive)
def walk(m, depth):
    out_blocks = [msg_block(m, depth)]
    replies = m.get("replies") or []
    for r in replies:
        out_blocks.extend(walk(r, depth + 1))
    return out_blocks

lines.append("\n## COMMENTS\n")
for c in ft.get("comments", []):
    for blk in walk(c, depth=0):
        lines.append(blk)

text = "\n".join(lines)
(out / "thread.md").write_text(text, encoding="utf-8")

print(f"Written {len(text)} chars to thread.md")
print(f"Title: {title}")
print(f"Total messages: {ft.get('total_messages')}")
