"""Extract cells from MCP-saved notebook JSON for analysis."""
import json
import sys
from pathlib import Path

f = Path(r"C:\Users\maeke\.claude\projects\C--Users-maeke-work-kaggle-birdclef-2026\ac229504-285e-44e1-a368-aa714279d269\tool-results\mcp-kaggle-get_notebook_info-1777198154363.txt")
out_dir = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\scripts\_pubnb_dump")
out_dir.mkdir(exist_ok=True)

# Outer wrapper is [{type:'text', text:'...json...'}]
outer = json.load(open(f, "r", encoding="utf-8"))
inner_str = outer[0]["text"]
inner = json.loads(inner_str)

# inner is the get_notebook_info response: {metadata, blob}
print("inner keys:", list(inner.keys()))
print("metadata title:", inner["metadata"].get("title"))
print("ds sources:", inner["metadata"].get("dataset_data_sources"))
print("model sources:", inner["metadata"].get("model_data_sources"))
print("comp sources:", inner["metadata"].get("competition_data_sources"))
print("version:", inner["metadata"].get("current_version_number"))

source_str = inner["blob"]["source"]
print(f"source length: {len(source_str)}")
nb = json.loads(source_str)

cells = nb["cells"]
print(f"# cells: {len(cells)}")

# Dump each cell
md_outline = []
for i, c in enumerate(cells):
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    p = out_dir / f"cell_{i:03d}_{c['cell_type']}.txt"
    p.write_text(src, encoding="utf-8")
    head = src.split("\n")[0][:120]
    md_outline.append(f"{i:03d} [{c['cell_type']:8}] {len(src):>6}c | {head}")

(out_dir / "_outline.txt").write_text("\n".join(md_outline), encoding="utf-8")
print("\n".join(md_outline))
