"""Parse Tucker SED notebook to extract key training info."""
import json
from pathlib import Path

p = Path(r"C:\Users\maeke\.claude\projects\C--Users-maeke-work-kaggle-birdclef-2026\10b3c073-565f-4aa3-897d-438b8fd4776a\tool-results\mcp-kaggle-get_notebook_info-1779582937669.txt")
data = json.loads(p.read_text(encoding="utf-8"))
print(f"keys: {list(data.keys())}")
print(f"metadata: {json.dumps(data.get('metadata', {}), indent=2)[:1000]}")
print()
if "source" in data:
    src = data["source"]
    print(f"Source length: {len(src)} chars")
    # Identify key sections
    keywords = ["distill", "Perch", "perch", "MIXUP", "mixup", "MIX", "BACKBONE", "lr", "epoch", "BCE",
                "augment", "DistillHead", "USE_PERCH", "data_loader", "Dataset"]
    for kw in keywords:
        cnt = src.count(kw)
        if cnt > 0:
            print(f"  [{kw}] {cnt} occurrences")
elif "cells" in data:
    cells = data["cells"]
    print(f"NB cells: {len(cells)}")
    for i, c in enumerate(cells):
        cid = c.get("id", "?")
        ctype = c.get("cell_type", "?")
        src = "".join(c.get("source", []))
        nlines = len(src.split("\n"))
        # Find first non-comment line
        head = ""
        for ln in src.split("\n"):
            s = ln.strip()
            if s and not s.startswith("#"):
                head = s[:60]; break
        print(f"  [{i:2d}] {ctype:8s} L={nlines:4d}  | {head}")
