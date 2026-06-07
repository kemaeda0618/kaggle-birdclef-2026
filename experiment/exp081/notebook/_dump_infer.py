"""Dump exp081 inference NB cells for review."""
import json
from pathlib import Path

nb = json.load(open(Path(__file__).with_name("nb_infer.ipynb"), encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
print()
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    print(f"======== Cell {i} ({c['cell_type']}) ========")
    print(src)
    print()
