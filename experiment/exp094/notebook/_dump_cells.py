"""Dump all cells of inference NB."""
import json
from pathlib import Path

nb = json.load(open(Path(__file__).with_name("nb_infer_r1_ali.ipynb"), encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    print(f"======== Cell {i} ({c['cell_type']}) ========")
    print(src)
    print()
