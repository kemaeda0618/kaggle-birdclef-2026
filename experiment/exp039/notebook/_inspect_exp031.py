"""Dump cell IDs / sizes / first lines of exp031 NB to plan extraction."""
import json
from pathlib import Path

EXP031_NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp031\notebook\nb_blend.ipynb")
nb = json.loads(EXP031_NB.read_text(encoding="utf-8"))
print(f"exp031 NB cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    cid = c.get("id", "")
    first = src.split("\n")[0][:120] if src else "(empty)"
    print(f"  [{i:02d}] {c['cell_type']:8s} id={cid:18s} {len(src):6d}c | {first}")
