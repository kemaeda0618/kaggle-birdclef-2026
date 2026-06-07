"""Dump sed-infer, e29-infer, blend cells from exp031 to identify external deps."""
import json
from pathlib import Path

EXP031_NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp031\notebook\nb_blend.ipynb")
nb = json.loads(EXP031_NB.read_text(encoding="utf-8"))

for cid in ["sed-infer", "e29-infer", "blend"]:
    c = next(x for x in nb["cells"] if x.get("id") == cid)
    src = "".join(c["source"])
    print(f"\n{'='*60}\n=== CELL {cid} ({len(src)}c) ===\n{'='*60}\n{src}")
