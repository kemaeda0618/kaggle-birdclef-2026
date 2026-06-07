"""Check where probs_blend is defined vs where preds_array uses it."""
import json
from pathlib import Path
NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp056\notebook\nb_blend_aves_stage2a.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    has_blend_def = "probs_blend = blend_flat.reshape" in src
    has_preds_array = "preds_array = probs_blend.reshape" in src
    if has_blend_def or has_preds_array:
        print(f"\n=== Cell {i} (id={c.get('id', '?')}, len={len(src)}) ===")
        print(f"  has probs_blend = ...: {has_blend_def}")
        print(f"  has preds_array = ...: {has_preds_array}")
