"""Inspect TA retrieval code in exp055 NB to understand structure for A3 patch."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp055\notebook\nb_blend_a3.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Find the cell containing TA retrieval logic
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if "compute_retrieval_ta_logit" in src or "_ta_meta" in src or "ta_pool_emb" in src and "load" in src.lower():
        print(f"\n{'='*60}\n[{c.get('id', '?')}] (len {len(src)} chars)\n{'='*60}")
        print(src[:5000])  # first 5000 chars
        if len(src) > 5000:
            print(f"\n... [truncated, total {len(src)} chars]")
