"""Dump exp078 cells 0, 26, 27, 28, 29 fully for inspection."""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
nb = json.load(open("experiment/exp078/notebook/nb_blend_v313_swap.ipynb", encoding="utf-8"))
target_ids = ["v313_00", "bridge", "sed-infer", "e29-infer", "blend"]
for c in nb["cells"]:
    if c.get("id") in target_ids:
        print(f"\n{'='*80}\n# CELL id={c.get('id')}\n{'='*80}")
        print("".join(c.get("source", [])))
