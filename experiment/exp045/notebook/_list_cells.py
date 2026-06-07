"""List exp045 NB cells with first line."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb", encoding="utf-8"))
print(f"cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    head = src.split("\n")[0][:80]
    n = len(src.splitlines())
    print(f"  [{i}] {c.get('cell_type'):8s} id={cid:20s} L={n:4d} | {head}")
