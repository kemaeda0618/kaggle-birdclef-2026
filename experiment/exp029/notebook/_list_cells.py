"""List cells of nb_train_r3_l1_single.ipynb."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\nb_train_r3_l1_single.ipynb", encoding="utf-8"))
print(f"cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    ctype = c.get("cell_type", "?")
    head = src.split("\n")[0][:100] if src else ""
    print(f"  [{i}] {ctype:8s} id={cid:25s} | {head}")
