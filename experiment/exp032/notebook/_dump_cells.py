import json, sys
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))
ids = sys.argv[1:]
print(f"=== Cells ({len(nb['cells'])}) ===")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    cid = c.get("id", "")
    first = src.split("\n")[0][:100] if src else "(empty)"
    print(f"  [{i:02d}] {c['cell_type']:8s} id={cid:30s} {len(src):6d}c | {first}")
if ids:
    for x in ids:
        i = int(x)
        src = "".join(nb["cells"][i].get("source", []))
        print(f"\n=== CELL {i} FULL ===\n{src}\n")
