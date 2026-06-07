"""List cells in exp010 nb4 v11 and exp040 blend."""
import json
for p in [
    r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v11_g26.ipynb",
    r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp040\notebook\nb_blend.ipynb",
]:
    nb = json.load(open(p, encoding="utf-8"))
    print(f"\n===== {p} =====")
    print(f"cells: {len(nb['cells'])}")
    for i, c in enumerate(nb["cells"]):
        cid = c.get("id", "?")
        src = "".join(c.get("source", []))
        head = src.split("\n")[0][:80] if src else ""
        nlines = len(src.splitlines())
        print(f"  [{i}] {c.get('cell_type'):8s} id={cid:30s} | L={nlines:4d} | {head}")
