"""Show diff between exp010 v11 and exp040 for shared cells (model, train, nb4-infer, blend)."""
import json
import difflib

v11 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v11_g26.ipynb", encoding="utf-8"))
e40 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp040\notebook\nb_blend.ipynb", encoding="utf-8"))

def get_cell(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"]).splitlines()
    return []

for cid in ("hdr", "config", "model", "train", "nb4-infer", "blend"):
    v = get_cell(v11, cid)
    e = get_cell(e40, cid)
    print(f"\n========= cell {cid} (v11 L={len(v)} | exp040 L={len(e)}) =========")
    diff = list(difflib.unified_diff(v, e, lineterm="", fromfile="v11", tofile="exp040", n=2))
    if not diff:
        print("  IDENTICAL")
        continue
    for line in diff[:250]:
        print(line)
    if len(diff) > 250:
        print(f"  ... ({len(diff)-250} more diff lines)")
