"""Show the actual diff for model + dataset cells."""
import json
import difflib

r1 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train.ipynb", encoding="utf-8"))
r2 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))

def get_cell(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"]).splitlines()
    return []

for cid in ("model", "dataset", "train_loop"):
    r1_c = get_cell(r1, cid)
    r2_c = get_cell(r2, cid)
    print(f"\n========= cell: {cid} (R1 {len(r1_c)} lines | R2 {len(r2_c)} lines) =========")
    diff = list(difflib.unified_diff(r1_c, r2_c, lineterm="", fromfile="R1", tofile="R2", n=2))
    if not diff:
        print("  IDENTICAL")
        continue
    for line in diff[:120]:
        print(line)
    if len(diff) > 120:
        print(f"  ... ({len(diff)-120} more diff lines)")
