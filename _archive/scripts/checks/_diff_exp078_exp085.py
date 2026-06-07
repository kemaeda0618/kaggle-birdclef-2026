"""Full diff between exp078 NB and exp085 NB.
Verify only 4 intended changes:
1. hdr (markdown): updated text
2. v313_00 (Cell 0): added openvino wheel install
3. NEW e015-infer cell (exp015 OV inference)
4. blend cell: 3-way → 4-way
All other cells must be IDENTICAL.
"""
import json, io, sys, difflib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb78 = json.load(open("experiment/exp078/notebook/nb_blend_v313_swap.ipynb", encoding="utf-8"))
nb85 = json.load(open("experiment/exp085/notebook/nb_exp085_4model.ipynb", encoding="utf-8"))

# Build dict by cell id
def by_id(nb):
    return {c.get("id", f"__noid_{i}__"): c for i, c in enumerate(nb["cells"])}

d78 = by_id(nb78)
d85 = by_id(nb85)

print(f"exp078 cell count: {len(nb78['cells'])}")
print(f"exp085 cell count: {len(nb85['cells'])}")
print(f"exp078 ids: {set(d78.keys())}")
print(f"exp085 ids: {set(d85.keys())}")

# Cells only in exp085
only_in_85 = set(d85.keys()) - set(d78.keys())
only_in_78 = set(d78.keys()) - set(d85.keys())
print(f"\nOnly in exp085 (NEW): {only_in_85}")
print(f"Only in exp078 (REMOVED): {only_in_78}")

# Cells in both: check for diff
print("\n=== Per-cell diff ===")
identical = 0
diff_cells = []
for cid in sorted(set(d78.keys()) & set(d85.keys())):
    s78 = "".join(d78[cid].get("source", []))
    s85 = "".join(d85[cid].get("source", []))
    if s78 == s85:
        identical += 1
    else:
        diff_cells.append(cid)

print(f"Identical cells: {identical}/{len(d78)}")
print(f"Differing cells: {len(diff_cells)}")
for cid in diff_cells:
    s78 = "".join(d78[cid].get("source", []))
    s85 = "".join(d85[cid].get("source", []))
    print(f"\n--- DIFF in cell '{cid}' (78: {len(s78)} chars, 85: {len(s85)} chars) ---")
    diff = list(difflib.unified_diff(
        s78.splitlines(keepends=False),
        s85.splitlines(keepends=False),
        lineterm="", n=2, fromfile=f"exp078/{cid}", tofile=f"exp085/{cid}",
    ))
    # Show first 50 lines of diff
    for ln in diff[:80]:
        print(ln)
    if len(diff) > 80:
        print(f"  ... ({len(diff)-80} more diff lines)")

# Show NEW cells (only in 85)
print("\n=== NEW cells (only in exp085) ===")
for cid in only_in_85:
    s = "".join(d85[cid]["source"])
    print(f"\n--- NEW cell '{cid}' ({len(s)} chars) ---")
    print(s)
