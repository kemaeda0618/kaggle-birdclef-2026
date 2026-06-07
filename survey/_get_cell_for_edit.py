"""Extract specific cells from exp048 nb_blend_topn1.ipynb for editing context."""
import json, sys, io, os
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

p = r"experiment/exp048/notebook/nb_blend_topn1.ipynb"
nb = json.load(open(p, encoding="utf-8"))

# Find cells containing the Tucker SED inference and exp029 inference patterns
import re
patterns = {
    "tucker_sed_block": r"frame_max\s*=\s*outs\[1\]\.max\(axis=1\)",
    "exp029_block": r"_frame_max\s*=\s*_frame\.max\(dim=1\)\.values",
    "TTA_SHIFTS_cfg": r"TTA_SHIFTS\s*=",
}

for label, pat in patterns.items():
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if re.search(pat, src):
            print(f"=== {label} → Cell {i} (chars: {len(src)}) ===")
            print(src)
            print()
            break
