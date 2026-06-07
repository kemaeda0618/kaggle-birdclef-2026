"""Check POWER_GAMMA usage in exp017 R2, exp029 R3, and exp046 R3."""
import json
from pathlib import Path
import re

files = {
    "exp017 R1": Path("experiment/exp017/notebook/nb_train.ipynb"),
    "exp017 R2": Path("experiment/exp017/notebook/nb_train_r2.ipynb"),
    "exp029 R3 (gen script)": Path("experiment/exp029/notebook/_gen_nb_train_r3_l1_single.py"),
    "exp046 R3 γ=1.2": Path("experiment/exp046/notebook/nb_train_r3_l1_gamma12.ipynb"),
    "exp045 R3 filter": Path("experiment/exp045/notebook/nb_train_r3_l1_filtered.ipynb"),
    "exp029 R4": Path("experiment/exp029/notebook/nb_train_r4_l1_single.ipynb"),
}

KEYWORDS = ["POWER_GAMMA", "power_gamma", "gamma", "**GAMMA", "** GAMMA", "**gamma", "np.power",
            "torch.pow", "p**", "pseudo.*power", ".pow(", "** POWER", "** power",
            "POWER_TRANSFORM", "power_transform"]

for name, fp in files.items():
    if not fp.exists():
        print(f"\n=== {name}: NOT FOUND ===")
        continue
    print(f"\n=== {name} ===")
    if fp.suffix == ".ipynb":
        nb = json.loads(fp.read_text(encoding="utf-8"))
        all_src = "\n".join("".join(c.get("source", []))
                            for c in nb["cells"] if c["cell_type"] == "code")
    else:
        all_src = fp.read_text(encoding="utf-8")

    # Find lines with gamma-related keywords
    matches = []
    for line_no, line in enumerate(all_src.splitlines(), 1):
        line_lower = line.lower()
        if "gamma" in line_lower or "power_transform" in line_lower or "** power" in line_lower:
            if "pseudo" in line_lower or "gamma" in line_lower or "power_transform" in line_lower:
                matches.append((line_no, line.strip()))
    if matches:
        for ln, txt in matches[:20]:
            print(f"  L{ln}: {txt[:180]}")
    else:
        print("  (no gamma references found)")

    # Also find np.power or torch.pow with context
    print("  --- np.power / torch.pow usage ---")
    for ln, line in enumerate(all_src.splitlines(), 1):
        if "np.power" in line or "torch.pow" in line or ".pow(" in line:
            print(f"  L{ln}: {line.strip()[:180]}")
