import json
from pathlib import Path

nb = json.loads(Path("experiment/exp082/notebook/nb_train_r2.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "dataset":
        src = "".join(c.get("source", []))
        lines = src.splitlines()
        # Find LabeledSCDS class
        in_block = False
        depth = 0
        for i, ln in enumerate(lines):
            if "class LabeledSCDS" in ln:
                in_block = True
                depth = 0
            if in_block:
                print(f"L{i:3d}: {ln}")
                if "class " in ln and "LabeledSCDS" not in ln and "PseudoScDS" not in ln:
                    break
                if "class PseudoScDS" in ln:
                    print(f"L{i:3d}: ----- (PseudoScDS class start, continuing) -----")
        # Also find PseudoScDS for 20s
        for i, ln in enumerate(lines):
            if "class PseudoScDS" in ln:
                # Print next 40 lines
                end = min(i + 50, len(lines))
                for j in range(i, end):
                    print(f"L{j:3d}: {lines[j]}")
                break
