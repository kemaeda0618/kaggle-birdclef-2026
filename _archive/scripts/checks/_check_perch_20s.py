import json
from pathlib import Path

nb = json.loads(Path("experiment/exp082/notebook/nb_train_r2.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "train_loop":
        src = "".join(c.get("source", []))
        # Find the Perch distill block
        lines = src.splitlines()
        in_block = False
        for i, ln in enumerate(lines):
            if "Perch distill" in ln or "perch_teacher" in ln:
                in_block = True
            if in_block:
                print(f"L{i:3d}: {ln}")
            if in_block and ("distill_loss = " in ln or "loss = cls_loss" in ln):
                # End of Perch block
                break
        break
