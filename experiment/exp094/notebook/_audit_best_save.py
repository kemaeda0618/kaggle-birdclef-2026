"""Find where ckpt_best_ns22 is actually saved."""
import json
from pathlib import Path

nb = json.load(open(Path(__file__).with_name("nb_train_ali.ipynb"), encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if "ckpt_best_ns22" in src and "torch.save" in src:
        lines = src.splitlines()
        for j, line in enumerate(lines):
            if "ckpt_best_ns22" in line:
                start = max(0, j - 5)
                end = min(len(lines), j + 10)
                print(f"=== Cell {i}, around line {j+1} ===")
                for ln in lines[start:end]:
                    print(f"  {ln}")
                print()
