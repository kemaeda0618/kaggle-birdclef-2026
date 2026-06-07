"""Find save logic in training NB."""
import json
from pathlib import Path

nb = json.load(open(Path(__file__).with_name("nb_train_ali.ipynb"), encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if "torch.save" in src or "model_state" in src:
        # Find the save context
        for j, line in enumerate(src.splitlines()):
            if "torch.save" in line or "model_state" in line or "ckpt_best" in line:
                start = max(0, j - 3)
                end = min(len(src.splitlines()), j + 8)
                print(f"=== Cell {i}, around line {j} ===")
                for ln in src.splitlines()[start:end]:
                    print(f"  {ln}")
                print()
                break
