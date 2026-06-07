"""v6 patch for exp069b: train_labels も DEVICE 移動。

V5 で train_emb/train_logits/train_site/train_hour/train_prior の 5 個を DEVICE 移動したが
train_labels が漏れていた → BCE loss で再 device mismatch
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    {
        "old": 'train_labels = torch.tensor(lab_labels_files, dtype=torch.float32)\n',
        "new": 'train_labels = torch.tensor(lab_labels_files, dtype=torch.float32).to(DEVICE)\n',
    },
]

total = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    new_src = src
    for p in patches:
        if p["old"] in new_src:
            new_src = new_src.replace(p["old"], p["new"], 1)
            total += 1
            print(f"  Cell {i}: patched train_labels device")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

assert total == 1, f"Expected 1 patch, applied {total}"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v6 patched: {NB_PATH}")
