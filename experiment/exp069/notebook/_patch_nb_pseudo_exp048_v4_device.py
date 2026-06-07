"""v4 patch for exp069b: Fix device mismatch in ProtoSSM init_prototypes.

Error:
  model.init_prototypes(emb_flat, lab_flat)
  → x = self.input_proj(emb_flat)
  → RuntimeError: mat1 is on cpu, different from other tensors on cuda:0

Cause: emb_flat / lab_flat は CPU で作成、model は DEVICE=cuda 上にある (v3 で GPU 化済)

Fix: init_prototypes 呼び出し時に .to(DEVICE) を追加
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    {
        "old": '        model.init_prototypes(emb_flat, lab_flat)\n',
        "new": '        model.init_prototypes(emb_flat.to(DEVICE), lab_flat.to(DEVICE))\n',
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
            print(f"  Cell {i}: applied init_prototypes device fix")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

assert total == 1, f"Expected 1 patch, applied {total}"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v4 device fix applied: {NB_PATH}")
