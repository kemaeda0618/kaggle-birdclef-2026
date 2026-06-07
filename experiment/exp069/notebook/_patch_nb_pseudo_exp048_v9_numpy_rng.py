"""v9 patch for exp069b: Replace torch.randperm with numpy for cleaner device-free RNG.

Error V8: RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
原因: set_default_device("cuda") で torch.randperm が cuda 期待、torch.Generator() は cpu

Fix: numpy.random.RandomState(42).permutation() に置換、torch RNG を完全に回避
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    {
        "old": "rs_perm = torch.randperm(n_lab, generator=torch.Generator().manual_seed(42)).cpu().numpy()\n",
        "new": "rs_perm = np.random.RandomState(42).permutation(n_lab)\n",
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
            print(f"  Cell {i}: rs_perm → np.random.RandomState fix applied")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

assert total == 1
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v9 patched: {NB_PATH}")
