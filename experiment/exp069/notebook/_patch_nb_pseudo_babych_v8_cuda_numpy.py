"""v8 patch for exp069a: Babych model's get_head_preds() .numpy() needs .cpu() first.

Error V7: TypeError: can't convert cuda:0 device type tensor to numpy.
Cause: model.get_head_preds() の `framewise_prob = ...sigmoid().numpy()` line で
       sigmoid 後の tensor が cuda にあるが直接 numpy 変換不可

Fix: .sigmoid().numpy() → .sigmoid().cpu().numpy()
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    {
        "old": "        framewise_prob = head_output[\"framewise_logit\"].sigmoid().numpy()\n",
        "new": "        framewise_prob = head_output[\"framewise_logit\"].sigmoid().cpu().numpy()\n",
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
            print(f"  Cell {i}: cuda→cpu→numpy fix applied")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

assert total == 1, f"Expected 1 patch, applied {total}"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v8 patched: {NB_PATH}")
