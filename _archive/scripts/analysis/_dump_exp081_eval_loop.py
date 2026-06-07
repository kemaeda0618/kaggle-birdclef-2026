import json
from pathlib import Path

nb = json.loads(Path("experiment/exp081/notebook/nb_train_r1.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    cid = c.get("id", "?")
    if cid in ("eval", "train_loop"):
        src = "".join(c.get("source", []))
        print(f"\n\n========= cell id={cid} =========")
        print(src)
