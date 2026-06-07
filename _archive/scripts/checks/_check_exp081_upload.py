import json
from pathlib import Path

nb = json.loads(Path("experiment/exp081/notebook/nb_train_r1.ipynb").read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", f"?{i}")
    src = "".join(c.get("source", []))
    if "kaggle" in src.lower() and ("dataset" in src.lower() or "upload" in src.lower()) and len(src) > 200:
        print(f"\n========= cell {i} id={cid} (code, {len(src.splitlines())} lines) =========")
        print(src)
