import json
from pathlib import Path

nb = json.loads(Path("experiment/exp082/notebook/nb_train_r2.ipynb").read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", f"?{i}")
    src = "".join(c.get("source", []))
    ctype = c.get("cell_type", "?")
    print(f"\n========= cell {i} id={cid} ({ctype}, {len(src.splitlines())} lines) =========")
    # Print just first 80 lines per cell
    lines = src.splitlines()
    for ln in lines[:80]:
        print(f"  {ln[:150]}")
    if len(lines) > 80:
        print(f"  ... ({len(lines) - 80} more lines)")
