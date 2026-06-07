"""Print specific cell sources for exp082 nb_train_r2 (load_data, upload_state, manual upload)."""
import json
from pathlib import Path

nb = json.loads(Path("experiment/exp082/notebook/nb_train_r2.ipynb").read_text(encoding="utf-8"))
import sys
target_ids = sys.argv[1:] if len(sys.argv) > 1 else []
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", f"?{i}")
    if not target_ids or cid in target_ids:
        src = "".join(c.get("source", []))
        print(f"\n========= cell {i} id={cid} =========")
        print(src)
