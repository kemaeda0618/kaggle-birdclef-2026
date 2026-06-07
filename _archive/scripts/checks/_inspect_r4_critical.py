"""Deep inspection of critical R4 NB cells."""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

R4_NB = Path("experiment/exp029/notebook/nb_train_r4_l1_single.ipynb")
nb = json.loads(R4_NB.read_text(encoding="utf-8"))

# Show critical cells in detail
critical_ids = ["dl_data", "config", "train_r2", "fold_loop", "upload"]
for cid in critical_ids:
    for c in nb["cells"]:
        if c.get("id") == cid:
            src = "".join(c["source"])
            print(f"\n{'='*70}\n# CELL {cid} ({len(src)} chars)\n{'='*70}")
            # Show first 80 lines
            for i, ln in enumerate(src.split("\n")[:80]):
                print(f"  L{i:3d}: {ln[:140]}")
            if src.count("\n") > 80:
                print(f"  ... ({src.count(chr(10)) - 80} more lines)")
            break
