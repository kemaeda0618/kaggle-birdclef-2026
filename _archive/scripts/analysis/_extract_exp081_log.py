import json
from pathlib import Path

nb = json.loads(Path("experiment/exp081/notebook/nb_train_r1.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    # Look for train_loop cell (has training output prints)
    if ("for epoch in range" in src or "epoch_times" in src or "trained_this_session" in src or
        "val_per_taxon" in src or "class_stats" in src):
        print(f"--- cell id={c.get('id', '?')} ({len(src.splitlines())} lines) ---")
        # Print just the print statements + key log lines
        for ln in src.splitlines():
            lns = ln.strip()
            if ("print(" in ln or "BEST" in ln or "val_per_taxon" in ln or
                "class_stats" in ln or "median" in ln or "perfect" in ln):
                print(f"  {ln[:200]}")
        print()
