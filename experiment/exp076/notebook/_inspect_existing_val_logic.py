"""Inspect existing R3 NB's validation logic to reuse in M7 v2."""
import json, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = "experiment/exp029/notebook/nb_train_r3_l1_single.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

# Find cells with val_ns22 / val_macro / taxon
keywords = ["val_ns22", "val_macro", "TAXON_MASKS", "taxon_auc", "non_s22"]
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = "".join(c.get("source", []))
    if any(k in src for k in keywords):
        head = src.split("\n")[0][:80]
        print(f"=== Cell {i} ({len(src)} chars) | {head} ===")
        # Print matched lines + context
        lines = src.split("\n")
        for j, ln in enumerate(lines):
            if any(k in ln for k in keywords):
                # context: 2 lines before/after
                start = max(0, j - 1)
                end = min(len(lines), j + 3)
                for k in range(start, end):
                    print(f"  {k:4d}: {lines[k][:130]}")
                print()
        print("--- end ---")
        print()
