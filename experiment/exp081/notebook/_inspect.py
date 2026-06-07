"""Inspect exp081 NB structure."""
import json
nb = json.load(open(r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_train_r2.ipynb", encoding="utf-8"))

# Print all cell IDs + first lines
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    cid = cell.get("id", "?")
    ctype = cell["cell_type"]
    first_line = src.splitlines()[0] if src.splitlines() else "(empty)"
    n_lines = len(src.splitlines())
    flag = ""
    for k in ["TRAIN_DURATION", "WINDOW", "VAL_DURATION", "SHARES", "pseudo_sc", "class FocalDS", "class LabeledSCDS", "class PseudoScDS", "PSEUDO_CSV"]:
        if k in src:
            flag += f" [{k}]"
    print(f"Cell {i:2d} [{ctype:8s}] id={cid:30s} L={n_lines:4d} | {first_line[:80]}{flag}")
