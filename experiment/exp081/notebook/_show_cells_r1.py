"""Show specific cells of exp081 NB."""
import json, sys
nb = json.load(open(r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_train_r1.ipynb", encoding="utf-8"))

if len(sys.argv) < 2:
    print("usage: python _show_cells.py <cell_idx>")
    sys.exit(1)

idx = int(sys.argv[1])
cell = nb["cells"][idx]
src = "".join(cell.get("source", []))
print(f"=== Cell {idx} ({cell.get('id', '?')}) [{cell['cell_type']}] ===")
print(src)
