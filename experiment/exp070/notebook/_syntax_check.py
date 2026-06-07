"""Verify Python syntax of NB cells.

Skip cells with Colab/shell-only ops (cell 1, 2, 12).
"""
import ast, json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(__file__).with_name("nb_train_m1.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

SKIP_CELLS = {1, 2, 10, 12}  # contain Colab/shell/kaggle-runtime-only deps
errs = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    if i in SKIP_CELLS:
        print(f"Cell {i:2d}: skipped (Colab/shell-specific)")
        continue
    src = "".join(c["source"])
    src_clean = "\n".join(l for l in src.splitlines() if not l.strip().startswith("!"))
    try:
        ast.parse(src_clean)
        print(f"Cell {i:2d}: OK ({len(src):6d} chars)")
    except SyntaxError as e:
        errs += 1
        print(f"Cell {i:2d}: SyntaxError line {e.lineno}: {e.msg}")
        print(f"  {e.text!r}")

print(f"\nTotal errors: {errs}")
sys.exit(0 if errs == 0 else 1)
