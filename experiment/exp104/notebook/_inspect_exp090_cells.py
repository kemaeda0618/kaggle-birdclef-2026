"""Inspect all exp090 cells to plan the swap."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    first_line = next((ln for ln in src.splitlines() if ln.strip() and not ln.startswith("#")), "")
    title_line = next((ln for ln in src.splitlines() if ln.startswith("# ===")), "")
    print(f"{i:2d} id={c.get('id','?'):25s} chars={len(src):6d}  {title_line[:80] or first_line[:80]}")
