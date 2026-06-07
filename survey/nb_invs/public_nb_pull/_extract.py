"""Extract code cells from each downloaded ipynb to .code.txt for grep-friendly analysis."""
import json
from pathlib import Path

HERE = Path(__file__).parent
for nb_path in HERE.rglob("*.ipynb"):
    out_path = nb_path.with_suffix(".code.txt")
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ERR reading {nb_path.name}: {e}")
        continue
    parts = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            parts.append(f"# ====== Cell {ci} ======\n{src}")
    out_path.write_text("\n\n".join(parts), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"  {nb_path.name} -> {out_path.name} ({size_kb:.0f} KB, {len(parts)} cells)")

print("\nDone.")
