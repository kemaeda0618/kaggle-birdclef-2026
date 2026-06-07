"""Extract Tucker's training code cells - focus on what makes 0.94+ work."""
import json
from pathlib import Path

p = Path(r"C:\Users\maeke\.claude\projects\C--Users-maeke-work-kaggle-birdclef-2026\10b3c073-565f-4aa3-897d-438b8fd4776a\tool-results\mcp-kaggle-get_notebook_info-1779582937669.txt")
data = json.loads(p.read_text(encoding="utf-8"))
source_str = data["blob"]["source"]
nb = json.loads(source_str)
print(f"cells: {len(nb['cells'])}")
print()
# Save each cell to inspect
out_dir = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\survey\nb_invs\tucker_sed_cells")
out_dir.mkdir(exist_ok=True, parents=True)
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", f"c{i}")
    src = "".join(c.get("source", []))
    nlines = len(src.split("\n"))
    head = ""
    for ln in src.split("\n"):
        s = ln.strip()
        if s and not s.startswith("#"):
            head = s[:70]; break
    print(f"  [{i:2d}] {c['cell_type']:8s} L={nlines:4d}  | {head}")
    out_file = out_dir / f"cell_{i:02d}_{c['cell_type']}.txt"
    out_file.write_text(src, encoding="utf-8")
print(f"\nCells saved to {out_dir}")
