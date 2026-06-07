"""Extract full TTA function from kojimar 0.949."""
import json
from pathlib import Path

p = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\kojimar_0-949-lb-birdclef-2026-prior-axis-rank-fusion\0-949-lb-birdclef-2026-prior-axis-rank-fusion.ipynb")
nb = json.load(open(p, encoding="utf-8"))

# Find cell containing np.roll
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "np.roll" in src:
        print(f"=== Cell {i} full content ===")
        print(src)
        break
