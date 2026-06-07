"""Extract TTA cells from kojimar 0.949 NB."""
import json
from pathlib import Path

p = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\kojimar_0-949-lb-birdclef-2026-prior-axis-rank-fusion\0-949-lb-birdclef-2026-prior-axis-rank-fusion.ipynb")
nb = json.load(open(p, encoding="utf-8"))
print(f"=== kojimar 0.949 NB ===")
print(f"  cells: {len(nb['cells'])}")
print()

# Search for TTA + key concepts
keywords = ["TTA", "tta_shifts", "temporal_shift", "np.roll", "sliding", "stride", "shift",
            "Prior", "rank", "fusion", "smooth"]
seen = set()
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c.get("cell_type") != "code": continue
    for kw in keywords:
        if kw.lower() in src.lower() and i not in seen:
            seen.add(i)
            print(f"--- Cell {i} (kw='{kw}') first 50 lines ---")
            for ln in src.split("\n")[:50]:
                print(f"  {ln}")
            print()
            break
