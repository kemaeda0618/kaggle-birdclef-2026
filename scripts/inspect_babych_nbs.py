"""Inspect 3 Babych-related NBs for pseudo-labeling code."""
import json
from pathlib import Path

NBS = [
    ("Babych 2025 inference (本人)",
     "experiment/exp010/notebook/_reference_babych/birdclef2025-1st-place-inference/birdclef2025-1st-place-inference.ipynb"),
    ("kosuke BC2026 fork",
     "experiment/exp010/notebook/_reference_babych/birdclef-2026-babych-2025-fork/birdclef-2026-babych-2025-fork.ipynb"),
    ("emanuellcs BC2026 target-domain pseudo",
     "experiment/exp010/notebook/_reference_babych/birdclef-2026-target-domain-pseudo-labeling/birdclef-2026-target-domain-pseudo-labeling.ipynb"),
]

for label, path in NBS:
    print("=" * 80)
    print(f"NB: {label}")
    print(f"path: {path}")
    print("=" * 80)
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"Total cells: {len(cells)}")
    print()
    print("Cell summaries:")
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        ct = c["cell_type"][:4]
        first = src.split("\n")[0][:140]
        print(f"  [{i:3d} {ct}] {first}")
    print()
