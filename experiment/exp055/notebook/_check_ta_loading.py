"""Check exact TA loading cell content."""
import json
from pathlib import Path
NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp055\notebook\nb_blend_a3.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if "ta_pool_emb_norm = (_ta_emb_raw / _ta_norm)" in src and "del _ta_emb_raw" in src:
        print(f"=== Cell {c.get('id', '?')} ===")
        # Print only TA loading section
        lines = src.split("\n")
        start = None
        for i, line in enumerate(lines):
            if "_ta_meta = pd.read_parquet" in line:
                start = i
            if start is not None and "del _ta_emb_raw" in line:
                end = i + 1
                break
        if start is not None:
            for i in range(max(0, start-2), end+2):
                print(f"{i:4d}|{lines[i]}")
        break
