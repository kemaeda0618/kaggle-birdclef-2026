"""Find the external pool loop variable name."""
import json
from pathlib import Path
NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp055\notebook\nb_blend_a3.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if "_ext_emb" in src and "_ext_lbl" in src:
        lines = src.split("\n")
        # Find context around "_ext_emb"
        for i, line in enumerate(lines):
            if "_ext_emb" in line and "_ext_lbl" in line:
                start = max(0, i - 15)
                end = min(len(lines), i + 10)
                for j in range(start, end):
                    print(f"{j:4d}|{lines[j]}")
                print("---")
                break
        break
