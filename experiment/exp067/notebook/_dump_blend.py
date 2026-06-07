"""Dump exp048 blend cell structure to find insertion point."""
import json
from pathlib import Path
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        print(src)
        break
