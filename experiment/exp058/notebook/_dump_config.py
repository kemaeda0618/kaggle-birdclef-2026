"""Dump config cell of exp048 to understand EMB_DIR setup."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "config":
        print("".join(c["source"]))
        break
