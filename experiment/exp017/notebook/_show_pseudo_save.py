"""Show exp017 pseudo_save cell to find Power Transform logic."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp017\notebook\nb_train.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "pseudo_save":
        print("".join(c["source"]))
        break
