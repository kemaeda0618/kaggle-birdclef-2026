"""Show full model cell to know what to edit."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_infer.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "model":
        print("".join(c["source"]))
        break
