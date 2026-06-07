"""Dump config cell to find current CHUNK_FILTER block."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp046\notebook\nb_train_r3_l1_gamma12.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        # Find CHUNK_FILTER section
        idx = src.find("CHUNK_FILTER")
        if idx >= 0:
            print(src[max(0, idx-300):idx+400])
        break
