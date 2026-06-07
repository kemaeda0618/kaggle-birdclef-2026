"""Dump hdr, config, fold_loop, upload cells."""
import json, sys
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb", encoding="utf-8"))
target = sys.argv[1] if len(sys.argv) > 1 else "config"
for c in nb["cells"]:
    if c.get("id") == target:
        print("".join(c["source"]))
        break
