"""Show full upload_state cell from exp032 R2 train NB."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "upload_state":
        print("".join(c["source"]))
        break
