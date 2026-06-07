"""Show fold_loop cell to find where to add filter."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "fold_loop":
        src = "".join(c["source"])
        # Find the read_csv section
        if "read_csv" in src:
            idx = src.find("read_csv")
            print(src[max(0,idx-300):idx+1500])
        break
