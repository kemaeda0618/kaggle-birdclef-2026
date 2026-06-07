"""Check upload cell pattern used in exp032/exp020/exp017/exp029 train NBs."""
import json
from pathlib import Path

NBS = [
    (r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", "exp032_r2"),
    (r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train.ipynb", "exp032_r1"),
    (r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp020\notebook\nb_train_r2.ipynb", "exp020_r2"),
    (r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp017\notebook\nb_train.ipynb", "exp017"),
    (r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\nb_train_r3_l1_single.ipynb", "exp029_r3"),
]

for path, label in NBS:
    if not Path(path).exists():
        print(f"\n=== {label} === MISSING: {path}")
        continue
    nb = json.load(open(path, encoding="utf-8"))
    # find upload-related cells
    print(f"\n=== {label} ===")
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "dataset_create_new" in src or "dataset_create_version" in src or "dataset_initialize" in src:
            cid = c.get("id", "?")
            # Extract key lines
            for line in src.split("\n"):
                ls = line.strip()
                if ("dataset_create_new" in ls or "dataset_create_version" in ls or
                    "dataset_initialize" in ls or "kaggle.json" in ls or
                    "id\":" in ls or "title\":" in ls or
                    "version_notes" in ls or "DATASET_SLUG" in ls or
                    "DATASET_TITLE" in ls):
                    print(f"  [{cid}] {ls[:130]}")
