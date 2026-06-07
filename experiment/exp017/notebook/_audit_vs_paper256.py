"""Audit exp017 config vs paper 256 spec."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp017\notebook\nb_train.ipynb", encoding="utf-8"))

# Cells of interest
TARGET_CELLS = ["config", "dataset", "train_r2", "train_r3", "pseudo_gen", "pseudo_save"]
for c in nb["cells"]:
    cid = c.get("id", "?")
    if cid not in TARGET_CELLS:
        continue
    src = "".join(c.get("source", []))
    print(f"\n{'='*78}")
    print(f"=== CELL: {cid} ===")
    print(f"{'='*78}")
    # Filter relevant lines
    for line in src.split("\n"):
        ls = line.strip()
        # Look for pseudo / mixup / sampling / threshold / filter related lines
        keywords = ["MIXUP", "mixup", "SHARES", "pseudo", "max_prob", "threshold",
                    "label_smooth", "PSEUDO", "secondary", "PROB", "ALPHA",
                    "filter", "min_conf", "trim_min", "primary_label_min"]
        if any(k in ls for k in keywords):
            print(f"  {line}")
