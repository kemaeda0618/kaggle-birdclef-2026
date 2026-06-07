"""Find DRIVE_EXP_DIR assignment."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    for line in src.split("\n"):
        if "DRIVE_EXP_DIR" in line and "=" in line and "/" in line and "fold" not in line:
            print(f"[{c.get('id'):20s}] {line.strip()}")
        if "LOCAL_OUT" in line and "=" in line and "/" in line:
            print(f"[{c.get('id'):20s}] {line.strip()}")
        if "exp029" in line.lower() and "=" in line and ("DRIVE" in line or "LOCAL" in line):
            print(f"[{c.get('id'):20s}] {line.strip()}")
