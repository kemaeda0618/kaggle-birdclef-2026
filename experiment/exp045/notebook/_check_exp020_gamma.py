"""Check exp020 R1 and R2 NBs for Power Transform gamma usage."""
import json
from pathlib import Path

NBS = [
    ("exp020 R1 5-fold", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp020\notebook\nb_train_r1_5fold.ipynb"),
    ("exp020 R2 5-fold", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp020\notebook\nb_train_r2_5fold.ipynb"),
    ("exp020 5-fold TTA", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp020\notebook\nb_train_5fold_tta.ipynb"),
]

KEYWORDS = ["POWER_GAMMA", "power_gamma", "np.power", "GAMMA", "Power Transform", "power transform"]

for label, path in NBS:
    if not Path(path).exists():
        print(f"=== {label}: MISSING ===")
        continue
    nb = json.load(open(path, encoding="utf-8"))
    print(f"\n=== {label} ===")
    found_any = False
    for c in nb["cells"]:
        cid = c.get("id", "?")
        src = "".join(c.get("source", []))
        for kw in KEYWORDS:
            if kw.lower() in src.lower():
                # show lines containing this keyword
                for line in src.split("\n"):
                    if kw.lower() in line.lower():
                        ls = line.strip()
                        if not ls.startswith("#") or "POWER" in ls or "GAMMA" in ls:
                            print(f"  [{cid}] {ls[:130]}")
                            found_any = True
                break
    if not found_any:
        print("  (No Power Transform / gamma reference found)")
