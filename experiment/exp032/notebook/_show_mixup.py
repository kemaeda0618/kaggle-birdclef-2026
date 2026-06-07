"""Show full MixUp implementation in train NBs."""
import json
for path in [
    r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb",
    r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\nb_train_r3_l1_single.ipynb",
]:
    nb = json.load(open(path, encoding="utf-8"))
    print(f"\n===== {path.split(chr(92))[-1]} =====")
    for c in nb["cells"]:
        if c.get("id") == "dataset":
            src = "".join(c["source"])
            # Find mixup block
            lines = src.split("\n")
            for i, ln in enumerate(lines):
                if "MIXUP" in ln or "mixup" in ln.lower() or "lam" in ln:
                    # Print context
                    for j in range(max(0, i-1), min(len(lines), i+3)):
                        print(f"  {lines[j]}")
                    print()
                    break
            # Print 30 lines around the first MIXUP
            for i, ln in enumerate(lines):
                if "USE_MIXUP" in ln and "self.aug" in ln:
                    print("--- Full MixUp block ---")
                    for j in range(i, min(len(lines), i+20)):
                        print(f"  {lines[j]}")
                    break
