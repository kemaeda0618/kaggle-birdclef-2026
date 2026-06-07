"""Locate where teacher pseudo CSV is loaded so we can add chunk filter."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    if "pseudo" in src.lower() and ("read_csv" in src or "pseudo_e17" in src or "TEACHER_PSEUDO" in src):
        print(f"\n=== cell {cid} ===")
        # Show first 30 lines or relevant section
        lines = src.split("\n")
        for i, ln in enumerate(lines):
            if "pseudo" in ln.lower() or "csv" in ln.lower() or "filter" in ln.lower():
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                for j in range(start, end):
                    print(f"  L{j+1}: {lines[j]}")
                print("  ---")
                break
        # Print full snippet of relevant area
        if "TEACHER_PSEUDO" in src:
            idx = src.find("TEACHER_PSEUDO")
            print("\n=== TEACHER_PSEUDO context ===")
            print(src[max(0,idx-200):idx+1500])
