"""Dump original exp048 blend cell to find correct row_ids variable name."""
import json, re
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Find where submission.csv is built
        lines = src.split("\n")
        for i, ln in enumerate(lines):
            if "submission.csv" in ln or "row_id" in ln or "to_csv" in ln or "DataFrame" in ln:
                print(f"L{i}: {ln}")
        break
print()
print("=== last 30 lines of blend cell ===")
for ln in src.split("\n")[-30:]:
    print(ln)
