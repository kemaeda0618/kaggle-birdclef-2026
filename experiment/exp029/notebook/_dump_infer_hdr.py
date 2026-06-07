"""Dump infer NB header and lookup R1/R2/R3 val tables."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\nb_infer_r3_single.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "markdown" or "val" in src.lower()[:500]:
        print(f"=== cell {cid} ({c.get('cell_type')}) ===")
        print(src[:2000])
        print("...\n")
