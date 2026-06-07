"""Dump the blend + e29-infer cells of exp048 to understand prediction format."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    cid = c.get("id", "?")
    if cid in ("e29-infer", "blend"):
        print(f"=== {cid} ===")
        print("".join(c["source"]))
        print()
