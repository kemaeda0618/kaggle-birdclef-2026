"""Inspect actual content of exp038 nb4-infer cell."""
import json
nb = json.loads(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp038\notebook\nb_blend.ipynb", encoding="utf-8").read())
for c in nb["cells"]:
    if c.get("id") == "nb4-infer":
        src = "".join(c["source"])
        print(f"=== nb4-infer ({len(src)}c) ===")
        # show the blend logic
        for i, line in enumerate(src.split("\n")):
            if 50 < i < 90 or "blend" in line.lower() or "l_proto" in line or "l_mlp" in line or "rs_" in line:
                print(f"  {i:3d}: {line}")
        break
