"""Inspect tucker-only NB."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_only.ipynb", encoding="utf-8"))
print(f"cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    head = src.split("\n")[0][:80] if src else ""
    n = len(src.splitlines())
    print(f"  [{i}] {c.get('cell_type'):8s} id={cid:30s} L={n:4d} | {head}")

# also show hdr content
for c in nb["cells"]:
    if c.get("id") == "hdr":
        print("\n=== hdr ===")
        print("".join(c["source"]))
