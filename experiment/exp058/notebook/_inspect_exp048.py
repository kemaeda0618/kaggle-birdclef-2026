"""Inspect exp048 NB structure for understanding integration points."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb", encoding="utf-8"))
print(f"Cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "?")
    ctype = c["cell_type"]
    src = "".join(c["source"])
    nlines = len(src.split("\n"))
    # show first non-empty non-comment line
    head = ""
    for ln in src.split("\n"):
        s = ln.strip()
        if s and not s.startswith("#"):
            head = s[:80]
            break
    if not head:
        head = src.split("\n")[0][:80] if src else ""
    print(f"  [{i:2d}] {ctype:8s} id={cid:30s} lines={nlines:4d}  | {head}")
