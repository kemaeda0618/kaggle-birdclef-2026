import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
nb = json.load(open("experiment/convert_to_ov/notebook/convert_to_ov.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    label = "?"
    for ln in src.split("\n")[:6]:
        s = ln.strip()
        if s and not s.startswith("# ===") and not s.startswith("```"):
            label = s[:90]
            break
    print(f"{i:2d} [{c['cell_type'][:4]}] {label}")
