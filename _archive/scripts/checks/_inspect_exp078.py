import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
nb = json.load(open("experiment/exp078/notebook/nb_blend_v313_swap.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    label = ""
    for ln in src.split("\n")[:5]:
        s = ln.strip()
        if s and not s.startswith("# ===") and not s.startswith("```"):
            label = s[:100]; break
    cid = c.get("id", "?")
    print(f"{i:2d} [{c['cell_type'][:4]}] id={cid:30s} | {label}")
