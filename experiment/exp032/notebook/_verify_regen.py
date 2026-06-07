import json, ast
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_regen_r1_pseudo.ipynb", encoding="utf-8"))
print(f"cells: {len(nb['cells'])}")
fail = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if not src.strip(): continue
    try:
        ast.parse(src)
        print(f"  OK [{i:02d}] {c.get('id','')}")
    except SyntaxError as e:
        print(f"  FAIL [{i:02d}] {c.get('id','')}: {e.msg} line {e.lineno}: {e.text}")
        fail += 1
print(f"{fail} failures")
