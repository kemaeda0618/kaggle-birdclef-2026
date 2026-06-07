import json, ast, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

for stream in ["nb4", "tucker", "exp029"]:
    nb_path = f"experiment/exp069/notebook/nb_pseudo_{stream}.ipynb"
    nb = json.load(open(nb_path, encoding="utf-8"))
    all_pass = True
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        cleaned = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
        try:
            compile(cleaned, f"<{stream}_cell{i}>", "exec")
        except SyntaxError as e:
            print(f"  {stream} Cell {i} SyntaxError: {e}")
            all_pass = False
    print(f"{stream}: syntax {'PASS' if all_pass else 'FAIL'}")
