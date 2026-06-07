"""Syntax check both train + infer NBs."""
import json, ast
for nb_path in [
    r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp041\notebook\nb_train_birdset_b1.ipynb",
    r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp041\notebook\nb_infer_birdset_b1.ipynb",
]:
    nb = json.loads(open(nb_path, encoding="utf-8").read())
    print(f"\n=== {nb_path.split(chr(92))[-1]} ({len(nb['cells'])} cells) ===")
    fail = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code": continue
        src = "".join(c.get("source", []))
        if not src.strip(): continue
        try:
            ast.parse(src)
            print(f"  OK [{i:02d}] {c.get('id','')}")
        except SyntaxError as e:
            print(f"  FAIL [{i:02d}] {c.get('id','')} {e.msg} line {e.lineno}: {e.text}")
            fail += 1
    print(f"  {fail} failures")
