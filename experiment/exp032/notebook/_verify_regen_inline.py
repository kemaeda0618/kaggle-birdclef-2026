"""Final verification of integrated R2 NB."""
import json, ast
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))

# Check ordering
print("=== Cell order check ===")
ids = [c.get("id", "") for c in nb["cells"]]
for i, cid in enumerate(ids):
    print(f"  [{i:02d}] {cid}")
regen_idx = ids.index("regen-r1-pseudo") if "regen-r1-pseudo" in ids else -1
load_idx = ids.index("load_data") if "load_data" in ids else -1
print(f"\nregen-r1-pseudo: position {regen_idx}")
print(f"load_data: position {load_idx}")
if regen_idx > 0 and load_idx > 0:
    if regen_idx < load_idx:
        print("OK: regen runs BEFORE load_data")
    else:
        print("FAIL: regen runs AFTER load_data")

# Syntax check (skip cell 1 setup which has !pip magic)
print("\n=== Syntax check (skip Jupyter-magic cells) ===")
fail = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if not src.strip(): continue
    # Skip cells with bash magic
    if src.lstrip().startswith("!") or "get_ipython().system" in src or "\n!" in src[:300]:
        print(f"  SKIP [{i:02d}] {c.get('id',''):20s} (Jupyter magic)")
        continue
    try:
        ast.parse(src)
        print(f"  OK [{i:02d}] {c.get('id','')}")
    except SyntaxError as e:
        print(f"  FAIL [{i:02d}] {c.get('id','')}: {e.msg} line {e.lineno}: {e.text}")
        fail += 1
print(f"\n{fail} syntax failures (excluding Jupyter-magic cells)")
