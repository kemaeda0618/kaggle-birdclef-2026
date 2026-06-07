"""Verify all cells are syntactically valid Python (skipping %% and ! magics)."""
import json, ast, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb = json.load(open("experiment/convert_to_ov/notebook/convert_to_ov.ipynb", encoding="utf-8"))
n_ok, n_err = 0, 0
errs = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        ast.parse(clean)
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        errs.append(f"cell {i}: {e}")
print(f"Syntax: {n_ok} OK, {n_err} ERRORS")
for e in errs:
    print(e)

# Also verify specific content in new cells
for needle in [
    "regnety_008",
    "exp016_r2",
    "birdclef2026-exp016-weights",
    "eca_nfnet_l1",
    "exp029_r3",
    "birdclef2026-exp029-l1-single",
    "r3_fold0_ckpt_best_ns22.pth",
    "r3_ov",   # exp029 uses r3_ov/ subfolder
]:
    found = any(needle in "".join(c.get("source", [])) for c in nb["cells"])
    print(f"  {'[OK]' if found else '[MISS]'} '{needle}' in NB")
