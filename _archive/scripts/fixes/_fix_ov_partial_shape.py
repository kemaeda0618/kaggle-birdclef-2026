"""Fix: use partial_shape instead of shape for dynamic-axis OV models.
shape errors out on dynamic axis, partial_shape works.
"""
import json
from pathlib import Path

P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "inputs[0].shape" in src:
        # Replace .shape with .partial_shape
        src_new = src.replace("inputs[0].shape", "inputs[0].partial_shape")
        src_new = src_new.replace("{o.any_name} shape={o.shape}", "{o.any_name} shape={o.partial_shape}")
        c["source"] = src_new.splitlines(keepends=True)
        print("OK Replaced .shape → .partial_shape in cell")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Syntax check
import ast
nb = json.loads(P.read_text(encoding="utf-8"))
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
