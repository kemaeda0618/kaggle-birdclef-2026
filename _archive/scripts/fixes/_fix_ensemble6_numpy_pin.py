"""Fix numpy/scipy compat: constrain numpy<2 when installing openvino."""
import json
from pathlib import Path

P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "!pip install -q openvino==2024.4.0" in src:
        # Fix: pin numpy < 2 to keep scipy compat
        old = "!pip install -q openvino==2024.4.0"
        new = '''# ★ openvino install with numpy pin (scipy._no_nep50_warning issue 回避)
!pip install -q "openvino==2024.4.0" "numpy<2"
# Restart numpy + scipy import (in case stale)
import importlib, numpy, scipy
importlib.reload(numpy)
importlib.reload(scipy)
print(f'numpy={numpy.__version__}, scipy={scipy.__version__}')'''
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("OK Updated cell 1: numpy<2 pin + reload")
        break

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
