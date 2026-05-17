"""Pre-check: extract all cell sources from nb_pl2_sed_train.ipynb,
concatenate, and run py_compile to catch syntax errors."""
import json
import py_compile
import tempfile
import re
from pathlib import Path

NB = Path(__file__).with_name("nb_pl2_sed_train.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

merged = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # Comment out shell escapes (! and %) so py_compile doesn't choke
    # (handle leading whitespace too — Jupyter allows indented !)
    src_safe = re.sub(r"^(\s*)([!%].*)$", r"\1# \2", src, flags=re.MULTILINE)
    merged.append(f"# === cell {i} ({c['id']}) ===")
    merged.append(src_safe)
    merged.append("")

text = "\n".join(merged)
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(text)
    tmp = f.name

try:
    py_compile.compile(tmp, doraise=True)
    print("OK: syntax check passed")
except py_compile.PyCompileError as e:
    print("FAIL: syntax error")
    print(e.msg)
    raise
finally:
    Path(tmp).unlink(missing_ok=True)
