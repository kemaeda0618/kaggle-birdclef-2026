"""Pre-push sanity check: extract all cell sources from nb_pl1_pseudo.ipynb,
concatenate, and run py_compile to catch syntax errors before pushing."""
import json
import py_compile
import tempfile
from pathlib import Path

NB = Path(__file__).with_name("nb_pl1_pseudo.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

merged = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    merged.append(f"# === cell {i} ({c['id']}) ===")
    merged.append(src)
    merged.append("")

text = "\n".join(merged)
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(text)
    tmp = f.name

print(f"Wrote {len(merged)//3} cells to {tmp}")
print(f"Total chars: {len(text)}")

try:
    py_compile.compile(tmp, doraise=True)
    print("OK: syntax check passed")
except py_compile.PyCompileError as e:
    print("FAIL: syntax error")
    print(e.msg)
    Path(tmp).unlink(missing_ok=True)
    raise

Path(tmp).unlink(missing_ok=True)
