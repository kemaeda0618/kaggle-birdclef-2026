"""Pre-push sanity check: extract code cells from each notebook in this dir,
concatenate, and run py_compile to catch syntax errors before pushing.

Checks both nb_train.ipynb and nb_infer.ipynb.
"""
import json
import py_compile
import tempfile
import sys
from pathlib import Path

HERE = Path(__file__).parent
TARGETS = ["nb_train.ipynb", "nb_infer.ipynb"]

failures = 0
for nb_name in TARGETS:
    nb_path = HERE / nb_name
    if not nb_path.exists():
        print(f"\n[skip] {nb_name} not found")
        continue
    print(f"\n=== {nb_name} ===")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    merged = []
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        merged.append(f"# === cell {i} ({c.get('id', '?')}) ===")
        merged.append(src)
        merged.append("")
        n_code += 1

    text = "\n".join(merged)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name

    print(f"  code cells: {n_code}, total chars: {len(text)}")
    try:
        py_compile.compile(tmp, doraise=True)
        print(f"  OK: syntax check passed")
    except py_compile.PyCompileError as e:
        print(f"  FAIL: syntax error")
        print(e.msg)
        failures += 1
    finally:
        Path(tmp).unlink(missing_ok=True)

if failures:
    print(f"\n{failures} file(s) failed syntax check")
    sys.exit(1)
print("\nAll syntax checks passed")
