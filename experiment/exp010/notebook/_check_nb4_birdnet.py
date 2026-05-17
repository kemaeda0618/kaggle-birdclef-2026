"""Pre-push sanity check: extract all cell sources from nb4_birdnet.ipynb and
nb_birdnet_onnx_export.ipynb, concatenate, and run py_compile to catch syntax
errors before pushing."""
import json
import py_compile
import tempfile
from pathlib import Path

TARGETS = [
    Path(__file__).with_name("nb4_birdnet.ipynb"),
    Path(__file__).with_name("nb_birdnet_onnx_export.ipynb"),
]

for NB in TARGETS:
    print(f"\n=== {NB.name} ===")
    if not NB.exists():
        print(f"SKIP: {NB} not found (run generator first)")
        continue
    nb = json.loads(NB.read_text(encoding="utf-8"))

    merged = []
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(c["source"])
        merged.append(f"# === cell {i} ({c['id']}) ===")
        merged.append(src)
        merged.append("")

    text = "\n".join(merged)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name

    print(f"Wrote {n_code} code cells to {tmp}")
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
