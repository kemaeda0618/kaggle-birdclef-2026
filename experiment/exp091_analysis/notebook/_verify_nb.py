"""Verify nb_infer_analysis with compile()."""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_infer_analysis.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
        cid = c.get("id", "?")
        print(f"  [OK] cell {i} (id={cid})")
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} (id={c.get('id','?')}): {e}")

print(f"\nResult: {n_ok} OK, {n_err} ERRORS")
