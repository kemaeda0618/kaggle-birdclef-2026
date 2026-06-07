"""Verify syntax of inference NBs."""
import json, ast
from pathlib import Path

NB_PATHS = [
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_infer.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_infer_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_infer.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/ensemble_e14_17/notebook/nb_infer_ensemble.ipynb",
]

for p in NB_PATHS:
    fname = "/".join(p.split("/")[-2:])
    print(f"\n=== {fname} ===")
    nb = json.loads(Path(p).read_text(encoding="utf-8"))
    n_ok, n_err = 0, 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        lines = []
        for ln in src.splitlines():
            stripped = ln.lstrip()
            if stripped.startswith("!") or stripped.startswith("%"):
                lines.append("# " + ln)
            else:
                lines.append(ln)
        clean_src = "\n".join(lines)
        try:
            ast.parse(clean_src)
            n_ok += 1
        except SyntaxError as e:
            n_err += 1
            print(f"  SyntaxError in cell id={c.get('id', '?')}: {e}")
            err_line = e.lineno or 1
            src_lines = clean_src.splitlines()
            start = max(0, err_line - 3)
            end = min(len(src_lines), err_line + 3)
            for i in range(start, end):
                marker = ">>> " if i + 1 == err_line else "    "
                print(f"  {marker}{i+1:4d}: {src_lines[i]}")
    print(f"  Total: {n_ok} OK, {n_err} ERRORS")
