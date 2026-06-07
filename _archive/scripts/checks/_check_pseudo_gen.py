import json
from pathlib import Path

for nb_path in ["experiment/exp081/notebook/nb_train_r1.ipynb",
                "experiment/exp081/notebook/nb_train_r2.ipynb"]:
    print(f"\n=== {nb_path} ===")
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    for ci, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        # Look for pseudo-related logic
        if any(kw in src for kw in ["pseudo", "infer_pseudo", "r1-pseudo", "PSEUDO_OUT"]):
            # Print top 30 lines + cell id
            lines = src.splitlines()
            print(f"  --- Cell {ci} id={c.get('id', '?')} ({len(lines)} lines) ---")
            for ln in lines[:60]:
                ln2 = ln[:120]
                print(f"    {ln2}")
            if len(lines) > 60:
                print(f"    ... ({len(lines) - 60} more lines)")
            print()
