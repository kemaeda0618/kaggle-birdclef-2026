import json
from pathlib import Path

for exp in ["exp014", "exp015", "exp016", "exp017"]:
    p = Path(f"experiment/{exp}/notebook/nb_train_r2.ipynb")
    nb = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n=== {exp} CFG cells ===")
    # Find the CFG cell (usually cell 1 or 2)
    for ci, c in enumerate(nb["cells"][:5]):
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        if "BACKBONE" in src or "N_TOTAL_EPOCHS" in src:
            # Print only assignments
            for ln in src.splitlines():
                ln2 = ln.strip()
                if "=" in ln2 and not ln2.startswith("#") and not ln2.startswith("from") and not ln2.startswith("import"):
                    print(f"  {ln2[:100]}")
            break
