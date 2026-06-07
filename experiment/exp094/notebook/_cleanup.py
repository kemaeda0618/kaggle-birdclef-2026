"""Clean up stale outputs + comment in exp094 NB."""
import json
from pathlib import Path

fp = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(fp.read_text(encoding="utf-8"))

# Fix the comment
OLD_LINE = 'PERCH_META_DIR = LOCAL_DATA / "perch-meta"   # for lbl_to_bc.csv (Mapped index map)'
NEW_LINE = '# PERCH_META_DIR removed -- using labels.csv from perch-onnx dataset instead'

for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if OLD_LINE in src:
        new_src = src.replace(OLD_LINE, NEW_LINE)
        c["source"] = new_src.splitlines(keepends=True)
        print("[OK] PERCH_META_DIR comment updated")
        break

# Clear stale error outputs
n_cleared = 0
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    outs = c.get("outputs", [])
    has_stale = any(
        (isinstance(o, dict) and (o.get("ename") == "AssertionError" or "lbl_to_bc" in str(o.get("evalue", ""))))
        for o in outs
    )
    if has_stale:
        c["outputs"] = []
        c["execution_count"] = None
        n_cleared += 1

print(f"[OK] Cleared {n_cleared} stale error outputs")

fp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {fp}")
