"""Build exp106 standalone OV NB (3-fold ensemble using e106 OV ckpts).

Replaces the PyTorch version (which times out). Uses pre-converted OV IR from
maekeso/birdclef2026-e106-3fold-ov (exp029 R3 fold 0 + exp106 fold 1 + fold 2).

Output replaces existing nb_exp106_standalone_3fold.ipynb (Version 2 of same slug).
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Clone from exp104 standalone (OV-based 3-fold) and swap to exp106
EXP104_NB = Path("experiment/exp104/notebook/nb_exp104_standalone_3fold.ipynb")
EXP106_NB = Path("experiment/exp106/notebook/nb_exp106_standalone_3fold.ipynb")

nb = json.loads(EXP104_NB.read_text(encoding="utf-8"))

# Patch cells: swap exp102 → exp106
patched_cells = []
for c in nb["cells"]:
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    # Variable name swaps
    src = src.replace("FOLDS_E102", "FOLDS_E106")
    src = src.replace("EXP102_DIR", "EXP106_DIR")
    src = src.replace("EXP102_IR", "EXP106_IR")
    src = src.replace("_ov_e102", "_ov_e106")
    src = src.replace("_core_e102", "_core_e106")
    # Dataset slug
    src = src.replace("birdclef2026-exp102-3fold-ov", "birdclef2026-e106-3fold-ov")
    # File name pattern
    src = src.replace("exp102_fold", "exp106_fold")
    # Comments / messages
    src = src.replace("exp102 OV", "exp106 OV")
    src = src.replace("e102-ov", "e106-ov")
    src = src.replace("exp102 3-fold", "exp106 3-fold")

    # FOLDS = [0, 1, 2] should already include all 3 folds (fold 0 = exp029 R3, fold 1/2 = exp106)
    # Confirm: in exp104 standalone, FOLDS = [0, 1, 2] for exp102 3-fold. Same logic for exp106.

    c["source"] = src.splitlines(keepends=True)
    patched_cells.append(c)

nb["cells"] = patched_cells

EXP106_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {EXP106_NB} ({EXP106_NB.stat().st_size} bytes)")

# Verify critical patches
nb_check = json.loads(EXP106_NB.read_text(encoding="utf-8"))
all_src = "".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb_check["cells"]
)
assert "EXP106_DIR" in all_src
assert "birdclef2026-e106-3fold-ov" in all_src
assert "exp106_fold" in all_src
assert "_ov_e106" in all_src
assert "FOLDS = [0, 1, 2]" in all_src or "FOLDS_E106 = [0, 1, 2]" in all_src
print("[OK] all critical patches present")
print("Cells:", len(nb_check["cells"]))
