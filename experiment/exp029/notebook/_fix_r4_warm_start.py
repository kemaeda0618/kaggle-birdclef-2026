"""Fix warm start logic in nb_train_r4_l1_single.ipynb.

R3 saves to Drive: output/exp029/fold_{k}/ckpt_best_ns22.pth (no r3 prefix in filename)
Kaggle Dataset: r3_fold0_ckpt_best_ns22.pth (with r3 prefix)

Make warm start logic try both.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "nb_train_r4_l1_single.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

OLD = '''    R3_CKPT_PATH = None
    R3_DRIVE_CKPT = DRIVE_INPUT_DIR / "output" / "exp029" / f"fold_{fold_k}" / "r3_fold0_ckpt_best_ns22.pth"
    R3_KAGGLE_CKPT = LOCAL_DATA / "r3_fold0_ckpt_best_ns22.pth"

    if R3_DRIVE_CKPT.exists():'''

NEW = '''    R3_CKPT_PATH = None
    # R3 NB saved to Drive without "r3_" prefix; Kaggle Dataset has "r3_fold0_" prefix
    R3_DRIVE_CANDIDATES = [
        DRIVE_INPUT_DIR / "output" / "exp029" / f"fold_{fold_k}" / "ckpt_best_ns22.pth",
        DRIVE_INPUT_DIR / "output" / "exp029" / f"fold_{fold_k}" / "r3_fold0_ckpt_best_ns22.pth",
    ]
    R3_KAGGLE_CKPT = LOCAL_DATA / "r3_fold0_ckpt_best_ns22.pth"
    R3_DRIVE_CKPT = next((p for p in R3_DRIVE_CANDIDATES if p.exists()), None)

    if R3_DRIVE_CKPT is not None:'''

n_patched = 0
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        c["source"] = new_src.splitlines(keepends=True)
        n_patched += 1

print(f"Patched {n_patched} cell(s)")
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")

# Compile check
print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
