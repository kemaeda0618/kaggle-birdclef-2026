"""Patch R4 train NB for cleaner Dataset organization.

Changes:
1. Pseudo source: kernel output → Dataset (more standard)
   OLD: kernel `maekeso/birdclef2026-exp029-pseudo-r3-train-sc` output download
   NEW: Dataset `maekeso/birdclef2026-exp029-pseudo-r3` download

2. R4 ckpt upload target: rename for R3 pattern consistency
   OLD: `birdclef2026-exp029-r4-l1`
   NEW: `birdclef2026-exp029-r4-l1-single` (matches R3 pattern: birdclef2026-exp029-l1-single)
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_train_r4_l1_single.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Patch 1: Replace pseudo source (kernel output → Dataset)
# ============================================================
# Current dl_data cell has logic like:
#   api.kernels_output_download_file("maekeso/birdclef2026-exp029-pseudo-r3-train-sc",
#                                     file_name="pseudo_exp029.csv", ...)
# Replace with Dataset download.

OLD_KERNEL_BLOCK = '''        api.kernels_output_download_file("maekeso/birdclef2026-exp029-pseudo-r3-train-sc",
                                         file_name="pseudo_exp029.csv",
                                         path=str(TEACHER_PSEUDO_TMP))'''

NEW_DATASET_BLOCK = '''        # ★ V2: Download from Dataset (more standard than kernel output)
        api.dataset_download_file("maekeso/birdclef2026-exp029-pseudo-r3",
                                   "pseudo_exp029.csv",
                                   path=str(TEACHER_PSEUDO_TMP), force=True)
        # Unzip if needed (dataset_download_file may zip single files)
        _zip_path = TEACHER_PSEUDO_TMP / "pseudo_exp029.csv.zip"
        if _zip_path.exists():
            import zipfile
            with zipfile.ZipFile(_zip_path, "r") as _zf:
                _zf.extractall(TEACHER_PSEUDO_TMP)
            _zip_path.unlink()'''

n_patched_pseudo = 0
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if OLD_KERNEL_BLOCK in src:
        new_src = src.replace(OLD_KERNEL_BLOCK, NEW_DATASET_BLOCK)
        c["source"] = new_src.splitlines(keepends=True)
        n_patched_pseudo += 1
print(f"[Pseudo source] Patched {n_patched_pseudo} cell(s)")

# ============================================================
# Patch 2: Rename R4 ckpt upload Dataset slug
# ============================================================
OLD_SLUG = "birdclef2026-exp029-r4-l1"
NEW_SLUG = "birdclef2026-exp029-r4-l1-single"

n_patched_slug = 0
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    # Only replace EXACT OLD_SLUG (not when it's part of NEW_SLUG which contains OLD_SLUG)
    # We need precise replacement: OLD_SLUG NOT followed by "-single"
    import re
    pattern = re.compile(re.escape(OLD_SLUG) + r"(?!-single)")
    if pattern.search(src):
        new_src = pattern.sub(NEW_SLUG, src)
        c["source"] = new_src.splitlines(keepends=True)
        n_patched_slug += 1
print(f"[Upload slug] Patched {n_patched_slug} cell(s): {OLD_SLUG} → {NEW_SLUG}")

# ============================================================
# Save
# ============================================================
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")

# ============================================================
# Verify
# ============================================================
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verification ===")
checks = [
    ("Pseudo from Dataset (NEW)", "dataset_download_file" in all_src and "birdclef2026-exp029-pseudo-r3\"" in all_src),
    ("Old kernel download removed", "kernels_output_download_file" not in all_src or "pseudo_exp029.csv" not in all_src.split("kernels_output_download_file")[1][:500] if "kernels_output_download_file" in all_src else True),
    ("Upload slug: r4-l1-single (NEW)", "birdclef2026-exp029-r4-l1-single" in all_src),
    ("Old slug (without -single) removed", not __import__("re").search(r"birdclef2026-exp029-r4-l1(?!-single)", all_src)),
    ("R3 ckpt path intact", "r3_fold0_ckpt_best_ns22.pth" in all_src),
    ("Warm start intact", "R4 WARM START" in all_src),
]
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

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
