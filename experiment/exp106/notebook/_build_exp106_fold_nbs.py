"""Build 4 fold-specific train NBs for exp106 = 5-fold exp029 R3 ensemble.

Approach: Run the existing exp029 gen script 4 times with FOLDS=[k] each,
producing 4 separate NBs (fold 1, 2, 3, 4) to run in parallel on Colab.

This avoids re-implementing the full exp029 R3 recipe — we leverage the
already-verified gen_nb_train_r3_l1_single.py with just FOLDS patched.

Output:
  experiment/exp106/notebook/nb_exp106_fold1.ipynb
  experiment/exp106/notebook/nb_exp106_fold2.ipynb
  experiment/exp106/notebook/nb_exp106_fold3.ipynb
  experiment/exp106/notebook/nb_exp106_fold4.ipynb
"""
import io, re, shutil, sys, subprocess
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ORIG_GEN = Path("experiment/exp029/notebook/_gen_nb_train_r3_l1_single.py")
OUT_DIR = Path("experiment/exp106/notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)

orig_src = ORIG_GEN.read_text(encoding="utf-8")

# Verify the FOLDS line exists (defensive)
assert "FOLDS = [0]" in orig_src, "FOLDS = [0] anchor not found in original gen"

# Verify DRIVE_EXP_DIR is exp029 (we'll redirect to exp106)
assert "DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / \"output\" / \"exp029\"" in orig_src, \
    "DRIVE_EXP_DIR exp029 anchor not found"

# Verify NB output path
assert "nb_train_r3_l1_single.ipynb" in orig_src or "NB_PATH" in orig_src, \
    "NB output path anchor not found (will patch find by NB_PATH search)"

# Find the NB_PATH line so we can override it
nb_path_match = re.search(r'NB_PATH\s*=\s*Path\([^)]+\)', orig_src)
print(f"Original NB_PATH line: {nb_path_match.group(0) if nb_path_match else 'NOT FOUND'}")


def patch_for_fold(fold_k):
    """Patch original gen src for one specific fold."""
    src = orig_src

    # 1. FOLDS = [0]  ->  FOLDS = [k]
    src = src.replace("FOLDS = [0]", f"FOLDS = [{fold_k}]")

    # 2. DRIVE_EXP_DIR  exp029  ->  exp106  (keep ckpts segregated)
    src = src.replace(
        'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029"',
        'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp106"'
    )

    # 3. out_path redirect to exp106 dir, per-fold name (orig uses HERE / "nb_train_r3_l1_single.ipynb")
    new_nb_name = f"nb_exp106_fold{fold_k}.ipynb"
    old_out = 'out_path = HERE / "nb_train_r3_l1_single.ipynb"'
    new_out = f'out_path = HERE / "{new_nb_name}"'
    assert old_out in src, f"out_path anchor not found"
    src = src.replace(old_out, new_out)

    # 4. Upload SLUG: ensure per-fold slug to avoid version conflict
    # search for SLUG = "birdclef2026-exp029-l1-single"
    if 'SLUG  = "birdclef2026-exp029-l1-single"' in src:
        src = src.replace(
            'SLUG  = "birdclef2026-exp029-l1-single"',
            f'SLUG  = "birdclef2026-exp106-fold{fold_k}"'
        )
    elif 'SLUG = "birdclef2026-exp029-l1-single"' in src:
        src = src.replace(
            'SLUG = "birdclef2026-exp029-l1-single"',
            f'SLUG = "birdclef2026-exp106-fold{fold_k}"'
        )

    # 5. TITLE: keep <= 50 chars
    title_pattern = r'TITLE\s*=\s*"[^"]*"'
    src = re.sub(title_pattern, f'TITLE = "birdclef2026 exp106 fold{fold_k}"', src, count=1)

    return src


for fold_k in [1, 2]:   # ★ Plan C: 2 folds for 3-fold ensemble (fold 0 既存 + fold 1, 2)
    out_script = OUT_DIR / f"_gen_nb_exp106_fold{fold_k}.py"
    patched = patch_for_fold(fold_k)
    out_script.write_text(patched, encoding="utf-8")
    print(f"\nWrote {out_script}")
    # Run the patched gen script to produce the .ipynb
    result = subprocess.run(
        [sys.executable, str(out_script)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    if result.returncode != 0:
        print(f"  [FAIL] gen: {result.stderr[-500:]}")
    else:
        print(f"  [OK] generated fold{fold_k}")
        if result.stdout:
            print(f"    stdout: {result.stdout[-200:]}")

print("\n=== All 4 fold gen scripts done ===")
print(f"Run each NB on Colab (4 in parallel):")
for k in [1, 2, 3, 4]:
    print(f"  experiment/exp106/notebook/nb_exp106_fold{k}.ipynb")
