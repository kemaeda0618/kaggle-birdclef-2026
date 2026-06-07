"""Fix R4 NB path nesting and naming issues.

Issues fixed:
1. DRIVE_EXP_DIR revert: output/exp029/r4 → output/exp029 (back to R3 convention)
   - Use /r4 subfolder INSIDE fold_dir instead (R3 convention pattern)
2. /r3 → /r4 subfolder inside fold dir
3. Upload stage names: r3_fold → r4_fold
4. Upload TITLE: R3 → R4 R3-self-pseudo
5. Print messages: "R3 (e17 teacher)" → "R4 (R3 self-pseudo)"
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_train_r4_l1_single.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Patch dict: search → replace
# ============================================================
patches = [
    # Path: revert DRIVE_EXP_DIR (don't add /r4 at top level)
    ('DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029" / "r4"  # ★ R4 separate dir',
     'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029"  # ★ R4 uses /r4 subfolder inside fold dir'),

    # In fold_loop: rename /r3/ subfolder to /r4/
    ('DRIVE_R2_DIR    = DRIVE_FOLD_DIR / "r3"',
     'DRIVE_R2_DIR    = DRIVE_FOLD_DIR / "r4"  # ★ R4 round subfolder'),

    ('LOCAL_R2 = LOCAL_OUT / f"fold{FOLD_K}" / "r3"',
     'LOCAL_R2 = LOCAL_OUT / f"fold{FOLD_K}" / "r4"  # ★ R4 round subfolder'),

    # In upload cell: DRIVE_R3_DIR also needs /r4
    ('DRIVE_R3_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r3"',
     'DRIVE_R3_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r4"  # ★ R4 round subfolder'),

    # Upload stage filename prefix r3 → r4
    ('dst = td / f"r3_fold{fold_k}_{fn}"',
     'dst = td / f"r4_fold{fold_k}_{fn}"'),

    # Upload TITLE
    ('TITLE = "birdclef2026 exp029 l1 single (R3 from e17)"',
     'TITLE = "birdclef2026 exp029 R4 l1 (R3 self-pseudo, warm start)"'),

    # Upload version_notes
    ('version_notes = f"exp029 l1 single (R3 from e17) | {summary}"[:498]',
     'version_notes = f"exp029 R4 l1 single (R3 self-pseudo, warm start) | {summary}"[:498]'),

    # fold_loop print message
    ('# exp029 R3 (l1 student, e17 teacher) Fold {FOLD_K} (of {N_FOLDS}-fold split, single fold training)',
     '# exp029 R4 (l1 student, R3 self-pseudo, warm start) Fold {FOLD_K} (of {N_FOLDS}-fold split, single fold training)'),

    # fold_loop comment (teacher pseudo)
    ('# ★ exp029: teacher pseudo は exp017 R2 (pseudo_exp029.csv) を Drive 上に置く前提\n    # pseudo_exp029.csv は exp028 で生成済 → Kaggle Notebook output から download → Drive へ\n    # Drive path: kaggle/birdclef2026/pseudo_exp029.csv (固定)',
     '# ★ R4: teacher pseudo は R3 self-inference (pseudo_exp029.csv) を Drive 上に置く前提\n    # pseudo_exp029.csv は exp028-pseudo-eca-nfnet-l1-exp029 kernel で生成済 → Drive へ\n    # Drive path: kaggle/birdclef2026/pseudo_exp029.csv (固定)'),

    # fold_loop teacher load print
    ('print(f"  Loading teacher pseudo (exp017 R2): {DRIVE_TEACHER_PSEUDO}")',
     'print(f"  Loading teacher pseudo (R3 self): {DRIVE_TEACHER_PSEUDO}")'),

    # fold_loop print "R3 ckpt already on Drive"
    ('print(f"  Fold {FOLD_K}: R3 ckpt already on Drive — skip")',
     'print(f"  Fold {FOLD_K}: R4 ckpt already on Drive — skip")'),

    # fold_loop final print
    ('print(f"\\n  >>> Fold {FOLD_K} R2 DONE in {fold_elapsed/60:.1f}min: R2={r2_best_ns22:.4f} <<<\\n")',
     'print(f"\\n  >>> Fold {FOLD_K} R4 DONE in {fold_elapsed/60:.1f}min: R4={r2_best_ns22:.4f} <<<\\n")'),

    # 5-fold R2 complete
    ('print(f"\\n{\'=\'*60}\\n5-fold R2 complete\\n{\'=\'*60}")',
     'print(f"\\n{\'=\'*60}\\n5-fold R4 complete\\n{\'=\'*60}")'),

    # train_r2 function header (cosmetic comment)
    ('# Cell 8: train_r2 function',
     '# Cell 8: train function (R4: warm start + R3 self-pseudo)'),

    # train_r2 internal print
    ('print(f"\\n{\'=\'*60}\\n[Fold {fold_k} R2] train ({N_TOTAL_EPOCHS} ep)\\n{\'=\'*60}")',
     'print(f"\\n{\'=\'*60}\\n[Fold {fold_k} R4] train ({N_TOTAL_EPOCHS} ep, warm start from R3)\\n{\'=\'*60}")'),

    # Upload cell header
    ('# Cell 10: Upload R1 + R2 ckpts to Kaggle Dataset',
     '# Cell 10: Upload R4 ckpts to Kaggle Dataset'),

    # Upload comment "R3 ckpt のみ upload"
    ('# ★ exp029: R3 ckpt のみ upload (R1/R2 は別 exp で管理)',
     '# ★ R4: R4 ckpt のみ upload'),

    # Upload DRIVE_R3_DIR variable name (cosmetic, but consistent)
    # Skip — keeping as DRIVE_R3_DIR is OK since it just points to the directory we save R4 into now

    # Header markdown structure (R3 → R4)
    ('## 想定: ~5.5-6h on Blackwell (Plan I、20 ep 比 -30%)',
     '## 想定: ~4-5h on Colab L4 (warm start で R3 20ep の 60%)'),
]

print(f"Applying {len(patches)} patches...")
n_applied = 0
n_missed = 0
for old, new in patches:
    if old in "".join("".join(c.get("source", [])) for c in nb["cells"]):
        n_applied += 1
    else:
        n_missed += 1
        print(f"  [MISS] {old[:80]!r}")

for c in nb["cells"]:
    src = "".join(c.get("source", []))
    new_src = src
    for old, new in patches:
        if old in new_src:
            new_src = new_src.replace(old, new)
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

print(f"\nApplied: {n_applied}, Missed: {n_missed}")

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")

# Verify
print("\n=== Verification (post-patch markers) ===")
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
checks = [
    ("DRIVE_EXP_DIR reverted (no top-level /r4)",
     'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029"' in all_src and '/ "exp029" / "r4"' not in all_src),
    ("DRIVE_R2_DIR uses /r4 subfolder", '"r4"  # ★ R4 round subfolder' in all_src),
    ("Upload uses r4_fold prefix", 'r4_fold{fold_k}_{fn}' in all_src),
    ("TITLE updated", 'R4 l1 (R3 self-pseudo, warm start)' in all_src),
    ("teacher pseudo comment updated", 'R3 self-inference' in all_src),
    ("fold_loop R4 header", '# exp029 R4 (l1 student, R3 self-pseudo' in all_src),
    ("R3 ckpt path (warm start) intact", 'r3_fold0_ckpt_best_ns22.pth' in all_src),
    ("pseudo_exp029.csv reference", 'pseudo_exp029.csv' in all_src),
]
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

# Compile-strict check
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
