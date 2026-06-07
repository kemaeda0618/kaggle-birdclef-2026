"""Generate exp029 R4: R3 self-pseudo → l1 student warm-start NB.

Design (Multi-iter Noisy Student R3 → R4, standard NS pattern):
- Backbone: eca_nfnet_l1 (same as R3)
- Architecture: basic SED + Perch v2 distill (same as R3)
- Warm start: R3 ckpt (r3_fold0_ckpt_best_ns22.pth)
- Teacher pseudo: ★ R3 self-inference on train_soundscapes (pseudo_exp029.csv)
  - kernel: maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029
  - 10,658 files × 12 windows = 127,896 rows, ~410MB
- N_TOTAL_EPOCHS = 12 (was 20 in R3, warm start で短縮)
- WARMUP_EPOCHS = 2 (was 4)
- LR = 1.5e-4 (was 3e-4、warm start で overshoot 防止)
- Same SHARES_R2 (focal 0.65 / labeled 0.10 / pseudo 0.25)

期待: R3 val 0.9409 → R4 val 0.943-0.948 (+0.002-0.007、memory 警告: R3→R4 diminishing)
LB: 0.923 → 0.925-0.930 (+0.002-0.007 single)
exp090 blend (weight 0.20) で lift: +0.001-0.002

Estimated: ~4-5h on Colab L4 (warm start で R3 20ep の 60% 程度)

Run: python _gen_nb_train_r4_l1_single.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
R3_GEN = HERE / "_gen_nb_train_r3_l1_single.py"
assert R3_GEN.exists(), f"R3 generator not found: {R3_GEN}"

src = R3_GEN.read_text(encoding="utf-8")

# ============================================================
# Apply patches: R3 → R4
# ============================================================
patches = [
    # 1. Header markdown
    ('# exp029 R3 (l1 single fold, exp017 R2 teacher)',
     '''# exp029 R4 (l1 single fold, R3 self-pseudo, warm start)

**前提**: R3 完走済、`maekeso/birdclef2026-exp029-l1-single` に r3_fold0_ckpt_best_ns22.pth
**前提**: Kaggle kernel `maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029` 完走済、pseudo_exp029.csv (410MB) available'''),

    # 2. Teacher pseudo header
    ('Teacher pseudo from Drive (`kaggle/birdclef2026/pseudo_e17.csv`、exp028 で生成済)',
     'Teacher pseudo: pseudo_exp029.csv (R3 self-inference on train_soundscapes、exp028 で生成済)'),

    # 3. N_EPOCHS in markdown
    ('- **N_EPOCHS=15** (Plan I、peak ep 4-8 に buffer +7-11、best_ckpt で確実 peak 捕捉)、WARMUP=3、batch=192、LR=3e-4',
     '- **N_EPOCHS=12** (warm start でR3 20ep より -8、peak は早期到達想定)、WARMUP=2、batch=192、LR=1.5e-4'),

    # 4. Drive output path: exp029 → exp029/r4
    # (exp029 NB outputs ckpt to Drive/output/exp029/{fold}/)
    # For R4 we want Drive/output/exp029/r4/
    # Look for the OUTPUT_DIR definition
    ('DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029"',
     'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029" / "r4"  # ★ R4 separate dir'),

    # 5. Teacher pseudo path: pseudo_e17 → pseudo_exp029 (all occurrences)
    ('pseudo_e17.csv', 'pseudo_exp029.csv'),
    ('pseudo_e17_kaggle', 'pseudo_exp029_kaggle'),

    # 6. Pseudo source kernel slug
    # R3 generates pseudo via exp028-pseudo-e17 kernel
    # We use exp028-pseudo-eca-nfnet-l1-exp029 (already complete)
    ('maekeso/birdclef2026-exp028-pseudo-e17',
     'maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029'),

    # 7. N_TOTAL_EPOCHS: 20 → 12
    ('N_TOTAL_EPOCHS = 20         # ★ exp029 Option C: 15 → 20、warmup 4/20 = 20% で適正',
     'N_TOTAL_EPOCHS = 12         # ★ R4: warm start から 12 ep (R3 20ep の 60%)'),

    # 8. WARMUP_EPOCHS: 4 → 2
    ('WARMUP_EPOCHS = 4         # ★ exp029: l1 で +1 ep (exp022 同設定)',
     'WARMUP_EPOCHS = 2         # ★ R4: warm start で短縮'),

    # 9. LR: 3e-4 → 1.5e-4
    ('LR = 3e-4',
     'LR = 1.5e-4               # ★ R4: warm start で overshoot 防止'),

    # 10. CKPT names: r3_fold0 → r4_fold0 (output side)
    # Look for save path patterns
    ('r3_fold{fold_k}_ckpt', 'r4_fold{fold_k}_ckpt'),
    ('r3_fold0_ckpt_best_ns22.pth', 'r4_fold0_ckpt_best_ns22.pth'),
    ('r3_fold0_ckpt_best_macro.pth', 'r4_fold0_ckpt_best_macro.pth'),
    ('r3_fold0_ckpt_latest.pth', 'r4_fold0_ckpt_latest.pth'),
    ('r3_fold0_history.json', 'r4_fold0_history.json'),

    # 11. Comment "R2" -> "R3/R4" 等 (markdown only)
    ('R2 pseudo gen は skip', 'R4: pseudo gen skip (pseudo_exp029.csv 利用)'),

    # 12. Kaggle Dataset upload slug (rename to indicate R4)
    ('birdclef2026-exp029-l1-single',
     'birdclef2026-exp029-r4-l1'),
]

print(f"Applying {len(patches)} patches to R3 generator...")
for old, new in patches:
    if old not in src:
        print(f"  [WARN] not found: {old[:80]!r}")
    else:
        n = src.count(old)
        src = src.replace(old, new)
        print(f"  [OK] {n}x: {old[:60]!r}")

# ============================================================
# CRITICAL: Add warm start logic
# ============================================================
# Find the cell that creates fresh model and inject R3 ckpt load before training loop
# The pattern is: "model = make_model()" — this is where we should load R3 ckpt

WARM_START_MARKER = "model = make_model()"
WARM_START_CODE = '''model = make_model()

    # ============================================================
    # ★ R4 WARM START: Load R3 ckpt as initial weights
    # ============================================================
    R3_CKPT_PATH = None
    R3_DRIVE_CKPT = DRIVE_INPUT_DIR / "output" / "exp029" / f"fold_{fold_k}" / "r3_fold0_ckpt_best_ns22.pth"
    R3_KAGGLE_CKPT = LOCAL_DATA / "r3_fold0_ckpt_best_ns22.pth"

    if R3_DRIVE_CKPT.exists():
        R3_CKPT_PATH = R3_DRIVE_CKPT
        print(f"[R4 WARM START] Loading R3 ckpt from Drive: {R3_CKPT_PATH}")
    else:
        # Download from Kaggle Dataset
        if not R3_KAGGLE_CKPT.exists():
            print(f"[R4 WARM START] R3 ckpt not in Drive, downloading from Kaggle Dataset...")
            try:
                subprocess.check_call([sys.executable, "-m", "kaggle", "datasets", "download",
                                        "-d", "maekeso/birdclef2026-exp029-l1-single",
                                        "-p", str(LOCAL_DATA), "--unzip", "-f", "r3_fold0_ckpt_best_ns22.pth"])
            except Exception:
                # Fallback to api
                from kaggle.api.kaggle_api_extended import KaggleApi
                _api = KaggleApi(); _api.authenticate()
                _api.dataset_download_file("maekeso/birdclef2026-exp029-l1-single",
                                            "r3_fold0_ckpt_best_ns22.pth",
                                            path=str(LOCAL_DATA), force=True)
                # Move from zip if needed
                _zip = LOCAL_DATA / "r3_fold0_ckpt_best_ns22.pth.zip"
                if _zip.exists():
                    import zipfile
                    with zipfile.ZipFile(_zip, "r") as _zf:
                        _zf.extractall(LOCAL_DATA)
                    _zip.unlink()
        R3_CKPT_PATH = R3_KAGGLE_CKPT

    assert R3_CKPT_PATH and R3_CKPT_PATH.exists(), f"R3 ckpt required for warm start, not found"

    try:
        r3_state = torch.load(str(R3_CKPT_PATH), map_location="cpu", weights_only=False)
    except TypeError:
        r3_state = torch.load(str(R3_CKPT_PATH), map_location="cpu")

    r3_msg = model.load_state_dict(r3_state["model_state"], strict=False)
    print(f"[R4 WARM START] Loaded R3 ckpt: epoch {r3_state.get('epoch')}, "
          f"best_ns22 {r3_state.get('best_ns22', float('nan')):.4f}, "
          f"best_macro {r3_state.get('best_macro', float('nan')):.4f}")
    print(f"[R4 WARM START] missing={len(r3_msg.missing_keys)}, unexpected={len(r3_msg.unexpected_keys)}")
    del r3_state'''

if WARM_START_MARKER in src:
    n = src.count(WARM_START_MARKER)
    src = src.replace(WARM_START_MARKER, WARM_START_CODE, 1)
    print(f"  [OK] Warm start injected at first 'model = make_model()' occurrence ({n} total found)")
else:
    print(f"  [WARN] Warm start marker '{WARM_START_MARKER}' not found, manual injection needed")

# ============================================================
# Save R4 generator
# ============================================================
R4_GEN = HERE / "_gen_nb_train_r4_l1_single.py"
# Update docstring first line to avoid infinite loop if someone regen with this script as source
src = src.replace(
    '"""Generate exp029: exp017 R2 single teacher → l1 student (single fold) NB.',
    '"""Generate exp029 R4: R3 self-pseudo → l1 student warm start NB.',
    1,
)

# Don't overwrite ourselves — save as separate file
# Actually we're writing the GENERATOR not the NB; the generator name is _gen_nb_train_r4_l1_single.py
# But we ARE this file. So save as the NB generator, then execute it would be infinite loop.
# Instead, directly produce the NB JSON below using the modified src as a string.

# Approach: write modified src to a temp file, exec it, capture the NB output
# Simpler: just save the modified generator AND run it via subprocess
TEMP_GEN = HERE / "_gen_nb_train_r4_l1_single_temp.py"
TEMP_GEN.write_text(src, encoding="utf-8")
print(f"\n[OK] Wrote modified generator to {TEMP_GEN}")

# Run the temp generator
import subprocess, sys
print("\nRunning temp generator to produce nb_train_r4_l1_single.ipynb...")
result = subprocess.run(
    [sys.executable, str(TEMP_GEN)],
    capture_output=True, text=True, encoding="utf-8",
)
print("STDOUT:", result.stdout[-500:])
print("STDERR:", result.stderr[-500:])

# Find the output NB — R3 generator writes to nb_train_r3_l1_single.ipynb in same dir
# After our patch (r3 -> r4 not applied to filename in generator), the output is still nb_train_r3_l1_single.ipynb
# We need to rename
import shutil
old_nb = HERE / "nb_train_r3_l1_single.ipynb"
new_nb = HERE / "nb_train_r4_l1_single.ipynb"
if old_nb.exists():
    # Check if it was just created by our exec (compare timestamps)
    shutil.move(str(old_nb), str(new_nb))
    print(f"[OK] Renamed {old_nb.name} → {new_nb.name}")
elif new_nb.exists():
    print(f"[OK] {new_nb.name} already exists")
else:
    print(f"[WARN] No output NB found")

# Cleanup temp
if TEMP_GEN.exists():
    TEMP_GEN.unlink()
    print(f"[OK] Cleaned up temp generator")

# Verify NB
if new_nb.exists():
    nb = json.loads(new_nb.read_text(encoding="utf-8"))
    print(f"\n[OK] Final NB: {new_nb}")
    print(f"  cells: {len(nb['cells'])}")
    # Check for R4 markers
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    markers = {
        "exp029 R4": "exp029 R4 in NB",
        "pseudo_exp029.csv": "pseudo_exp029.csv reference",
        "R4 WARM START": "R4 warm start logic",
        "N_TOTAL_EPOCHS = 12": "12 epochs",
        "WARMUP_EPOCHS = 2": "2 warmup epochs",
        "LR = 1.5e-4": "LR 1.5e-4",
        "birdclef2026-exp029-r4-l1": "new Dataset slug",
    }
    print("\nMarker checks:")
    for m, desc in markers.items():
        present = m in all_src
        print(f"  {'[OK]' if present else '[MISS]'} {desc}")
