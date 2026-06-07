"""Fix missed patches in exp094 NB.

4 missed in build:
1. PRIMARY_LABELS / l2i mapping - actual pattern is LABEL2IDX
2. Drive output path - actual pattern is "exp017" / "r1"
3-4. Title / slug - different naming
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# ============================================================
# Patch 1: After PRIMARY_LABELS assertion, add BC_INDICES / MAPPED_POS construction
# ============================================================
PATCH1_OLD = '''PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES'''

PATCH1_NEW = '''PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES

# === exp094: Build BC_INDICES / MAPPED_POS after PRIMARY_LABELS is defined ===
if USE_ALI_KD:
    BC_INDICES = np.array([
        int(_lbl2bc.loc[c]) if c in _lbl2bc.index else NO_LABEL
        for c in PRIMARY_LABELS
    ], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != NO_LABEL
    MAPPED_POS_NP = np.where(MAPPED_MASK)[0].astype(np.int64)
    MAPPED_BC_IDX_NP = BC_INDICES[MAPPED_MASK].astype(np.int64)
    MAPPED_POS = torch.from_numpy(MAPPED_POS_NP).long().to(device)
    MAPPED_BC_IDX = torch.from_numpy(MAPPED_BC_IDX_NP).long().to(device)
    print(f"[ALI KD] Mapped: {MAPPED_MASK.sum()} / {NUM_CLASSES} species have a Perch logit")
    print(f"[ALI KD] MAPPED_POS shape: {MAPPED_POS.shape}, MAPPED_BC_IDX shape: {MAPPED_BC_IDX.shape}")'''

# ============================================================
# Patch 2: Drive output path exp017 -> exp094
# ============================================================
PATCH2_OLD = 'DRIVE_OUTPUT_DIR = DRIVE_INPUT_DIR / "output" / "exp017" / "r1"'
PATCH2_NEW = 'DRIVE_OUTPUT_DIR = DRIVE_INPUT_DIR / "output" / "exp094" / "r1"  # ★ exp094: separate dir from exp017'

# ============================================================
# Patch 3: Pseudo Drive path
# ============================================================
PATCH3_OLD = '"output" / "exp017" / "r1-pseudo"'
PATCH3_NEW = '"output" / "exp094" / "r1-pseudo"'

# Also patch markdown comments mentioning exp017 r1 path
PATCH3B_OLD = 'output/exp017/r1/'
PATCH3B_NEW = 'output/exp094/r1/'

PATCH3C_OLD = 'output/exp017/r1-pseudo/'
PATCH3C_NEW = 'output/exp094/r1-pseudo/'

# ============================================================
# Patch 4: Dataset upload slug
# ============================================================
PATCH4_OLD = 'maekeso/birdclef2026-exp017-weights'
PATCH4_NEW = 'maekeso/birdclef2026-exp094-ali-r1'

PATCH4B_OLD = 'birdclef2026-exp017-weights'
PATCH4B_NEW = 'birdclef2026-exp094-ali-r1'

# Title (if exists)
PATCH4C_OLD = 'birdclef2026 exp017 weights'
PATCH4C_NEW = 'birdclef2026 exp094 ali r1'

# ============================================================
# Apply
# ============================================================
patches = [
    (PATCH1_OLD, PATCH1_NEW),
    (PATCH2_OLD, PATCH2_NEW),
    (PATCH3_OLD, PATCH3_NEW),
    (PATCH3B_OLD, PATCH3B_NEW),
    (PATCH3C_OLD, PATCH3C_NEW),
    (PATCH4_OLD, PATCH4_NEW),
    (PATCH4B_OLD, PATCH4B_NEW),
    (PATCH4C_OLD, PATCH4C_NEW),
]

print(f"Applying {len(patches)} fix patches...")
applied_total = 0
for old, new in patches:
    found = 0
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if old in src:
            new_src = src.replace(old, new)
            c["source"] = new_src.splitlines(keepends=True)
            found += src.count(old)
    if found > 0:
        applied_total += found
        print(f"  [OK] {found}x: {old[:80]!r}")
    else:
        print(f"  [MISS] {old[:80]!r}")

print(f"\nTotal replacements: {applied_total}")

# Save
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB}")

# Verify
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verification ===")
checks = [
    ("BC_INDICES construction present", "BC_INDICES = np.array([" in all_src),
    ("MAPPED_POS = torch.from_numpy", "MAPPED_POS = torch.from_numpy" in all_src),
    ("MAPPED_BC_IDX = torch.from_numpy", "MAPPED_BC_IDX = torch.from_numpy" in all_src),
    ("Drive path exp094/r1", '"output" / "exp094" / "r1"' in all_src),
    ("Drive path exp094/r1-pseudo", '"output" / "exp094" / "r1-pseudo"' in all_src),
    ("No old exp017/r1 Drive path", '"output" / "exp017" / "r1"' not in all_src),
    ("Dataset slug exp094", "birdclef2026-exp094-ali-r1" in all_src),
    ("No old dataset slug exp017", "birdclef2026-exp017-weights" not in all_src),
    # Sanity check (existing functionality intact)
    ("ALPHA_DISTILL = 1.0 intact", "ALPHA_DISTILL = 1.0" in all_src),
    ("USE_ALI_KD = True", "USE_ALI_KD     = True" in all_src),
    ("T_DISTILL = 1.0", "T_DISTILL      = 1.0" in all_src),
    ("ALPHA_LOGIT = 0.1", "ALPHA_LOGIT    = 0.1" in all_src),
    ("KL loss code", "distill_logit_loss = F.binary_cross_entropy_with_logits" in all_src),
    ("Loss combined", "+ ALPHA_LOGIT * distill_logit_loss" in all_src),
    ("embed_and_logits method", "def embed_and_logits" in all_src),
    ("BACKBONE eca_nfnet_l0", 'BACKBONE = "eca_nfnet_l0"' in all_src),
    ("N_TOTAL_EPOCHS = 25", "N_TOTAL_EPOCHS = 25" in all_src),
    ("BATCH = 192", "BATCH = 192" in all_src),
    ("LR = 3e-4", "LR = 3e-4" in all_src),
]
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok: all_ok = False

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

if all_ok and n_err == 0:
    print("\n[OK] exp094 NB ready for Colab L4 training")
