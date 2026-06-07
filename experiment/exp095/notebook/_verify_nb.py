"""Verify exp095 generated NB."""
import json
from pathlib import Path

NB = Path(__file__).with_name("nb_train_r3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])} ({sum(1 for c in nb['cells'] if c['cell_type']=='code')} code, "
      f"{sum(1 for c in nb['cells'] if c['cell_type']=='markdown')} markdown)")
print()

# Show cell preview
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first_line = src.split("\n")[0][:120] if src.strip() else "(empty)"
    print(f"  Cell {i} ({c['cell_type']:8}): {first_line}")

# Verify key config params
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print()
print("=== Config sanity checks (should match exp029 R3 spec) ===")
checks = [
    ("BACKBONE = eca_nfnet_l1", 'BACKBONE = "eca_nfnet_l1"' in all_src),
    ("N_TOTAL_EPOCHS = 20", "N_TOTAL_EPOCHS = 20" in all_src),
    ("WARMUP_EPOCHS = 4", "WARMUP_EPOCHS = 4" in all_src),
    ("LR = 3e-4", "LR = 3e-4" in all_src),
    ("BATCH = 192", "BATCH = 192" in all_src),
    ("MIXUP_PROB = 0.5", "MIXUP_PROB = 0.5" in all_src),
    ("MIXUP_ALPHA = 0.4", "MIXUP_ALPHA = 0.4" in all_src),
    ("ALPHA_DISTILL = 1.0", "ALPHA_DISTILL = 1.0" in all_src),
    ("USE_PERCH_DISTILL = True", "USE_PERCH_DISTILL = True" in all_src),
    ("N_FOLDS = 5", "N_FOLDS = 5" in all_src),
    ("FOLDS = [0]", "FOLDS = [0]" in all_src),
    ("TRAIN_DURATION = 5", "TRAIN_DURATION = 5" in all_src),
    ("N_MELS = 256", "N_MELS = 256" in all_src),
    ("N_FFT = 2048", "N_FFT = 2048" in all_src),
    ("HOP_LENGTH = 512", "HOP_LENGTH = 512" in all_src),
    ('SHARES_R2 pseudo_sc=0.25', '"pseudo_sc": 0.25' in all_src),
    ('Teacher: exp017 R2 pseudo_e17.csv', "pseudo_e17.csv" in all_src),
    ('Output dir: exp095', '"output" / "exp095"' in all_src),
    ('Slug: exp095', "birdclef2026-exp095-l1-single" in all_src),
    ('No exp029-l1-single slug', "birdclef2026-exp029-l1-single" not in all_src),
]
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok: all_ok = False

# Compile-strict
print()
print("=== Compile check ===")
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
    print("\n[OK] exp095 NB ready for Colab upload")
