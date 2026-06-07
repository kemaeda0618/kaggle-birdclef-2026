"""Verify exp095 NB with l0 + R2 exact spec."""
import json
from pathlib import Path

NB = Path(__file__).with_name("nb_train_r3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])}")
print()

all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print("=== Architecture & Backbone ===")
checks_arch = [
    ('BACKBONE = "eca_nfnet_l0"', 'BACKBONE = "eca_nfnet_l0"' in all_src),
    ('NO eca_nfnet_l1', '"eca_nfnet_l1"' not in all_src),
    ('USE_PERCH_DISTILL = True', "USE_PERCH_DISTILL = True" in all_src),
    ('ALPHA_DISTILL = 1.0', "ALPHA_DISTILL = 1.0" in all_src),
    ('drop_path_rate = 0.1', "drop_path_rate=0.1" in all_src or "drop_path_rate = 0.1" in all_src),
]
for name, ok in checks_arch:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

print()
print("=== Training params (R2 exact spec) ===")
checks_train = [
    ('N_TOTAL_EPOCHS = 20', "N_TOTAL_EPOCHS = 20" in all_src),
    ('WARMUP_EPOCHS = 3 (R2 spec)', "WARMUP_EPOCHS = 3" in all_src),
    ('NO WARMUP_EPOCHS = 4', "WARMUP_EPOCHS = 4" not in all_src),
    ('BATCH = 192', "BATCH = 192" in all_src),
    ('LR = 3e-4', "LR = 3e-4" in all_src),
    ('MIN_LR = 1e-6', "MIN_LR = 1e-6" in all_src),
    ('WD = 1e-4', "WD = 1e-4" in all_src),
    ('N_FOLDS = 5, FOLDS = [0]', "N_FOLDS = 5" in all_src and "FOLDS = [0]" in all_src),
]
for name, ok in checks_train:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

print()
print("=== Data / Mel (exp017 R2 spec) ===")
checks_data = [
    ('TRAIN_DURATION = 5', "TRAIN_DURATION = 5" in all_src),
    ('SR = 32000', "SR = 32000" in all_src),
    ('N_FFT = 2048', "N_FFT = 2048" in all_src),
    ('HOP_LENGTH = 512', "HOP_LENGTH = 512" in all_src),
    ('N_MELS = 256', "N_MELS = 256" in all_src),
    ('FMIN = 20', "FMIN = 20" in all_src),
    ('FMAX = 16000', "FMAX = 16000" in all_src),
]
for name, ok in checks_data:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

print()
print("=== Augmentation (R2 spec same) ===")
checks_aug = [
    ('MIXUP_PROB = 0.5', "MIXUP_PROB = 0.5" in all_src),
    ('MIXUP_ALPHA = 0.4', "MIXUP_ALPHA = 0.4" in all_src),
    ('FREQ_MASK_PARAM = 25', "FREQ_MASK_PARAM = 25" in all_src),
    ('TIME_MASK_PARAM = 30', "TIME_MASK_PARAM = 30" in all_src),
    ('NUM_FREQ_MASKS = 2', "NUM_FREQ_MASKS = 2" in all_src),
    ('NUM_TIME_MASKS = 2', "NUM_TIME_MASKS = 2" in all_src),
    ('AUG_PROB = 0.5', "AUG_PROB = 0.5" in all_src),
]
for name, ok in checks_aug:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

print()
print("=== SHARES (R2 exact spec) ===")
checks_shares = [
    ('SHARES_R2 focal=0.70', '"focal": 0.70' in all_src),
    ('SHARES_R2 labeled_sc=0.10', '"labeled_sc": 0.10' in all_src),
    ('SHARES_R2 pseudo_sc=0.20', '"pseudo_sc": 0.20' in all_src),
    ('NO pseudo_sc=0.25', '"pseudo_sc": 0.25' not in all_src),
]
for name, ok in checks_shares:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

print()
print("=== Teacher + Output ===")
checks_data2 = [
    ('Teacher: pseudo_e17.csv', "pseudo_e17.csv" in all_src),
    ('Output dir: output/exp095', '"output" / "exp095"' in all_src),
    ('Slug: exp095-l0-r2spec', "birdclef2026-exp095-l0-r2spec" in all_src),
    ('NO old slug (l1-single)', "birdclef2026-exp095-l1-single" not in all_src),
]
for name, ok in checks_data2:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

# Compile
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
