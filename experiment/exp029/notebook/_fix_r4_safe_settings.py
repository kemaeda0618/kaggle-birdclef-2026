"""Revert R4 NB hyperparameters to R3-same settings (Safe path).

Only difference between R3 and R4:
- Warm start from R3 ckpt (R4)
- New pseudo (R3 self-inference) instead of exp017 R2 pseudo (R3)

All other hyperparameters identical to R3.
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_train_r4_l1_single.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    # Config: revert to R3 values
    ('N_TOTAL_EPOCHS = 12         # ★ R4: warm start から 12 ep (R3 20ep の 60%)',
     'N_TOTAL_EPOCHS = 20         # ★ R4: R3 同 setting (warm start のみ追加、parameter は変えず)'),

    ('WARMUP_EPOCHS = 2         # ★ R4: warm start で短縮',
     'WARMUP_EPOCHS = 4         # ★ R4: R3 同 setting'),

    ('LR = 1.5e-4               # ★ R4: warm start で overshoot 防止',
     'LR = 3e-4                 # ★ R4: R3 同 setting (Safe path)'),

    # Header markdown: revert expected times
    ('## 想定: ~4-5h on Colab L4 (warm start で R3 20ep の 60%)',
     '## 想定: ~5.5-6h on Colab L4 (R3 同 setting、warm start のみ追加)'),

    # Update docstring at top
    ('- N_TOTAL_EPOCHS = 12 (was 20 in R3, warm start で短縮)',
     '- N_TOTAL_EPOCHS = 20 (R3 と同、Safe path)'),

    ('- WARMUP_EPOCHS = 2 (was 4)',
     '- WARMUP_EPOCHS = 4 (R3 と同)'),

    ('- LR = 1.5e-4 (was 3e-4、warm start で overshoot 防止)',
     '- LR = 3e-4 (R3 と同、Safe path)'),

    # Update header table comments
    ('- **N_EPOCHS=12** (warm start でR3 20ep より -8、peak は早期到達想定)、WARMUP=2、batch=192、LR=1.5e-4',
     '- **N_EPOCHS=20** (R3 と同 setting)、WARMUP=4、batch=192、LR=3e-4'),
]

print(f"Applying {len(patches)} patches (revert to R3 settings)...")
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
n_applied = 0
n_missed = 0
miss_log = []
for old, new in patches:
    if old in all_src:
        n_applied += 1
    else:
        n_missed += 1
        miss_log.append(old[:80])

for c in nb["cells"]:
    src = "".join(c.get("source", []))
    new_src = src
    for old, new in patches:
        if old in new_src:
            new_src = new_src.replace(old, new)
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

print(f"Applied: {n_applied}, Missed: {n_missed}")
for m in miss_log:
    print(f"  [MISS] {m!r}")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")

# Verify
print("\n=== Verification ===")
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
checks = [
    ("N_TOTAL_EPOCHS = 20", "N_TOTAL_EPOCHS = 20" in all_src),
    ("WARMUP_EPOCHS = 4", "WARMUP_EPOCHS = 4" in all_src),
    ("LR = 3e-4", "LR = 3e-4" in all_src),
    ("NO N_TOTAL_EPOCHS = 12", "N_TOTAL_EPOCHS = 12" not in all_src),
    ("NO WARMUP_EPOCHS = 2", "WARMUP_EPOCHS = 2" not in all_src),
    ("NO LR = 1.5e-4", "LR = 1.5e-4" not in all_src),
    # Warm start intact
    ("Warm start logic intact", "R4 WARM START" in all_src),
    ("R3 ckpt load intact", "r3_fold0_ckpt_best_ns22.pth" in all_src),
    ("Pseudo source intact", "pseudo_exp029.csv" in all_src),
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
