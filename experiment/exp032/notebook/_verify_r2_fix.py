"""Verify exp032 R2 NB has correct mel params + Kaggle v1 force-DL."""
import json, ast
nb_path = r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

print(f"=== cells: {len(nb['cells'])} ===\n")

# Check mel params
print("=== Mel params (should be BC2026 standard, NOT Babych-fix) ===")
checks_mel = [
    ("N_FFT = 2048 (good)", "N_FFT          = 2048" in all_src or "N_FFT = 2048" in all_src),
    ("HOP_LENGTH = 512 (good)", "HOP_LENGTH     = 512" in all_src or "HOP_LENGTH = 512" in all_src),
    ("N_MELS = 256 (good)", "N_MELS         = 256" in all_src or "N_MELS = 256" in all_src),
    ("FMIN = 20 (good)", "FMIN           = 20" in all_src or "FMIN = 20" in all_src),
    ("N_FFT = 4096 (BAD, mel-fix)", "N_FFT          = 4096" in all_src or "N_FFT = 4096" in all_src),
    ("HOP_LENGTH = 312 (BAD)", "HOP_LENGTH     = 312" in all_src or "HOP_LENGTH = 312" in all_src),
    ("N_MELS = 224 (BAD)", "N_MELS         = 224" in all_src or "N_MELS = 224" in all_src),
]
for label, ok in checks_mel:
    if "good" in label:
        status = "OK" if ok else "FAIL"
    else:  # bad
        status = "FAIL (bad found!)" if ok else "OK (bad absent)"
    print(f"  [{status}] {label}")

# Check Kaggle v1 force-DL cell
print("\n=== Kaggle Dataset v1 force-DL ===")
checks_dl = [
    ("r1-v1-force-dl cell", "r1-v1-force-dl" in all_src),
    ("version_number=1", "version_number=1" in all_src),
    ("force overwrite Drive", "overwrote Drive ckpt with v1" in all_src),
    ("val_ns22 sanity assert", "0.910 < _v1_val < 0.920" in all_src),
]
for label, ok in checks_dl:
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")

# Syntax check
print("\n=== Syntax ===")
fail = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if not src.strip(): continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  FAIL [{i:02d}] {c.get('id','')}: {e.msg} line {e.lineno}")
        fail += 1
print(f"  {fail} syntax failures out of {sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells")
