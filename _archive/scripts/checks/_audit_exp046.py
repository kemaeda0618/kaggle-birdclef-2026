"""Deep audit exp046 R3 vs exp029 R3 to find ALL differences."""
import json, re
from pathlib import Path

EXP029_GEN = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\_gen_nb_train_r3_l1_single.py")
EXP046_NB  = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp046\notebook\nb_train_r3_l1_gamma12.ipynb")

# Read both
exp029_src = EXP029_GEN.read_text(encoding="utf-8")
exp046_nb = json.loads(EXP046_NB.read_text(encoding="utf-8"))
exp046_src = "\n".join("".join(c.get("source", [])) for c in exp046_nb["cells"] if c["cell_type"] == "code")

# Comprehensive list of config keys to check
KEYS = [
    # Architecture
    "BACKBONE", "USE_PERCH_DISTILL", "ALPHA_DISTILL", "PERCH_EMBED_DIM",
    "drop_path_rate", "hidden_dim",
    # Training
    "N_TOTAL_EPOCHS", "BATCH", "LR", "MIN_LR", "WD", "WARMUP_EPOCHS",
    "FOLDS", "N_FOLDS", "SEED",
    # Augmentation
    "USE_MIXUP", "MIXUP_PROB", "MIXUP_ALPHA", "MIXUP_HARD",
    "FREQ_MASK_PARAM", "TIME_MASK_PARAM", "NUM_FREQ_MASKS", "NUM_TIME_MASKS",
    "AUG_PROB", "AUG_GAIN_DB_RANGE", "AUG_NOISE_SNR_DB_RANGE",
    # Data
    "TRAIN_DURATION", "VAL_DURATION", "SR",
    "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX",
    "MIN_SAMPLE",
    # SHARES / pseudo
    "SHARES_R2", "SHARES", "SOURCE_WEIGHTS",
    "POWER_GAMMA", "PSEUDO_GAMMA",
    "FILTER_THRESH", "MIN_PROB",
    "USE_PSEUDO", "PSEUDO_PATH",
    # Other
    "NUM_WORKERS", "FORCE_FRESH", "PERSISTENT_WORKERS",
]

def extract_value(src, key):
    """Find first assignment value."""
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^#\n]+)", src, re.MULTILINE)
    if m:
        return m.group(1).strip().rstrip(",")
    return None

results = {}
for k in KEYS:
    v_029 = extract_value(exp029_src, k)
    v_046 = extract_value(exp046_src, k)
    results[k] = (v_029, v_046)

print(f"{'KEY':30} | {'exp029 R3':30} | {'exp046 R3 γ=1.2':30} | DIFF?")
print("-" * 110)
for k, (v0, v1) in results.items():
    s0 = (v0 or "-")[:30]
    s1 = (v1 or "-")[:30]
    diff = "* DIFF" if v0 != v1 and (v0 is not None or v1 is not None) else ""
    print(f"{k:30} | {s0:30} | {s1:30} | {diff}")

# Diff details
print("\n=== Key differences ===")
for k, (v0, v1) in results.items():
    if v0 != v1 and (v0 is not None or v1 is not None):
        print(f"  {k}:")
        print(f"    exp029 R3: {v0}")
        print(f"    exp046 R3: {v1}")

# Now compare LOSS, training loop, pseudo loading
print("\n" + "=" * 80)
print("=== Pseudo loading / Power Transform code search ===")
print("=" * 80)

# Find pseudo loading and Power Transform in each
for name, src in [("exp046", exp046_src), ("exp029 (gen)", exp029_src)]:
    print(f"\n--- {name} ---")
    print("  Pseudo loading:")
    for ln, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if "pseudo" in s.lower() and ("read_csv" in s or "load" in s or "DRIVE" in s or "PSEUDO_PATH" in s):
            print(f"    L{ln}: {s[:170]}")
    print("  Power Transform application:")
    for ln, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if "np.power" in s or "POWER_GAMMA" in s:
            print(f"    L{ln}: {s[:170]}")
