"""Compare exp017 R1, R2, exp029 R3 architecture + key params."""
import json, re
from pathlib import Path

files = {
    "exp017 R1": Path("experiment/exp017/notebook/nb_train.ipynb"),
    "exp017 R2": Path("experiment/exp017/notebook/nb_train_r2.ipynb"),
    "exp029 R3": Path("experiment/exp020/notebook/nb_train_r3.ipynb"),  # R3 might be in exp020 or exp029
}

# Try multiple R3 locations
r3_candidates = [
    Path("experiment/exp020/notebook/nb_train_r3.ipynb"),
    Path("experiment/exp020/notebook/nb_train_r3_l1.ipynb"),
    Path("experiment/exp029/notebook/nb_train_r3.ipynb"),
    Path("experiment/exp029/notebook/nb_train_r3_l1.ipynb"),
    Path("experiment/exp029/notebook/nb_train_r3_l1_single.ipynb"),
]
for c in r3_candidates:
    if c.exists():
        files["exp029 R3"] = c
        print(f"Found R3 at: {c}")
        break
else:
    print("R3 nb not found! Searching for any R3 related...")
    import glob
    for p in glob.glob("experiment/exp029/notebook/*.ipynb"):
        print(f"  exp029: {p}")
    for p in glob.glob("experiment/exp020/notebook/*.ipynb"):
        print(f"  exp020: {p}")

# Keys to extract
KEYS_NUMERIC = [
    "N_TOTAL_EPOCHS", "BATCH", "LR", "MIN_LR", "WD", "WARMUP_EPOCHS",
    "ALPHA_DISTILL", "PERCH_EMBED_DIM",
    "MIXUP_PROB", "MIXUP_ALPHA",
    "FREQ_MASK_PARAM", "TIME_MASK_PARAM", "NUM_FREQ_MASKS", "NUM_TIME_MASKS",
    "MIN_SAMPLE",
    "AUG_PROB",
    "TRAIN_DURATION", "VAL_DURATION",
    "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX",
    "SR",
    "drop_path_rate",
    "NUM_WORKERS",
]
KEYS_STRING = [
    "BACKBONE",
    "SHARES", "SOURCE_WEIGHTS",
    "USE_PERCH_DISTILL", "USE_MIXUP", "MIXUP_HARD",
    "AUG_GAIN_DB_RANGE", "AUG_NOISE_SNR_DB_RANGE",
    "FOLDS", "N_FOLDS",
    "FORCE_FRESH", "PERSISTENT_WORKERS",
]

ALL_KEYS = KEYS_NUMERIC + KEYS_STRING

for name, fp in files.items():
    print(f"\n{'='*80}")
    print(f"=== {name}  ({fp})")
    print(f"{'='*80}")
    if not fp.exists():
        print(f"  NOT FOUND")
        continue
    nb = json.loads(fp.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    found = {}
    for k in ALL_KEYS:
        m = re.search(rf"^\s*{re.escape(k)}\s*=\s*([^#\n]+)", all_src, re.MULTILINE)
        if m:
            found[k] = m.group(1).strip().rstrip(",")
    for k in ALL_KEYS:
        if k in found:
            v = found[k][:90]
            print(f"  {k:25} = {v}")
    # Loss-related code (extracting key snippet)
    print()
    print("  --- LOSS / pseudo handling ---")
    if "USE_PSEUDO" in all_src or "pseudo" in all_src.lower():
        # Extract small snippets
        for kw in ["pseudo_loss", "PSEUDO", "pseudo_csv", "pseudo_npz", "labeled_sc", "PSEUDO_GAMMA", "POWER"]:
            for line in all_src.splitlines():
                if kw in line and ("=" in line or "loss" in line):
                    s = line.strip()
                    if s and not s.startswith("#") and len(s) < 200:
                        print(f"  {s[:150]}")
                        break
