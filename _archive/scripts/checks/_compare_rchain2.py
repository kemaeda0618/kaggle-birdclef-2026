"""Compare exp017 R1, R2, R3 (from exp045 base) configs."""
import json, re
from pathlib import Path

files = {
    "exp017 R1": Path("experiment/exp017/notebook/nb_train.ipynb"),
    "exp017 R2": Path("experiment/exp017/notebook/nb_train_r2.ipynb"),
    "exp045 R3 (R3 base + filter)": Path("experiment/exp045/notebook/nb_train_r3_l1_filtered.ipynb"),
    "exp046 R3 (R3 base + gamma)": Path("experiment/exp046/notebook/nb_train_r3_l1_gamma12.ipynb"),
    "exp029 R4 (R3 base + warm start)": Path("experiment/exp029/notebook/nb_train_r4_l1_single.ipynb"),
}

KEYS_NUMERIC = [
    "N_TOTAL_EPOCHS", "BATCH", "LR", "MIN_LR", "WD", "WARMUP_EPOCHS",
    "ALPHA_DISTILL",
    "MIXUP_PROB", "MIXUP_ALPHA",
    "FREQ_MASK_PARAM", "TIME_MASK_PARAM", "NUM_FREQ_MASKS", "NUM_TIME_MASKS",
    "MIN_SAMPLE", "AUG_PROB",
    "TRAIN_DURATION", "VAL_DURATION",
    "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX", "SR",
    "drop_path_rate", "NUM_WORKERS",
    "POWER_GAMMA", "PSEUDO_GAMMA",
    "FILTER_THRESH", "MIN_PROB",
]
KEYS_STRING = [
    "BACKBONE",
    "SHARES", "SOURCE_WEIGHTS",
    "USE_PERCH_DISTILL", "USE_MIXUP", "MIXUP_HARD",
    "FOLDS", "N_FOLDS",
    "FORCE_FRESH",
]

ALL_KEYS = KEYS_NUMERIC + KEYS_STRING

results = {}
for name, fp in files.items():
    if not fp.exists():
        print(f"=== {name}: NOT FOUND ({fp})")
        results[name] = {}
        continue
    nb = json.loads(fp.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    found = {}
    for k in ALL_KEYS:
        m = re.search(rf"^\s*{re.escape(k)}\s*=\s*([^#\n]+)", all_src, re.MULTILINE)
        if m:
            found[k] = m.group(1).strip().rstrip(",")
    results[name] = found

# Print comparison table
print(f"{'PARAM':27} | " + " | ".join(f"{n[:24]:24}" for n in files.keys()))
print("-" * (27 + 27 * len(files)))
for k in ALL_KEYS:
    values = []
    for name in files.keys():
        v = results.get(name, {}).get(k, "-")
        values.append(f"{v[:24]:24}")
    line = f"{k:27} | " + " | ".join(values)
    # Highlight differences
    if len(set(v.strip() for v in values if v.strip() != "-")) > 1:
        line = "* " + line
    print(line)
