import json, re
from pathlib import Path

NBS = {
    "R1": "experiment/exp081/notebook/nb_train_r1.ipynb",
    "R2": "experiment/exp081/notebook/nb_train_r2.ipynb",
}

KEYS = [
    "BACKBONE", "N_TOTAL_EPOCHS", "BATCH", "LR", "MIN_LR", "WD",
    "WARMUP_EPOCHS", "AUG_PROB", "USE_MIXUP", "MIXUP_PROB", "MIXUP_ALPHA",
    "MIXUP_HARD", "FREQ_MASK_PARAM", "TIME_MASK_PARAM", "NUM_FREQ_MASKS",
    "NUM_TIME_MASKS", "MIN_SAMPLE", "SHARES", "SOURCE_WEIGHTS", "ALPHA_DISTILL",
    "USE_PERCH_DISTILL", "DROP_PATH_RATE", "drop_path_rate", "TRAIN_DURATION",
    "VAL_DURATION", "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX",
    "POWER_GAMMA", "GAUSS_SIGMA", "FORCE_FRESH", "USE_LABEL_SMOOTH",
    "LABEL_SMOOTH_EPS", "PSEUDO_THRESHOLD", "PSEUDO_TEMP",
    "N_FOLDS", "FOLDS", "NUM_WORKERS",
]

def extract_assignments(nb_path):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    cfg = {}
    for c in nb["cells"]:
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        for k in KEYS:
            pat = rf"(?:^|\n)[\s]*{re.escape(k)}[\s]*=[\s]*([^\n#]+?)(?:\s*#|$|\n)"
            m = re.search(pat, src, re.MULTILINE)
            if m:
                val = m.group(1).strip().rstrip(",")
                # Only first occurrence
                if k not in cfg:
                    cfg[k] = val
    return cfg

cfgs = {name: extract_assignments(p) for name, p in NBS.items()}

print(f"{'Key':<25} {'R1':<55} {'R2':<55} {'★':<2}")
print("-" * 140)
all_keys = set()
for c in cfgs.values():
    all_keys.update(c.keys())
for k in sorted(all_keys):
    r1 = cfgs["R1"].get(k, "-")
    r2 = cfgs["R2"].get(k, "-")
    diff = "★" if r1 != r2 else " "
    r1_short = r1[:53] if r1 else "-"
    r2_short = r2[:53] if r2 else "-"
    print(f"{k:<25} {r1_short:<55} {r2_short:<55} {diff}")
