"""Summarize all pseudo experiments: backbone, teacher, params, results."""
import json
import re
from pathlib import Path

EXP_DIR = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment")

# Target experiments with NS pseudo / training paradigm
TARGETS = {
    "exp017_R1": (EXP_DIR / "exp017" / "notebook" / "nb_train.ipynb", "exp017 R1 (l0 base, NS iter 0)"),
    "exp017_R2": (EXP_DIR / "exp017" / "notebook" / "nb_train_r2.ipynb", "exp017 R2 (l0, NS iter 1)"),
    "exp020_R1": (None, "exp020 R1 (regnety_008 5-fold)"),  # might not exist
    "exp029_R3": (None, "exp029 R3 (l1, NS iter 2)"),  # check exp020 R3 or 029
    "exp029_R4": (EXP_DIR / "exp029" / "notebook" / "nb_train_r4_l1_single.ipynb", "exp029 R4 (l1, NS iter 3, R3 self)"),
    "exp032_R2": (EXP_DIR / "exp032" / "notebook" / "nb_train_r2.ipynb", "exp032 R2 (b3 + Babych init)"),
    "exp045_R3": (EXP_DIR / "exp045" / "notebook" / "nb_train_r3_l1_filtered.ipynb", "exp045 R3 (l1, chunk filter)"),
    "exp046_R3": (EXP_DIR / "exp046" / "notebook" / "nb_train_r3_l1_gamma12.ipynb", "exp046 R3 (l1, Power γ=1.2)"),
    "exp049_R1": (EXP_DIR / "exp049" / "notebook" / "nb_train_effv2s_r1.ipynb", "exp049 R1 (effv2s)"),
    "exp050_R1": (EXP_DIR / "exp050" / "notebook" / "nb_train_convnext_r1.ipynb", "exp050 R1 (convnext)"),
    "exp051_R1": (EXP_DIR / "exp051" / "notebook" / "nb_train_swin_r1.ipynb", "exp051 R1 (swin)"),
    "exp080":    (EXP_DIR / "exp080" / "notebook" / "nb_train_colab.ipynb", "exp080 (b0, 1-stage combined + XC)"),
    "exp081_R2": (EXP_DIR / "exp081" / "notebook" / "nb_train_r2.ipynb", "exp081 R2 (l0, 10s window)"),
    "exp082_R2": (EXP_DIR / "exp082" / "notebook" / "nb_train_r2.ipynb", "exp082 R2 (l0, 20s window)"),
    "exp083_R2": (EXP_DIR / "exp083" / "notebook" / "nb_train_r2.ipynb", "exp083 R2 (?)"),
    "exp084_R2": (EXP_DIR / "exp084" / "notebook" / "nb_train_r2_colab.ipynb", "exp084 R2 (?)"),
    "exp094_R1": (EXP_DIR / "exp094" / "notebook" / "nb_train_ali.ipynb", "exp094 R1-Ali (l0, Tucker MSE + Ali KL)"),
}

# Keys to extract
KEYS = [
    "BACKBONE", "N_TOTAL_EPOCHS", "BATCH", "LR", "WARMUP_EPOCHS", "WD",
    "ALPHA_DISTILL", "USE_PERCH_DISTILL",
    "MIXUP_PROB", "MIXUP_ALPHA", "USE_MIXUP",
    "MIXUP_RATIO", "PSEUDO_MIXUP",
    "SHARES", "SOURCE_WEIGHTS",
    "TRAIN_DURATION", "VAL_DURATION", "TRAIN_SAMPLES",
    "N_MELS", "N_FFT", "HOP_LENGTH",
    "drop_path_rate", "DROP_PATH",
    "T_DISTILL", "ALPHA_LOGIT", "USE_ALI_KD",
    "USE_PSEUDO", "PSEUDO_SHARE", "pseudo_share",
    "POWER", "GAMMA", "PSEUDO_GAMMA",
    "FILTER", "CHUNK_FILTER", "MIN_PROB",
]

def extract_value(line):
    """Get value after '=' from a config line."""
    if "=" not in line:
        return None
    val = line.split("=", 1)[1].split("#")[0].strip()
    if val.endswith(","): val = val[:-1]
    return val[:80]


for key, (path, desc) in TARGETS.items():
    print(f"\n{'='*70}")
    print(f"### {key}: {desc}")
    print(f"{'='*70}")
    if path is None:
        print("(file path unknown)")
        continue
    if not path.exists():
        print(f"(file not found: {path})")
        continue

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"(error reading: {e})")
        continue

    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        for line in src.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for k in KEYS:
                # match k followed by = (or spaces +=)
                m = re.match(rf"\s*{re.escape(k)}\s*=", line)
                if m and k not in found:
                    found[k] = extract_value(line)

    for k in KEYS:
        if k in found:
            print(f"  {k:25} = {found[k]}")
