import json, re
from pathlib import Path

NBS = {
    "exp014": "experiment/exp014/notebook/nb_train_r2.ipynb",
    "exp015": "experiment/exp015/notebook/nb_train_r2.ipynb",
    "exp016": "experiment/exp016/notebook/nb_train_r2.ipynb",
    "exp017": "experiment/exp017/notebook/nb_train_r2.ipynb",
}

KEYS = ["BACKBONE", "N_TOTAL_EPOCHS", "LR_BACKBONE", "LR_HEAD", "BATCH_SIZE",
        "N_MELS", "N_FFT", "HOP_LENGTH", "F_MIN", "F_MAX", "TRAIN_DURATION",
        "DROP_PATH", "drop_path_rate", "MIXUP_RATIO", "MIXUP_BLEND",
        "FOCAL_RATIO", "PSEUDO_RATIO", "LABELED_SC_RATIO", "USE_PERCH_DISTILL",
        "ALPHA_DISTILL", "WEIGHT_DECAY", "WARMUP_EPOCHS", "USE_LABEL_SMOOTH",
        "LABEL_SMOOTH_EPS"]

result = {}
for name, path in NBS.items():
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = {}
    for c in nb["cells"]:
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        for k in KEYS:
            # Find assignment
            pat = rf"{re.escape(k)}[\s]*=[\s]*([^\n#,]+)"
            m = re.search(pat, src, re.MULTILINE)
            if m and k not in cfg:
                cfg[k] = m.group(1).strip()
    result[name] = cfg

# Print comparison table
print(f"{'Key':<25} {'exp014':<25} {'exp015':<25} {'exp016':<25} {'exp017':<25}")
print("-" * 130)
all_keys = set()
for cfg in result.values():
    all_keys.update(cfg.keys())
for k in sorted(all_keys):
    vals = [result[e].get(k, "-") for e in ["exp014", "exp015", "exp016", "exp017"]]
    diff = "★" if len(set(vals)) > 1 else " "
    print(f"{diff} {k:<23} {vals[0]:<25} {vals[1]:<25} {vals[2]:<25} {vals[3]:<25}")
