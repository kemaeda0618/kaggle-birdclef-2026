"""Compare R1 vs R2 training hyperparameters in detail."""
import json
import re

r1 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train.ipynb", encoding="utf-8"))
r2 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))

def get_cell(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"])
    return None

# Compare key cells
for cid in ("model", "dataset", "train_loop", "resume_or_init"):
    r1_c = get_cell(r1, cid)
    r2_c = get_cell(r2, cid)
    if r1_c is None or r2_c is None:
        print(f"=== {cid}: r1={r1_c is not None}, r2={r2_c is not None} ===")
        continue
    if r1_c == r2_c:
        print(f"=== {cid}: IDENTICAL ✅ ===")
    else:
        print(f"=== {cid}: DIFFER ⚠️ ===")
        # Print length diff
        print(f"  r1 lines: {len(r1_c.splitlines())}, r2 lines: {len(r2_c.splitlines())}")

# Highlight key training params side by side
print("\n=== Training hyperparameter comparison ===")
r1_cfg = get_cell(r1, "config") or ""
r2_cfg = get_cell(r2, "config") or ""

def find_param(cfg, name):
    for line in cfg.split("\n"):
        m = re.match(rf"^({name})\s*=\s*(.+?)(?:\s*#|$)", line.strip())
        if m:
            return m.group(2).strip()
    return None

params = ["N_TOTAL_EPOCHS", "BATCH", "LR", "MIN_LR", "WD", "WARMUP_EPOCHS",
          "SEED", "N_FOLDS", "FOLDS", "FORCE_FRESH",
          "USE_MIXUP", "MIXUP_PROB", "MIXUP_ALPHA", "MIXUP_HARD",
          "AUG_PROB", "AUG_GAIN_DB_RANGE", "AUG_NOISE_SNR_DB_RANGE",
          "FREQ_MASK_PARAM", "TIME_MASK_PARAM", "NUM_FREQ_MASKS", "NUM_TIME_MASKS",
          "MIN_SAMPLE", "USE_PERCH_DISTILL", "PERCH_EMBED_DIM", "ALPHA_DISTILL",
          "NUM_WORKERS", "PERSISTENT_WORKERS", "MAX_RUNTIME_SEC"]

print(f"{'param':25s} | {'R1':30s} | {'R2':30s} | match")
print("-" * 100)
for p in params:
    v1 = find_param(r1_cfg, p) or "?"
    v2 = find_param(r2_cfg, p) or "?"
    match = "✅" if v1 == v2 else "⚠️"
    print(f"{p:25s} | {v1:30s} | {v2:30s} | {match}")

# Sources/shares
print("\n=== Sources / mix shares ===")
for line in r1_cfg.split("\n"):
    if "SHARES" in line or "SOURCE_WEIGHTS" in line:
        print(f"  R1: {line.strip()}")
for line in r2_cfg.split("\n"):
    if "SHARES" in line or "SOURCE_WEIGHTS" in line:
        print(f"  R2: {line.strip()}")
