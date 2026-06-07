"""Audit existing implementation of paper 256 techniques."""
import json
from pathlib import Path

# Files to inspect
TRAIN_NBS = [
    ("exp032 R2", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb"),
    ("exp017", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp017\notebook\nb_train.ipynb"),
    ("exp020 R2 5-fold", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp020\notebook\nb_train_r2_5fold.ipynb"),
    ("exp029 R3", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\nb_train_r3_l1_single.ipynb"),
]

INFER_NBS = [
    ("exp044 (NB4+ResSSM)", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp044\notebook\nb_blend_tucker_v12_resssm.ipynb"),
    ("exp040", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp040\notebook\nb_blend.ipynb"),
]

def grep_lines(path, patterns):
    if not Path(path).exists():
        return [("MISSING_FILE", "")]
    nb = json.load(open(path, encoding="utf-8"))
    hits = []
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        cid = c.get("id", "?")
        for line in src.split("\n"):
            ls = line.strip()
            if ls.startswith("#"):
                continue
            for pat in patterns:
                if pat.lower() in ls.lower():
                    hits.append((cid, line.strip()[:160]))
                    break
    return hits


# 1. TopN / FCS / file-level post-process
print("=" * 80)
print("CHECK: TopN / FCS / file-level post-process (Task #75)")
print("=" * 80)
patterns = ["FCS", "file_score", "max(prob", "axis=1", "topk", "top_k", "file_confidence"]
for label, path in INFER_NBS:
    print(f"\n--- {label} ---")
    hits = grep_lines(path, patterns)
    for cid, line in hits[:20]:
        print(f"  [{cid}] {line}")


# 2. MixUp / Background mixing / RandomFilter / Noise aug
print("\n" + "=" * 80)
print("CHECK: Aug — MixUp / Background / RandomFilter / Noise (Tasks #77, #78, #79)")
print("=" * 80)
patterns = ["mixup", "MIXUP", "mixup_audio", "background", "BG_", "ESC", "noise", "random_eq",
            "RandomFilter", "filter_aug", "random_pitch", "gain"]
for label, path in TRAIN_NBS:
    print(f"\n--- {label} ---")
    hits = grep_lines(path, patterns)
    for cid, line in hits[:25]:
        print(f"  [{cid}] {line}")


# 3. XC pretrain
print("\n" + "=" * 80)
print("CHECK: XC pretrain (Task #80)")
print("=" * 80)
patterns = ["xeno", "xc_pretrain", "pretrain_xc", "external", "extra_dataset",
            "BC2021", "BC2022", "BC2023", "BC2024", "BC2025", "additional_data"]
for label, path in TRAIN_NBS:
    print(f"\n--- {label} ---")
    hits = grep_lines(path, patterns)
    for cid, line in hits[:15]:
        print(f"  [{cid}] {line}")


# 4. Pseudo label spec audit
print("\n" + "=" * 80)
print("CHECK: Pseudo label (Task #81) — filter / sampling / soft target")
print("=" * 80)
patterns = ["pseudo", "0.5", "max_prob", "filter", "min_conf", "threshold",
            "MIX_RATIO", "SHARES", "binarize", "soft_target", "soft target"]
for label, path in TRAIN_NBS[2:]:  # exp020 R2 + exp029 R3
    print(f"\n--- {label} ---")
    hits = grep_lines(path, patterns)
    for cid, line in hits[:30]:
        print(f"  [{cid}] {line}")
