"""Cross-check: do other BC2025/2026 top solutions use the same pseudo spec items?"""
import json
from pathlib import Path

PUBLIC_NBS = [
    ("needless090 iter-pseudo 0.935", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft\birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft.ipynb"),
    ("needless090 iter-pseudo 0.934 s", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s\birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s.ipynb"),
    ("mtoshidesu 0.941", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_birdclef-2026-0-941-public-lb-onnx-perch-pr\birdclef-2026-0-941-public-lb-onnx-perch-pr.ipynb"),
    ("mtoshidesu 0.937", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_0-937-birdclef-2026-true-end-to-end-pipeline\0-937-birdclef-2026-true-end-to-end-pipeline.ipynb"),
    ("kojimar 0.949", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\kojimar_0-949-lb-birdclef-2026-prior-axis-rank-fusion\0-949-lb-birdclef-2026-prior-axis-rank-fusion.ipynb"),
    ("ttyn4519 perch-sed 0.943", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\ttyn4519_perch-task236-task233-snowflake-sed-w003\perch-task236-task233-snowflake-sed-w003.ipynb"),
    ("mattiaangeli 0.943", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mattiaangeli_birdclef-2026-0-943-better-blend\birdclef-2026-0-943-better-blend.ipynb"),
    ("marynaborovska 0.927 protossm-v5-resboosting", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\marynaborovska_birdclef-26-protossm-v5-resboosting-0-927-lb\birdclef-26-protossm-v5-resboosting-0-927-lb.ipynb"),
]

# 4 spec items to check
PATTERNS = {
    "1. Chunk filter (max_prob)": ["primary_label_prob", "max_prob", "min_prob", "MIN_PROB", "PSEUDO_THRESHOLD", "0.5"],
    "2. Class zero-out (<0.1)": ["trim_min_prob", "zero.*out", "prob.*0\\.1", "MIN_CONF", "TRIM"],
    "3. Conditional sampling (40/60)": ["sampling_prob", "pseudo_birds", "binomial.*0\\.4", "conditional"],
    "4. MixUp alpha=None": ["alpha.*None", "alpha.None", "(mixup_wave + wave) / 2", "target.*sum.*clamp"],
    "5. OOF pseudo": ["oof_df", "oof_pseudo", "fold_id", "OOF_"],
}

for label, path in PUBLIC_NBS:
    if not Path(path).exists():
        continue
    nb = json.load(open(path, encoding="utf-8"))
    all_src = ""
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            all_src += "".join(c.get("source", [])) + "\n"

    print(f"\n=== {label} ===")
    for spec, patterns in PATTERNS.items():
        hits = []
        for p in patterns:
            import re
            try:
                if re.search(p, all_src):
                    hits.append(p)
            except: pass
        status = "✓" if hits else "—"
        print(f"  {status} {spec}: hits = {hits[:2] if hits else '(none)'}")
