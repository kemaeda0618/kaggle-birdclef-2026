"""Audit v2: examine actual context lines around key patterns."""
import json, re
from pathlib import Path

PUBLIC_NBS = [
    ("needless090 0.935 ft", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft\birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft.ipynb"),
    ("needless090 0.934 s", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s\birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s.ipynb"),
    ("mtoshidesu 0.941", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_birdclef-2026-0-941-public-lb-onnx-perch-pr\birdclef-2026-0-941-public-lb-onnx-perch-pr.ipynb"),
    ("mtoshidesu 0.937", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_0-937-birdclef-2026-true-end-to-end-pipeline\0-937-birdclef-2026-true-end-to-end-pipeline.ipynb"),
    ("kojimar 0.949", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\kojimar_0-949-lb-birdclef-2026-prior-axis-rank-fusion\0-949-lb-birdclef-2026-prior-axis-rank-fusion.ipynb"),
    ("marynaborovska 0.927 protossm-v5", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\marynaborovska_birdclef-26-protossm-v5-resboosting-0-927-lb\birdclef-26-protossm-v5-resboosting-0-927-lb.ipynb"),
]

# Specific patterns (with context required)
SPEC_CHECKS = [
    ("Chunk filter (max_prob > threshold drops chunks)", [
        r"primary_label_prob\s*>",
        r"max\(.*\).*>\s*0\.\d+.*drop",
        r"PSEUDO_THRESHOLD",
        r"PSEUDO_MIN_CONF",
        r"min_chunk_prob",
        r"\.max\([^)]*\)\s*>=?\s*0\.\d+",
    ]),
    ("Class zero-out (prob < t → 0)", [
        r"trim_min_prob",
        r"pseudo_probs\[.*<.*\]\s*=\s*0",
        r"prob\[.*<.*\]\s*=\s*0",
        r"np\.where\(.*prob.*<.*0",
        r"PROB_FLOOR",
        r"clip.*prob",
    ]),
    ("Conditional pseudo sampling (40/60 / pseudo_birds)", [
        r"sampling_prob",
        r"pseudo_birds",
        r"if .* in .*pseudo",
        r"if .*pseudo.*has",
        r"conditional.*pseudo",
    ]),
    ("MixUp alpha=None / sum aggregation", [
        r"alpha['\"]?\s*[:=]\s*None",
        r"target_aggregation",
        r"target\s*\+\s*mixup_target",
        r"\(mixup_wave\s*\+\s*wave\)\s*/\s*2",
    ]),
    ("OOF pseudo (separate path with fold_id)", [
        r"oof_df_path",
        r"oof_pseudo",
        r"\.fold_id",
        r"'fold_id'",
        r"\"fold_id\"",
    ]),
]

for label, path in PUBLIC_NBS:
    if not Path(path).exists():
        continue
    nb = json.load(open(path, encoding="utf-8"))
    all_src = ""
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            all_src += "".join(c.get("source", [])) + "\n"

    print(f"\n=== {label} ===")
    for spec, patterns in SPEC_CHECKS:
        hits = []
        for p in patterns:
            try:
                m = re.search(p, all_src)
                if m:
                    # Get context line
                    start = max(0, m.start() - 50)
                    end = min(len(all_src), m.end() + 80)
                    snippet = all_src[start:end].replace("\n", " | ")
                    hits.append((p, snippet[:200]))
            except: pass
        if hits:
            print(f"  ✓ {spec}:")
            for p, snip in hits[:1]:
                print(f"      [{p}]")
                print(f"      ...{snip}...")
        else:
            print(f"  — {spec}: no match")
