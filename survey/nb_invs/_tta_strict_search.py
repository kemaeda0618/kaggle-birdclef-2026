"""Strict TTA search across public NBs."""
import json, re
from pathlib import Path

NBs = [
    ("kojimar 0.949", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\kojimar_0-949-lb-birdclef-2026-prior-axis-rank-fusion\0-949-lb-birdclef-2026-prior-axis-rank-fusion.ipynb"),
    ("mattiaangeli 0.943", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mattiaangeli_birdclef-2026-0-943-better-blend\birdclef-2026-0-943-better-blend.ipynb"),
    ("mtoshidesu 0.937", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_0-937-birdclef-2026-true-end-to-end-pipeline\0-937-birdclef-2026-true-end-to-end-pipeline.ipynb"),
    ("yuyajk 0.943", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\yuyajk_birdclef-2026-0-943-onnx-perch-sequence-sed-submit\birdclef-2026-0-943-onnx-perch-sequence-sed-submit.ipynb"),
    ("needless090 0.935", r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft\birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft.ipynb"),
]

# Real TTA patterns (not just "TTA" keyword which can match in comments)
patterns = {
    "tta_shifts CFG":         r"tta_shifts\s*[=:]",
    "np.roll":                r"np\.roll\(",
    "TTA_SHIFTS const":       r"TTA_SHIFTS\s*=",
    "temporal_shift_tta fn":  r"def\s+temporal_shift_tta",
    "circular_shift fn":      r"def\s+circular_shift|circular_shift\(",
    "test_time_aug fn":       r"def\s+test_time_aug",
    "sliding_window fn":      r"def\s+sliding_window|sliding_window\(",
    "stride < window":        r"stride\s*=\s*(2\.5|2|2500|80000)",  # 2.5s stride suggests overlap
}

for name, path in NBs:
    p = Path(path)
    if not p.exists():
        print(f"--- {name}: FILE MISSING ---"); continue
    nb = json.load(open(p, encoding="utf-8"))
    print(f"--- {name} ({len(nb['cells'])} cells) ---")
    found_any = False
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        for label, pat in patterns.items():
            matches = re.findall(pat, src)
            if matches:
                found_any = True
                # find first occurrence + 2 lines context
                m = re.search(pat, src)
                if m:
                    start = max(0, src.rfind("\n", 0, m.start()) + 1)
                    end = src.find("\n", m.end() + 50)
                    if end < 0: end = m.end() + 100
                    context = src[start:end].strip()[:150]
                    print(f"  Cell {i}: {label} → {context}")
    if not found_any:
        print(f"  (no real TTA pattern found)")
    print()
