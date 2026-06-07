"""Generate exp060: exp048 + exp020 R2 5-fold blend (post-PP CSV merge).

Kernel sources:
  - maekeso/birdclef2026-exp048-topn1-blend (LB 0.950)
  - maekeso/birdclef2026-exp020-r2-5fold-infer (LB 0.915)

Logic: 加重平均 (w_exp048=0.85, w_exp020=0.15)
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp060\notebook\nb_blend_b.ipynb")


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# exp060 — Plan B: exp048 + exp020 R2 4-way blend (post-PP merge)

**Kernel sources**:
- `maekeso/birdclef2026-exp048-topn1-blend` (LB 0.950)
- `maekeso/birdclef2026-exp020-r2-5fold-infer` (LB 0.915)

**Blend**: weighted average of post-PP submission.csv
- w_exp048 = 0.85
- w_exp020 = 0.15

**Expected LB**: 0.951-0.953 (+0.001-0.003)
"""),

        code_cell("imports", """import os
from pathlib import Path
import numpy as np
import pandas as pd

# Locate kernel_source outputs
EXP048_DIR_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp048-topn1-blend"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp048-topn1-blend"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp048-topn1-blend"),
]
EXP020_DIR_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp020-r2-5fold-infer"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp020-r2-5fold-infer"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp020-r2-5fold-infer"),
]

def find_submission(candidates, name):
    for d in candidates:
        if d.exists():
            sub = next(d.rglob("submission.csv"), None)
            if sub is not None:
                print(f"  {name}: found at {sub}")
                return sub
    # fallback: scan /kaggle/input
    for sub in Path("/kaggle/input").rglob("submission.csv"):
        if name.lower() in str(sub).lower():
            print(f"  {name}: fallback at {sub}")
            return sub
    raise FileNotFoundError(f"{name} submission.csv not found in any candidate dir")

EXP048_SUB = find_submission(EXP048_DIR_CANDIDATES, "exp048")
EXP020_SUB = find_submission(EXP020_DIR_CANDIDATES, "exp020")

print(f"\\nexp048: {EXP048_SUB}")
print(f"exp020: {EXP020_SUB}")
"""),

        code_cell("load", """sub_exp048 = pd.read_csv(EXP048_SUB)
sub_exp020 = pd.read_csv(EXP020_SUB)

print(f"exp048 shape: {sub_exp048.shape}")
print(f"exp020 shape: {sub_exp020.shape}")
print(f"exp048 columns[:5]: {sub_exp048.columns.tolist()[:5]}")
print(f"exp020 columns[:5]: {sub_exp020.columns.tolist()[:5]}")

# Validate alignment
assert sub_exp048.shape == sub_exp020.shape, f"shape mismatch"
assert (sub_exp048['row_id'].values == sub_exp020['row_id'].values).all(), "row_id mismatch"
assert list(sub_exp048.columns) == list(sub_exp020.columns), "columns mismatch"
print("\\n✓ alignment OK")
"""),

        code_cell("blend", """W_EXP048 = 0.85
W_EXP020 = 0.15
assert abs(W_EXP048 + W_EXP020 - 1.0) < 1e-6, "weights don't sum to 1"

# Weighted average for species columns
class_cols = [c for c in sub_exp048.columns if c != 'row_id']
print(f"Blending {len(class_cols)} species columns")

blended = sub_exp048.copy()
for col in class_cols:
    blended[col] = W_EXP048 * sub_exp048[col].values + W_EXP020 * sub_exp020[col].values

# Stats
print(f"\\nexp048 prob mean: {sub_exp048[class_cols].values.mean():.4f}, max: {sub_exp048[class_cols].values.max():.4f}")
print(f"exp020 prob mean: {sub_exp020[class_cols].values.mean():.4f}, max: {sub_exp020[class_cols].values.max():.4f}")
print(f"blend  prob mean: {blended[class_cols].values.mean():.4f}, max: {blended[class_cols].values.max():.4f}")
"""),

        code_cell("save", """OUT_PATH = "submission.csv"
blended.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print(f"  size: {os.path.getsize(OUT_PATH)/1e6:.2f} MB")
print(f"  rows: {len(blended)}")

# Sanity check
import pandas as pd
loaded = pd.read_csv(OUT_PATH)
print(f"\\nLoaded back: shape={loaded.shape}")
print(loaded.head(3))
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
