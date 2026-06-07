"""Generate exp059: exp048 + exp037 v313 blend (post-PP CSV merge).

Kernel sources:
  - maekeso/birdclef2026-exp048-topn1-blend (LB 0.950)
  - maekeso/birdclef2026-exp037-perch-lightprotossm-mlp-resssm (exp037 v313, LB 0.930)

Logic: weighted average (w_exp048=0.80, w_exp037=0.20)
exp037 v313 (LB 0.930) は exp020 R2 (LB 0.915) より強いので weight やや高い
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp059\notebook\nb_blend_a.ipynb")


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# exp059 — Plan A: exp048 + exp037 v313 4-way blend (post-PP merge)

**Kernel sources**:
- `maekeso/birdclef2026-exp048-topn1-blend` (LB 0.950)
- `maekeso/birdclef2026-exp037-perch-lightprotossm-mlp-resssm` (exp037 v313 LB 0.930)

**Blend**: weighted average of post-PP submission.csv
- w_exp048 = 0.80
- w_exp037 = 0.20

**Expected LB**: 0.952-0.954 (+0.002-0.004)
exp037 v313 (LB 0.930) = 我々 2nd best single model、blend に未活用
"""),

        code_cell("imports", """import os
from pathlib import Path
import numpy as np
import pandas as pd

EXP048_DIR_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp048-topn1-blend"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp048-topn1-blend"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp048-topn1-blend"),
]
EXP037_DIR_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp037-perch-lightprotossm-mlp-resssm"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp037-perch-lightprotossm-mlp-resssm"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp037-perch-lightprotossm-mlp-resssm"),
]

def find_submission(candidates, name):
    for d in candidates:
        if d.exists():
            sub = next(d.rglob("submission.csv"), None)
            if sub is not None:
                print(f"  {name}: found at {sub}")
                return sub
    for sub in Path("/kaggle/input").rglob("submission.csv"):
        if name.lower() in str(sub).lower():
            print(f"  {name}: fallback at {sub}")
            return sub
    raise FileNotFoundError(f"{name} submission.csv not found")

EXP048_SUB = find_submission(EXP048_DIR_CANDIDATES, "exp048")
EXP037_SUB = find_submission(EXP037_DIR_CANDIDATES, "exp037")

print(f"\\nexp048: {EXP048_SUB}")
print(f"exp037: {EXP037_SUB}")
"""),

        code_cell("load", """sub_exp048 = pd.read_csv(EXP048_SUB)
sub_exp037 = pd.read_csv(EXP037_SUB)

print(f"exp048 shape: {sub_exp048.shape}")
print(f"exp037 shape: {sub_exp037.shape}")
print(f"exp048 columns[:5]: {sub_exp048.columns.tolist()[:5]}")
print(f"exp037 columns[:5]: {sub_exp037.columns.tolist()[:5]}")

assert sub_exp048.shape == sub_exp037.shape, "shape mismatch"
assert (sub_exp048['row_id'].values == sub_exp037['row_id'].values).all(), "row_id mismatch"
assert list(sub_exp048.columns) == list(sub_exp037.columns), "columns mismatch"
print("\\n✓ alignment OK")
"""),

        code_cell("blend", """W_EXP048 = 0.80
W_EXP037 = 0.20
assert abs(W_EXP048 + W_EXP037 - 1.0) < 1e-6

class_cols = [c for c in sub_exp048.columns if c != 'row_id']
print(f"Blending {len(class_cols)} species columns")

blended = sub_exp048.copy()
for col in class_cols:
    blended[col] = W_EXP048 * sub_exp048[col].values + W_EXP037 * sub_exp037[col].values

print(f"\\nexp048 prob mean: {sub_exp048[class_cols].values.mean():.4f}, max: {sub_exp048[class_cols].values.max():.4f}")
print(f"exp037 prob mean: {sub_exp037[class_cols].values.mean():.4f}, max: {sub_exp037[class_cols].values.max():.4f}")
print(f"blend  prob mean: {blended[class_cols].values.mean():.4f}, max: {blended[class_cols].values.max():.4f}")
"""),

        code_cell("save", """OUT_PATH = "submission.csv"
blended.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print(f"  size: {os.path.getsize(OUT_PATH)/1e6:.2f} MB")
print(f"  rows: {len(blended)}")

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
