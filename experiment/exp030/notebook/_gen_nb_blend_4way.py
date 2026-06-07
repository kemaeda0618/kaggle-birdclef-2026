"""Generate exp030 nb_blend_4way: post-process 4-way blend of existing submission CSVs.

Pulls submission.csv from 4 kernel_sources, weighted average, writes submission.csv.

Sources (kernel_sources):
- maekeso/birdclef2026-exp019-blend-w35-40-25       0.85
- maekeso/birdclef2026-exp028-protossm-mirror       0.05
- maekeso/birdclef2026-exp029-r3-single-l1-infer    0.05
- maekeso/birdclef2026-exp020-r2-5fold-infer   0.05
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(src, cid):
    if isinstance(src, str):
        lines = src.split("\n")
        s = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        s = src
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": s}


def md_cell(src, cid):
    if isinstance(src, str):
        lines = src.split("\n")
        s = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        s = src
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": s}


cells = []

cells.append(md_cell(r"""# exp030 4-way post-process blend (lightweight CPU)

Weighted average of 4 existing submission CSVs:

| source | weight | LB single |
|---|---|---|
| exp019 (NB4+Tucker+e17 blend) | 0.85 | 0.947 (主軸) |
| exp028 (ProtoSSM + Sonotype mirror) | 0.05 | 0.919 |
| exp029 (l1 single fold student) | 0.05 | 0.923 |
| exp020 R2 5-fold (NFNet ensemble) | 0.05 | 0.915 |

Goal: 強軸 85% 死守、3 弱軸で diversity 注入。期待 LB 0.946-0.949。

Output: `/kaggle/working/submission.csv`
""", "hdr"))

cells.append(code_cell(r"""# Cell 1: Setup
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd

# Each kernel_source is mounted at /kaggle/input/notebooks/{user}/{slug}/
ROOT = Path("/kaggle/input/notebooks/maekeso")

SOURCES = [
    {"slug": "birdclef2026-exp019-blend-w35-40-25",     "weight": 0.85, "label": "exp019"},
    {"slug": "birdclef2026-exp028-protossm-mirror",     "weight": 0.05, "label": "exp028"},
    {"slug": "birdclef2026-exp029-r3-single-l1-infer",  "weight": 0.05, "label": "exp029"},
    {"slug": "birdclef2026-exp020-r2-5fold-infer", "weight": 0.05, "label": "exp020r2"},
]

# Sanity check weights sum to 1.0
ws = sum(s["weight"] for s in SOURCES)
assert abs(ws - 1.0) < 1e-6, f"Weights sum {ws} != 1.0"
print(f"Weights sum: {ws}")

# Locate submission.csv in each source
for s in SOURCES:
    d = ROOT / s["slug"]
    if not d.exists():
        print(f"  [MISSING] {s['label']:<10s}  {d}")
        continue
    # Find submission*.csv
    cands = sorted(d.glob("submission*.csv"), key=lambda p: -p.stat().st_size)
    if not cands:
        # try root /kaggle/input/notebooks/...
        cands = sorted(d.rglob("submission*.csv"), key=lambda p: -p.stat().st_size)
    if not cands:
        print(f"  [NO CSV] {s['label']:<10s}  {d}")
        continue
    s["csv"] = cands[0]
    print(f"  [OK] {s['label']:<10s}  {cands[0].name}  ({cands[0].stat().st_size/1e6:.1f} MB)")
""", "setup"))

cells.append(code_cell(r"""# Cell 2: Load + align by row_id, compute weighted average
dfs = []
for s in SOURCES:
    if "csv" not in s:
        raise RuntimeError(f"submission CSV not found for {s['label']}")
    df = pd.read_csv(s["csv"])
    df = df.sort_values("row_id").reset_index(drop=True)
    dfs.append((df, s["weight"], s["label"]))
    print(f"{s['label']:<10s}  rows={len(df):>7d}  cols={df.shape[1]}  weight={s['weight']}")

# Verify schema
base_df = dfs[0][0]
base_cols = base_df.columns.tolist()
base_rowids = base_df["row_id"].values
species_cols = [c for c in base_cols if c != "row_id"]
print(f"\nBase: {len(base_rowids)} rows x {len(species_cols)} species")

for df, w, lbl in dfs[1:]:
    assert df.columns.tolist() == base_cols, f"{lbl} cols mismatch"
    assert (df["row_id"].values == base_rowids).all(), f"{lbl} row_id mismatch"

# Weighted average
result = base_df.copy()
result[species_cols] = 0.0
for df, w, lbl in dfs:
    result[species_cols] += w * df[species_cols].values

# Clip to [0, 1] just in case
result[species_cols] = result[species_cols].clip(0.0, 1.0)

# Stats
print(f"\nResult: min={result[species_cols].values.min():.5f}  "
      f"max={result[species_cols].values.max():.5f}  "
      f"mean={result[species_cols].values.mean():.5f}")

out = Path("/kaggle/working/submission.csv")
result.to_csv(out, index=False)
print(f"\nSaved: {out}  ({out.stat().st_size/1e6:.1f} MB)")
print(result.head(2).iloc[:, :8])
""", "blend"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT = HERE / "nb_blend_4way.ipynb"
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {OUT} ({len(cells)} cells)")
