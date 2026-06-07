"""exp033: Wet/Dry season scaling — exp027 NB fork + post-process cell.

Kernel_source chain は Kernels Only 競技で preview-only (3 行) しか流れない問題対応。
exp027 NB の全 cells を copy + 末尾に wet/dry post-process cell 追加。
Submit 時に full test (~700 行) で exp027 推論実行 → post-process → submission.csv 上書き。
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
EXP027_NB_PATH = HERE.parent.parent / "exp027" / "notebook" / "nb_protossm_only.ipynb"


def code_cell(src, cid):
    lines = src.split("\n")
    s = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cid, "metadata": {}, "outputs": [], "execution_count": None, "source": s}


def md_cell(src, cid):
    lines = src.split("\n")
    s = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": s}


# Load exp027 NB
exp027_nb = json.loads(EXP027_NB_PATH.read_text(encoding="utf-8"))
cells = list(exp027_nb["cells"])
print(f"Loaded exp027 NB: {len(cells)} cells")

# Append Wet/Dry post-process cells
cells.append(md_cell(r"""# === exp033 post-process: Wet/Dry season scaling ===
Pantanal 季節性: wet (11-4 月) で Insecta/Amphibia 活性 ↑、dry (5-10 月) で 活性 ↓
""", "exp033-hdr"))

cells.append(code_cell(r"""# exp033: Wet/Dry scaling
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Reload exp027 submission (already written by previous cells)
sub_path_pp = Path("submission.csv")  # CWD = /kaggle/working/
assert sub_path_pp.exists(), "exp027 submission.csv must exist"
sub_pp = pd.read_csv(sub_path_pp)
print(f"exp033 input sub: {sub_pp.shape}")

# Parse month from row_id
DATE_RE = re.compile(r"_S\d{2}_(\d{8})_")
def parse_month(rid):
    m = DATE_RE.search(rid)
    return int(m.group(1)[4:6]) if m else None

months = sub_pp["row_id"].apply(parse_month)
print(f"Month dist:\n{months.value_counts().sort_index()}")
WET_MONTHS = {11, 12, 1, 2, 3, 4}
is_wet = months.isin(WET_MONTHS).values

# Target species
class_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
PRIMARY_LABELS_PP = [c for c in sub_pp.columns if c != "row_id"]
target_species = [sp for sp in PRIMARY_LABELS_PP if class_map.get(sp) in ("Insecta", "Amphibia")]
print(f"Target species (Insecta + Amphibia): {len(target_species)}")

# Scale
WET_BOOST, DRY_SCALE = 1.10, 0.90
target_arr = sub_pp[target_species].values
scale_arr = np.where(is_wet[:, None], WET_BOOST, DRY_SCALE)
sub_pp[target_species] = (target_arr * scale_arr).clip(0.0, 1.0)
sub_pp[PRIMARY_LABELS_PP] = sub_pp[PRIMARY_LABELS_PP].clip(0.0, 1.0)

# Overwrite submission.csv
sub_pp.to_csv("submission.csv", index=False)
print(f"\nexp033 wet/dry applied + saved: {sub_pp.shape}")
print(f"  before mean (target species): {target_arr.mean():.5f}")
print(f"  after  mean: {sub_pp[target_species].values.mean():.5f}")
print(sub_pp.head(3).iloc[:, :8])
""", "exp033-tweak"))

nb_out = {
    "cells": cells,
    "metadata": exp027_nb.get("metadata", {}),
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_wet_dry.ipynb"
out_path.write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")
