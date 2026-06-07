"""exp034: Day-of-Year prior — exp027 NB fork + post-process cell."""
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


exp027_nb = json.loads(EXP027_NB_PATH.read_text(encoding="utf-8"))
cells = list(exp027_nb["cells"])

cells.append(md_cell(r"""# === exp034 post-process: Day-of-Year species prior ===
labeled_ss から day_of_year × species presence rate 集計 → test row date に応じて prior multiply
""", "exp034-hdr"))

cells.append(code_cell(r"""# exp034: Day-of-Year prior
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

sub_path_pp = Path("submission.csv")
assert sub_path_pp.exists(), "exp027 submission.csv must exist"
sub_pp = pd.read_csv(sub_path_pp)
PRIMARY_LABELS_PP = [c for c in sub_pp.columns if c != "row_id"]
print(f"exp034 input sub: {sub_pp.shape}")

# Load labeled SS for DOY prior
labeled_ss = pd.read_csv(BASE / "train_soundscapes_labels.csv")
DATE_RE = re.compile(r"_S\d{2}_(\d{8})_")
def parse_doy(fn):
    m = DATE_RE.search(fn)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").timetuple().tm_yday
    return None
labeled_ss["doy"] = labeled_ss["filename"].apply(parse_doy)

# Build presence table (366, n_species)
label_to_idx = {sp: i for i, sp in enumerate(PRIMARY_LABELS_PP)}
presence = np.zeros((366, len(PRIMARY_LABELS_PP)), dtype=np.float32)
counts = np.zeros(366, dtype=np.float32)
for _, row in labeled_ss.iterrows():
    doy = row["doy"]
    if doy is None or doy < 1 or doy > 366:
        continue
    counts[doy - 1] += 1
    for sp in str(row["primary_label"]).split(";"):
        sp = sp.strip()
        if sp in label_to_idx:
            presence[doy - 1, label_to_idx[sp]] += 1
rate = presence / np.where(counts > 0, counts, 1)[:, None]

# Circular Gaussian smooth (sigma=14 day)
SIGMA = 14
rate_3x = np.concatenate([rate, rate, rate], axis=0)
smoothed_3x = gaussian_filter1d(rate_3x, sigma=SIGMA, axis=0, mode='constant')
rate_smoothed = smoothed_3x[366:732]

global_mean = rate.mean(axis=0)
relative_prior = (rate_smoothed / (global_mean[None, :] + 1e-6)).clip(0.5, 2.0)

# Apply to sub
def rowid_to_doy(rid):
    m = DATE_RE.search(rid)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").timetuple().tm_yday
    return None
doy_arr = sub_pp["row_id"].apply(rowid_to_doy)
print(f"sub DOY range: {doy_arr.min()}-{doy_arr.max()}")

ALPHA = 0.3
species_arr = sub_pp[PRIMARY_LABELS_PP].values.copy()
species_arr_new = species_arr.copy()
for i, doy in enumerate(doy_arr):
    if doy is not None and 1 <= doy <= 366:
        species_arr_new[i] *= relative_prior[doy - 1]
species_arr_final = ((1 - ALPHA) * species_arr + ALPHA * species_arr_new).clip(0.0, 1.0)
sub_pp[PRIMARY_LABELS_PP] = species_arr_final

sub_pp.to_csv("submission.csv", index=False)
print(f"\nexp034 DOY prior applied + saved: {sub_pp.shape}")
print(f"  before mean: {species_arr.mean():.5f}")
print(f"  after mean:  {sub_pp[PRIMARY_LABELS_PP].values.mean():.5f}")
print(sub_pp.head(3).iloc[:, :8])
""", "exp034-tweak"))

nb_out = {
    "cells": cells,
    "metadata": exp027_nb.get("metadata", {}),
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_doy.ipynb"
out_path.write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")
