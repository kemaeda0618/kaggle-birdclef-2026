"""exp035: Cross-file Site smoothing — exp027 NB fork + post-process cell."""
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

cells.append(md_cell(r"""# === exp035 post-process: Cross-file Site smoothing ===
同一 site の test files 横断で species top-K mean を計算 → row-level prediction を site-prior と blend
""", "exp035-hdr"))

cells.append(code_cell(r"""# exp035: Site smoothing
import re
import numpy as np
import pandas as pd
from pathlib import Path

sub_path_pp = Path("submission.csv")
assert sub_path_pp.exists(), "exp027 submission.csv must exist"
sub_pp = pd.read_csv(sub_path_pp)
PRIMARY_LABELS_PP = [c for c in sub_pp.columns if c != "row_id"]
print(f"exp035 input sub: {sub_pp.shape}")

# Parse site
SITE_RE = re.compile(r"_(S\d{2})_")
sub_pp["site"] = sub_pp["row_id"].apply(lambda r: (SITE_RE.search(r).group(1) if SITE_RE.search(r) else None))
print(f"Site dist:\n{sub_pp['site'].value_counts()}")

# Build site-level top-K mean per species
TOP_K = 5
site_priors = {}
for site, group in sub_pp.groupby("site"):
    if site is None:
        continue
    arr = group[PRIMARY_LABELS_PP].values
    if arr.shape[0] >= TOP_K:
        sorted_arr = np.sort(arr, axis=0)
        topk_mean = sorted_arr[-TOP_K:].mean(axis=0)
    else:
        topk_mean = arr.mean(axis=0) if arr.shape[0] > 0 else np.zeros(len(PRIMARY_LABELS_PP))
    site_priors[site] = topk_mean

print(f"site_priors: {len(site_priors)} sites")

# Apply
ALPHA = 0.4
EPS = 1e-6
species_arr = sub_pp[PRIMARY_LABELS_PP].values.copy()
for site, group_idx in sub_pp.groupby("site").groups.items():
    if site is None or site not in site_priors:
        continue
    prior = site_priors[site]
    scale = 1.0 - ALPHA + ALPHA * (prior + EPS) / (prior.max() + EPS)
    for ri in group_idx:
        species_arr[ri] *= scale

species_arr = np.clip(species_arr, 0.0, 1.0)
sub_pp[PRIMARY_LABELS_PP] = species_arr
sub_pp = sub_pp.drop(columns=["site"])

sub_pp.to_csv("submission.csv", index=False)
print(f"\nexp035 site smoothing applied + saved: {sub_pp.shape}")
print(f"  after mean: {sub_pp[PRIMARY_LABELS_PP].values.mean():.5f}")
print(sub_pp.head(3).iloc[:, :8])
""", "exp035-tweak"))

nb_out = {
    "cells": cells,
    "metadata": exp027_nb.get("metadata", {}),
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_site_smooth.ipynb"
out_path.write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")
