"""Generate blend notebook: gate-fake008-repro + NB4."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, lines):
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, lines):
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr", [
    "# Blend: gate-fake008-repro + NB4",
    "",
    "Rank blend of two complementary submissions:",
    "- gate-fake008-repro (LB 0.943): LightProtoSSM 60% + Tucker SED 40%",
    "- NB4 v5 (LB 0.923): ProtoSSM 50% + MLP 50% + retrieval + Prior Tables",
]))

cells.append(code_cell("setup", [
    "import os, time",
    "import numpy as np",
    "import pandas as pd",
    "from pathlib import Path",
    "from scipy.stats import rankdata",
    "",
    "START = time.time()",
    "",
    "GATE_W = 0.70",
    "NB4_W  = 0.30",
    "",
    "COMP_DIR = None",
    "for cand in [Path('/kaggle/input/competitions/birdclef-2026'), Path('/kaggle/input/birdclef-2026')]:",
    "    if cand.exists():",
    "        COMP_DIR = cand; break",
    "assert COMP_DIR, 'birdclef-2026 not mounted'",
    "",
    "GATE_DIR = Path('/kaggle/input/notebooks/maekeso/birdclef2026-gate-fake008-repro')",
    "NB4_DIR  = Path('/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb4-blend-protossm-mlp')",
    "",
    "print('GATE_DIR exists:', GATE_DIR.exists())",
    "print('NB4_DIR exists: ', NB4_DIR.exists())",
    "if GATE_DIR.exists(): print('  files:', sorted(os.listdir(GATE_DIR)))",
    "if NB4_DIR.exists():  print('  files:', sorted(os.listdir(NB4_DIR)))",
]))

cells.append(code_cell("load", [
    "gate_df = pd.read_csv(GATE_DIR / 'submission.csv').set_index('row_id')",
    "nb4_df  = pd.read_csv(NB4_DIR  / 'submission.csv').set_index('row_id')",
    "",
    "sample_sub = pd.read_csv(COMP_DIR / 'sample_submission.csv')",
    "row_ids = sample_sub['row_id'].tolist()",
    "species = sample_sub.columns[1:].tolist()",
    "",
    "gate_df = gate_df.loc[row_ids, species]",
    "nb4_df  = nb4_df.loc[row_ids, species]",
    "",
    "print(f'gate shape: {gate_df.shape}')",
    "print(f'nb4  shape: {nb4_df.shape}')",
    "print(f'species: {len(species)}')",
]))

cells.append(code_cell("blend", [
    "gate_arr = gate_df.values.astype(np.float32)",
    "nb4_arr  = nb4_df.values.astype(np.float32)",
    "",
    "def rank_norm(arr):",
    "    out = np.zeros_like(arr)",
    "    for j in range(arr.shape[1]):",
    "        out[:, j] = rankdata(arr[:, j]) / arr.shape[0]",
    "    return out",
    "",
    "gate_rank = rank_norm(gate_arr)",
    "nb4_rank  = rank_norm(nb4_arr)",
    "",
    "blended = GATE_W * gate_rank + NB4_W * nb4_rank",
    "",
    "print(f'Blend weights: gate={GATE_W}, nb4={NB4_W}')",
    "print(f'blended mean={blended.mean():.4f}, max={blended.max():.4f}')",
]))

cells.append(code_cell("submit", [
    "sub_df = pd.DataFrame(blended, index=row_ids, columns=species)",
    "sub_df.index.name = 'row_id'",
    "sub_df = sub_df.reset_index()",
    "",
    "assert list(sub_df['row_id']) == row_ids",
    "assert sub_df.shape == (len(row_ids), len(species) + 1)",
    "",
    "sub_df.to_csv('/kaggle/working/submission.csv', index=False)",
    "print(f'Written {len(sub_df)} rows to /kaggle/working/submission.csv')",
    "print(f'Wall time: {(time.time()-START)/60:.1f} min')",
    "sub_df.head(3)",
]))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_blend_gate_nb4.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
