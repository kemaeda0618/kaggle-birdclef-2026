"""Build exp080a: 3-stream adaptive blend NB.

Inputs (kernel_sources):
  - maekeso/birdclef2026-exp069b-nb4-pseudo    → nb4_raw_234.npz
  - maekeso/birdclef2026-exp069b-tucker-pseudo → tucker_raw_234.npz
  - maekeso/birdclef2026-exp069b-exp029-pseudo → exp029_raw_234.npz

Process:
  1. Load 3 raw stream NPZ
  2. Verify alignment (shape, file_ids, no NaN)
  3. Detect NB4 dead species (max < 0.05 across all chunks)
  4. Adaptive weight:
     - NB4 active (129 sp): (W_NB4, W_TUCKER, W_EXP029) = (0.30, 0.40, 0.30)
     - NB4 dead (105 sp):   (0.00, 0.571, 0.429) — renormalize remaining 2
  5. Compute blend = w_nb4 * p_n + w_tuc * p_t + w_e29 * p_e
  6. Save pseudo_adaptive_234.npz + metadata

Output → /kaggle/working/:
  - pseudo_adaptive_234.npz (probs (float16) + file_ids)
  - adaptive_meta.json (blend config, dead/active sp count)
  - primary_labels.json (234 sp ordering)
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_blend_adaptive.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# Cell 0: Title
cells.append(md_cell("""# exp080a: 3-stream Adaptive Blend Pseudo Gen

**Inputs (kernel_sources)**:
  - exp069b-nb4-pseudo: NB4 raw probs (10658 x 12 x 234)
  - exp069b-tucker-pseudo: Tucker SED raw probs
  - exp069b-exp029-pseudo: exp029 R3 raw probs

**Process**:
  1. Load 3 raw NPZ, verify alignment
  2. Detect NB4 dead species (max < 0.05 anywhere) → 105 sp expected
  3. Adaptive weight per species:
     - Active: (NB4=0.30, Tucker=0.40, exp029=0.30)
     - Dead:   (NB4=0.00, Tucker=0.571, exp029=0.429)
  4. Compute weighted blend (probability preserved, NO rank-pct, NO transform, NO filter)
  5. Save pseudo_adaptive_234.npz

**Output → /kaggle/working/**:
  - pseudo_adaptive_234.npz
  - adaptive_meta.json
  - primary_labels.json
"""))

# Cell 1: Setup + paths
cells.append(code_cell("""# Setup
import sys, os, time, json
from pathlib import Path
import numpy as np
import pandas as pd

print(f"Python: {sys.version[:50]}")
print(f"numpy: {np.__version__}")
START = time.time()

def find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None

# Locate kernel_source inputs (auto-detect Kaggle path patterns)
NB4_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069b-nb4-pseudo",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069b-nb4-pseudo",
])
TUCKER_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069b-tucker-pseudo",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069b-tucker-pseudo",
])
EXP029_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069b-exp029-pseudo",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069b-exp029-pseudo",
])
assert NB4_DIR is not None, "NB4 kernel source not attached"
assert TUCKER_DIR is not None, "Tucker kernel source not attached"
assert EXP029_DIR is not None, "exp029 kernel source not attached"
print(f"NB4 dir:    {NB4_DIR}")
print(f"Tucker dir: {TUCKER_DIR}")
print(f"exp029 dir: {EXP029_DIR}")

DATA_PATH = find_dir([
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
])
assert DATA_PATH is not None
TAXONOMY_CSV = DATA_PATH / "taxonomy.csv"

OUT_DIR = Path("/kaggle/working")
"""))

# Cell 2: Load 3 stream NPZ with verification
cells.append(code_cell("""# Load 3 raw stream NPZ with rigorous verification
print("=== Load 3 raw stream NPZ ===")

def load_stream(dir_path, expected_fname):
    p = dir_path / expected_fname
    if not p.exists():
        candidates = list(dir_path.glob("*.npz"))
        assert candidates, f"No NPZ in {dir_path}"
        p = candidates[0]
    print(f"  Loading {p}")
    npz = dict(np.load(p, allow_pickle=True))
    return npz

nb4_data    = load_stream(NB4_DIR, "nb4_raw_234.npz")
tucker_data = load_stream(TUCKER_DIR, "tucker_raw_234.npz")
exp029_data = load_stream(EXP029_DIR, "exp029_raw_234.npz")

p_n = nb4_data["probs"].astype(np.float32)
p_t = tucker_data["probs"].astype(np.float32)
p_e = exp029_data["probs"].astype(np.float32)

f_n = nb4_data["file_ids"]
f_t = tucker_data["file_ids"]
f_e = exp029_data["file_ids"]

# Shape verification
print(f"\\nShape verification:")
print(f"  NB4:    {p_n.shape}, dtype={p_n.dtype}")
print(f"  Tucker: {p_t.shape}, dtype={p_t.dtype}")
print(f"  exp029: {p_e.shape}, dtype={p_e.dtype}")
assert p_n.shape == p_t.shape == p_e.shape, "Shape mismatch across streams"
N_FILES, N_WIN, N_CLS = p_n.shape
assert N_CLS == 234, f"Expected 234 classes, got {N_CLS}"
print(f"  N_FILES={N_FILES}, N_WIN={N_WIN}, N_CLS={N_CLS}")

# File ID alignment
print(f"\\nFile ID alignment:")
assert np.array_equal(f_n, f_t), "NB4 vs Tucker file_ids mismatch"
assert np.array_equal(f_n, f_e), "NB4 vs exp029 file_ids mismatch"
print(f"  All 3 streams aligned: True")
file_ids = f_n

# NaN/Inf check
print(f"\\nNaN/Inf check:")
for name, p in [("NB4", p_n), ("Tucker", p_t), ("exp029", p_e)]:
    n_nan = int(np.isnan(p).sum())
    n_inf = int(np.isinf(p).sum())
    print(f"  {name}: NaN={n_nan}, Inf={n_inf}, min={p.min():.5f}, max={p.max():.5f}")
    assert n_nan == 0 and n_inf == 0, f"{name} has NaN or Inf"
    assert (p >= 0).all() and (p <= 1.0001).all(), f"{name} probs out of [0,1] range"
print(f"  All streams clean")

# Distribution stats
print(f"\\nDistribution stats per stream:")
for name, p in [("NB4", p_n), ("Tucker", p_t), ("exp029", p_e)]:
    print(f"  {name}: mean={p.mean():.5f}, median={np.median(p):.5f}, max={p.max():.5f}, >0.5 frac={(p>0.5).sum()/p.size:.5f}")
"""))

# Cell 3: Detect NB4 dead species
cells.append(code_cell("""# Detect NB4 dead species (per-species check)
DEAD_THRESHOLD = 0.05
print(f"=== Detect NB4 dead species (max < {DEAD_THRESHOLD}) ===")

nb4_sp_max = p_n.max(axis=(0, 1))  # (234,)
nb4_active_mask = (nb4_sp_max >= DEAD_THRESHOLD).astype(np.float32)  # 1=active, 0=dead

active_idx = np.where(nb4_active_mask == 1)[0]
dead_idx = np.where(nb4_active_mask == 0)[0]
n_active = len(active_idx)
n_dead = len(dead_idx)

print(f"  NB4 active species: {n_active}/234")
print(f"  NB4 dead species:   {n_dead}/234")

# Class breakdown
taxo = pd.read_csv(TAXONOMY_CSV)
dead_taxo = taxo.iloc[dead_idx]
active_taxo = taxo.iloc[active_idx]
print(f"\\n  Dead sp class breakdown: {dead_taxo['class_name'].value_counts().to_dict()}")
print(f"  Active sp class breakdown: {active_taxo['class_name'].value_counts().to_dict()}")

# Sanity: Tucker / exp029 still cover dead species
print(f"\\n  Tucker on dead species: max={p_t[:,:,dead_idx].max():.4f}, >0.5 frac={(p_t[:,:,dead_idx]>0.5).sum()/p_t[:,:,dead_idx].size:.5f}")
print(f"  exp029 on dead species: max={p_e[:,:,dead_idx].max():.4f}, >0.5 frac={(p_e[:,:,dead_idx]>0.5).sum()/p_e[:,:,dead_idx].size:.5f}")
print(f"  → Dead sp coverage by Tucker/exp029 still meaningful")
"""))

# Cell 4: Compute adaptive blend
cells.append(code_cell("""# Adaptive weight blend
W_NB4_ACTIVE = 0.30
W_TUCKER_ACTIVE = 0.40
W_EXP029_ACTIVE = 0.30

# Dead sp renormalize 残り 2 stream
W_NB4_DEAD = 0.00
W_TUCKER_DEAD = W_TUCKER_ACTIVE / (W_TUCKER_ACTIVE + W_EXP029_ACTIVE)  # 0.5714
W_EXP029_DEAD = W_EXP029_ACTIVE / (W_TUCKER_ACTIVE + W_EXP029_ACTIVE)  # 0.4286

print(f"=== Weight scheme ===")
print(f"  Active (NB4 fires): NB4={W_NB4_ACTIVE}, Tucker={W_TUCKER_ACTIVE}, exp029={W_EXP029_ACTIVE}")
print(f"  Dead (NB4 zero):    NB4={W_NB4_DEAD}, Tucker={W_TUCKER_DEAD:.4f}, exp029={W_EXP029_DEAD:.4f}")
print(f"  Active sum: {W_NB4_ACTIVE + W_TUCKER_ACTIVE + W_EXP029_ACTIVE}")
print(f"  Dead sum:   {W_NB4_DEAD + W_TUCKER_DEAD + W_EXP029_DEAD}")

# Per-species weight arrays
w_nb4 = nb4_active_mask * W_NB4_ACTIVE + (1 - nb4_active_mask) * W_NB4_DEAD
w_tuc = nb4_active_mask * W_TUCKER_ACTIVE + (1 - nb4_active_mask) * W_TUCKER_DEAD
w_e29 = nb4_active_mask * W_EXP029_ACTIVE + (1 - nb4_active_mask) * W_EXP029_DEAD

# Verify sum = 1 per species
w_sum = w_nb4 + w_tuc + w_e29
assert np.allclose(w_sum, 1.0), f"Weight sum mismatch: min={w_sum.min()}, max={w_sum.max()}"
print(f"\\n  Per-species weight sum: min={w_sum.min():.6f}, max={w_sum.max():.6f} (must be 1.0)")

# Broadcast (234,) → (1, 1, 234) and compute blend
print(f"\\n=== Compute blend ===")
pseudo = w_nb4[None, None, :] * p_n + w_tuc[None, None, :] * p_t + w_e29[None, None, :] * p_e
pseudo = pseudo.astype(np.float32)

# Verify output
print(f"\\nOutput distribution:")
print(f"  shape: {pseudo.shape}, dtype: {pseudo.dtype}")
print(f"  mean = {pseudo.mean():.5f}")
print(f"  median = {np.median(pseudo):.5f}")
print(f"  min  = {pseudo.min():.5f}, max = {pseudo.max():.5f}")
print(f"  std  = {pseudo.std():.5f}")
print(f"  >0.5 frac = {(pseudo > 0.5).sum()/pseudo.size:.5f}")
print(f"  >0.9 frac = {(pseudo > 0.9).sum()/pseudo.size:.5f}")

# Critical assertions
assert int(np.isnan(pseudo).sum()) == 0, "Output has NaN"
assert int(np.isinf(pseudo).sum()) == 0, "Output has Inf"
assert (pseudo >= 0).all(), "Output has negative values"
assert (pseudo <= 1.0).all(), "Output exceeds 1.0"
assert pseudo.mean() < 0.05, f"Output mean {pseudo.mean()} suspiciously high (rank-pct artifact?)"
print(f"\\n  All sanity assertions passed")
"""))

# Cell 5: Save output
cells.append(code_cell("""# Save outputs to /kaggle/working/
print("=== Save outputs ===")

# 1. Pseudo NPZ (float16 for storage efficiency)
out_npz = OUT_DIR / "pseudo_adaptive_234.npz"
np.savez_compressed(
    out_npz,
    probs=pseudo.astype(np.float16),
    file_ids=file_ids,
    nb4_active_mask=nb4_active_mask.astype(np.int8),
    w_nb4=w_nb4.astype(np.float32),
    w_tuc=w_tuc.astype(np.float32),
    w_e29=w_e29.astype(np.float32),
)
print(f"  {out_npz.name}: {out_npz.stat().st_size/1e6:.2f} MB")

# 2. primary_labels.json (234 sp ordering for downstream)
PRIMARY_LABELS = taxo["primary_label"].astype(str).tolist()
with open(OUT_DIR / "primary_labels.json", "w") as f:
    json.dump(PRIMARY_LABELS, f, indent=2)

# 3. file_index.json (file_id → idx mapping)
with open(OUT_DIR / "file_index.json", "w") as f:
    json.dump({str(fid): i for i, fid in enumerate(file_ids)}, f)

# 4. Metadata
meta = {
    "exp": "exp080a",
    "description": "3-stream adaptive blend (NB4 + Tucker + exp029, dead-sp renormalize)",
    "n_files": int(N_FILES),
    "n_windows": int(N_WIN),
    "n_classes": int(N_CLS),
    "dead_threshold": DEAD_THRESHOLD,
    "n_active_sp": int(n_active),
    "n_dead_sp": int(n_dead),
    "active_weights": {"nb4": W_NB4_ACTIVE, "tucker": W_TUCKER_ACTIVE, "exp029": W_EXP029_ACTIVE},
    "dead_weights": {"nb4": W_NB4_DEAD, "tucker": float(W_TUCKER_DEAD), "exp029": float(W_EXP029_DEAD)},
    "output_stats": {
        "mean": float(pseudo.mean()),
        "median": float(np.median(pseudo)),
        "max": float(pseudo.max()),
        "min": float(pseudo.min()),
        "frac_gt_0.5": float((pseudo > 0.5).sum() / pseudo.size),
        "frac_gt_0.9": float((pseudo > 0.9).sum() / pseudo.size),
    },
    "total_time_min": (time.time() - START) / 60,
}
with open(OUT_DIR / "adaptive_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  primary_labels.json, file_index.json, adaptive_meta.json saved")

print(f"\\n=== exp080a DONE ===")
print(f"Total time: {(time.time() - START)/60:.2f} min")
print(f"\\nMeta summary:")
for k, v in meta.items():
    print(f"  {k}: {v}")
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Built: {OUT_PATH}")
print(f"  cells: {len(cells)}")
for c in cells:
    n_lines = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} L={n_lines}")
