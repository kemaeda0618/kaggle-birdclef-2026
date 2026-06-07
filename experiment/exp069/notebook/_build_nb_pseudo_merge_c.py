"""Build exp069c (Hybrid merge) NB: 3 NPZ merge → 3 pseudo CSV 生成.

Inputs:
  exp069a kernel_source: maekeso/birdclef2026-exp069a-babych-pseudo-gen
    - babych_ensemble_206.npz: (n_files, 12, 206 BC25 class)
    - babych_label2ind.json: BC25 species index mapping

  exp069b kernel_sources (3 separate):
    - maekeso/birdclef2026-exp069b-nb4-pseudo
    - maekeso/birdclef2026-exp069b-tucker-pseudo
    - maekeso/birdclef2026-exp069b-exp029-pseudo

Process:
  1. Load 3 stream NPZ → rank avg + sonotype mirror = exp048 blend equivalent
  2. Load Babych ensemble → BC25→BC26 species mapping
  3. Generate 3 pseudo NPZ:
     - pseudo_babych_234.npz: Babych mapped (non-overlap = NaN)
     - pseudo_exp048_234.npz: exp048 blend (rank-avg + mirror)
     - pseudo_hybrid_234.npz: overlap=Babych、non-overlap=exp048
  4. Save to /kaggle/working/ (kernel output 経由で M-R3 から参照)
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_pseudo_merge_c.ipynb")
cells = []


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


# Cell 0: Title
cells.append(md_cell("""# exp069c: Hybrid pseudo merge (3 NPZ → 3 pseudo)

**Inputs (kernel_sources)**:
  - exp069a: Babych ensemble (206 BC25 class)
  - exp069b-nb4, tucker, exp029: 3 stream raw (234 BC26 class)

**Process**:
  1. exp069b 3 stream → exp048 blend (rank-avg + sonotype mirror、234 class)
  2. exp069a Babych ensemble → BC25→BC26 species mapping (inat_taxon_id 照合)
  3. Per-class merge:
     - pseudo_babych_234.npz: Babych mapped (non-overlap NaN)
     - pseudo_exp048_234.npz: exp048 blend
     - **pseudo_hybrid_234.npz**: ★ overlap=Babych, non-overlap=exp048 ★

**Output → /kaggle/working/**:
  - pseudo_babych_234.npz, pseudo_exp048_234.npz, pseudo_hybrid_234.npz
  - bc25_to_bc26_map.json: species mapping reference

**M-R3 usage**:
  - M1, M2 (Perch ON): pseudo_exp048_234.npz
  - M3-M7 (Babych init): pseudo_hybrid_234.npz
"""))


# Cell 1: Setup
cells.append(code_cell("""# Setup
import sys, os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
print(f"Python: {sys.version[:50]}")
print(f"numpy: {np.__version__}")
START = time.time()
"""))


# Cell 2: Paths
cells.append(code_cell("""# Locate inputs
def find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None

# BC2026 competition data
DATA_PATH = find_dir([
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
])
assert DATA_PATH is not None
TAXONOMY_CSV = DATA_PATH / "taxonomy.csv"

# exp069a output (Babych ensemble)
EXP069A_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069a-babych-pseudo-gen",   # if added as kernel_source
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069a-babych-pseudo-gen",
])
assert EXP069A_DIR is not None, "exp069a kernel output not attached"

# exp069b 3 stream outputs
EXP069B_NB4_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069b-nb4-pseudo",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069b-nb4-pseudo",
])
EXP069B_TUCKER_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069b-tucker-pseudo",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069b-tucker-pseudo",
])
EXP069B_EXP029_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp069b-exp029-pseudo",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp069b-exp029-pseudo",
])
assert EXP069B_NB4_DIR is not None, "exp069b-nb4 not attached"
assert EXP069B_TUCKER_DIR is not None, "exp069b-tucker not attached"
assert EXP069B_EXP029_DIR is not None, "exp069b-exp029 not attached"

OUT_DIR = Path("/kaggle/working")

print(f"DATA_PATH: {DATA_PATH}")
print(f"EXP069A_DIR: {EXP069A_DIR}")
print(f"EXP069B_NB4: {EXP069B_NB4_DIR}")
print(f"EXP069B_TUCKER: {EXP069B_TUCKER_DIR}")
print(f"EXP069B_EXP029: {EXP069B_EXP029_DIR}")
"""))


# Cell 3: Load taxonomy + BC25 mapping
cells.append(code_cell("""# Load BC26 taxonomy
taxo = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = taxo["primary_label"].astype(str).tolist()
N_CLASSES_BC26 = len(PRIMARY_LABELS)
assert N_CLASSES_BC26 == 234, f"Expected 234 BC26 classes, got {N_CLASSES_BC26}"
print(f"BC26 species: {N_CLASSES_BC26}")

# inat_taxon_id mapping
bc26_taxon_id_to_idx = {}
for idx, row in taxo.iterrows():
    tid = str(row["inat_taxon_id"])
    bc26_taxon_id_to_idx[tid] = idx
print(f"BC26 inat_taxon_id mapping: {len(bc26_taxon_id_to_idx)}")

# Load exp069a Babych label2ind (BC25 species → 206 index)
babych_label2ind_path = EXP069A_DIR / "babych_label2ind.json"
assert babych_label2ind_path.exists()
babych_label2ind = json.loads(babych_label2ind_path.read_text())
print(f"Babych BC25 species: {len(babych_label2ind)}")

# BC25 ↔ BC26 mapping (by inat_taxon_id where applicable)
# Babych label format: many are inat_taxon_id strings, some are codes
# Build overlap pairs
bc25_to_bc26_pairs = []
for bc25_label, bc25_idx in babych_label2ind.items():
    if bc25_label in bc26_taxon_id_to_idx:
        bc26_idx = bc26_taxon_id_to_idx[bc25_label]
        bc25_to_bc26_pairs.append((bc25_idx, bc26_idx, bc25_label))
print(f"BC25 ↔ BC26 overlap: {len(bc25_to_bc26_pairs)} species")
"""))


# Cell 4: Load 3 stream NPZ
cells.append(code_cell("""# Load 3 exp069b stream NPZ
def load_npz(dir_path, fname):
    p = dir_path / fname
    if not p.exists():
        # Try alternative
        candidates = list(dir_path.glob("*.npz"))
        if candidates:
            p = candidates[0]
        else:
            raise FileNotFoundError(f"NPZ not found in {dir_path}")
    print(f"  Loading {p}")
    return dict(np.load(p, allow_pickle=True))

print("=== Loading 3 stream NPZ ===")
nb4_data = load_npz(EXP069B_NB4_DIR, "nb4_raw_234.npz")
tucker_data = load_npz(EXP069B_TUCKER_DIR, "tucker_raw_234.npz")
exp029_data = load_npz(EXP069B_EXP029_DIR, "exp029_raw_234.npz")

probs_nb4 = nb4_data["probs"].astype(np.float32)
probs_tucker = tucker_data["probs"].astype(np.float32)
probs_exp029 = exp029_data["probs"].astype(np.float32)

file_ids_nb4 = nb4_data["file_ids"]
file_ids_tucker = tucker_data["file_ids"]
file_ids_exp029 = exp029_data["file_ids"]

print(f"NB4 v11:    {probs_nb4.shape}, files={len(file_ids_nb4)}")
print(f"Tucker SED: {probs_tucker.shape}, files={len(file_ids_tucker)}")
print(f"exp029 R3:  {probs_exp029.shape}, files={len(file_ids_exp029)}")

# Verify alignment
assert np.array_equal(file_ids_nb4, file_ids_tucker), "file_ids mismatch nb4/tucker"
assert np.array_equal(file_ids_nb4, file_ids_exp029), "file_ids mismatch nb4/exp029"
file_ids = file_ids_nb4
N_FILES, N_WINDOWS, N_CLASSES = probs_nb4.shape
"""))


# Cell 5: exp048 blend logic (rank-avg + sonotype mirror)
cells.append(code_cell("""# exp048 blend logic (3-way rank avg + sonotype mirror)
# Weights from exp048: 0.30 / 0.40 / 0.30
W_NB4 = 0.30
W_TUCKER = 0.40
W_EXP029 = 0.30

print("=== Building exp048 blend ===")

# Flatten for rank computation
flat_nb4 = probs_nb4.reshape(-1, N_CLASSES)
flat_tucker = probs_tucker.reshape(-1, N_CLASSES)
flat_exp029 = probs_exp029.reshape(-1, N_CLASSES)

# Rank-avg blend (per-class, percentile rank)
rank_nb4 = pd.DataFrame(flat_nb4).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_tucker = pd.DataFrame(flat_tucker).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_exp029 = pd.DataFrame(flat_exp029).rank(axis=0, pct=True).to_numpy().astype(np.float32)
blend_flat = W_NB4 * rank_nb4 + W_TUCKER * rank_tucker + W_EXP029 * rank_exp029

# Sonotype mirror (exp048 spec)
MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)
col_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
mirror_count = 0
for group in MIRROR_PAIRS:
    valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
    if len(valid_idx) >= 2:
        group_max = blend_flat[:, valid_idx].max(axis=1, keepdims=True)
        blend_flat[:, valid_idx] = group_max
        mirror_count += len(valid_idx)
print(f"Sonotype mirror applied to {mirror_count} columns")

# Reshape back
probs_exp048 = blend_flat.reshape(N_FILES, N_WINDOWS, N_CLASSES).astype(np.float16)
print(f"exp048 blend: {probs_exp048.shape}, mean={probs_exp048.mean():.4f}, max={probs_exp048.max():.4f}")
"""))


# Cell 6: Babych mapped to BC26
cells.append(code_cell("""# Map Babych ensemble (206 BC25) → 234 BC26 (overlap species のみ、non-overlap NaN)
babych_npz = dict(np.load(EXP069A_DIR / "babych_ensemble_206.npz", allow_pickle=True))
babych_probs = babych_npz["probs"].astype(np.float32)
babych_file_ids = babych_npz["file_ids"]
print(f"Babych ensemble: {babych_probs.shape}, files={len(babych_file_ids)}")

# Verify file_ids alignment (must match exp069b streams)
assert np.array_equal(babych_file_ids, file_ids), "Babych vs exp069b file_ids mismatch"

# Create mapped tensor (NaN for non-overlap)
probs_babych_mapped = np.full((N_FILES, N_WINDOWS, N_CLASSES), np.nan, dtype=np.float32)
for bc25_idx, bc26_idx, _ in bc25_to_bc26_pairs:
    probs_babych_mapped[:, :, bc26_idx] = babych_probs[:, :, bc25_idx].astype(np.float32)

# Verify coverage
overlap_classes = sorted([bc26_idx for _, bc26_idx, _ in bc25_to_bc26_pairs])
non_overlap_classes = [i for i in range(N_CLASSES) if i not in set(overlap_classes)]
print(f"Babych mapped: {len(overlap_classes)} overlap (no NaN), {len(non_overlap_classes)} non-overlap (NaN)")
"""))


# Cell 7: Hybrid merge
cells.append(code_cell("""# Hybrid: overlap = Babych, non-overlap = exp048
probs_hybrid = probs_exp048.astype(np.float32).copy()
for bc26_idx in overlap_classes:
    probs_hybrid[:, :, bc26_idx] = probs_babych_mapped[:, :, bc26_idx]

# Quality check: NaN should not exist in hybrid
n_nan = np.isnan(probs_hybrid).sum()
assert n_nan == 0, f"Hybrid has {n_nan} NaN values - bug!"

print(f"Hybrid pseudo: {probs_hybrid.shape}")
print(f"  Overlap ({len(overlap_classes)} cls, Babych): mean={probs_hybrid[:,:,overlap_classes].mean():.4f}")
print(f"  Non-overlap ({len(non_overlap_classes)} cls, exp048): mean={probs_hybrid[:,:,non_overlap_classes].mean():.4f}")

probs_hybrid_f16 = probs_hybrid.astype(np.float16)
"""))


# Cell 8: Save 3 NPZ + metadata
cells.append(code_cell("""# Save all outputs to /kaggle/working/
import json as _json

print("=== Saving NPZ outputs ===")

# 1. Babych mapped (with NaN for non-overlap)
np.savez_compressed(
    OUT_DIR / "pseudo_babych_234.npz",
    probs=probs_babych_mapped.astype(np.float16),
    file_ids=file_ids,
)
print(f"  pseudo_babych_234.npz: {(OUT_DIR / 'pseudo_babych_234.npz').stat().st_size/1e6:.1f} MB")

# 2. exp048 blend (full 234)
np.savez_compressed(
    OUT_DIR / "pseudo_exp048_234.npz",
    probs=probs_exp048,
    file_ids=file_ids,
)
print(f"  pseudo_exp048_234.npz: {(OUT_DIR / 'pseudo_exp048_234.npz').stat().st_size/1e6:.1f} MB")

# 3. Hybrid (overlap=Babych, non-overlap=exp048)
np.savez_compressed(
    OUT_DIR / "pseudo_hybrid_234.npz",
    probs=probs_hybrid_f16,
    file_ids=file_ids,
)
print(f"  pseudo_hybrid_234.npz: {(OUT_DIR / 'pseudo_hybrid_234.npz').stat().st_size/1e6:.1f} MB")

# Metadata
with open(OUT_DIR / "primary_labels.json", "w") as f:
    _json.dump(list(PRIMARY_LABELS), f, indent=2)

bc25_bc26_map = {
    "n_overlap": len(bc25_to_bc26_pairs),
    "overlap_pairs": [{"bc25_idx": p[0], "bc26_idx": p[1], "label": p[2]} for p in bc25_to_bc26_pairs],
    "overlap_bc26_indices": overlap_classes,
    "non_overlap_bc26_indices": non_overlap_classes,
    "blend_weights": {"nb4": W_NB4, "tucker": W_TUCKER, "exp029": W_EXP029},
}
with open(OUT_DIR / "bc25_to_bc26_map.json", "w") as f:
    _json.dump(bc25_bc26_map, f, indent=2)

with open(OUT_DIR / "file_index.json", "w") as f:
    _json.dump({str(fid): i for i, fid in enumerate(file_ids)}, f)

with open(OUT_DIR / "exp069c_summary.json", "w") as f:
    _json.dump({
        "n_files": int(N_FILES),
        "n_windows": int(N_WINDOWS),
        "n_classes_bc26": int(N_CLASSES),
        "n_overlap_species": len(overlap_classes),
        "n_non_overlap_species": len(non_overlap_classes),
        "total_time_min": (time.time() - START) / 60,
    }, f, indent=2)

print(f"\\nOK exp069c DONE: total {(time.time()-START)/60:.1f} min")
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
print(f"Built: {OUT_PATH} ({len(cells)} cells)")
