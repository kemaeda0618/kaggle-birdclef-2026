"""Generate Kaggle NB: pseudo diversity analysis (6 teachers).

Inputs (kernel_sources, 6 pseudo NBs):
- maekeso/birdclef2026-exp028-pseudo-e17           -> pseudo_e17.csv      (eca_nfnet_l0 R2 single fold = e17)
- maekeso/birdclef2026-exp028-pseudo-l0r2          -> pseudo_l0r2.csv     (eca_nfnet_l0 R2 5-fold)
- maekeso/birdclef2026-exp028-pseudo-exp015        -> pseudo_exp015.csv   (convnext_pico R2)
- maekeso/birdclef2026-exp028-pseudo-exp016        -> pseudo_exp016.csv   (regnety_008 R2)
- maekeso/birdclef2026-exp028-pseudo-hgnetv2-tucker -> pseudo_tucker.csv  (Tucker SED public)
- maekeso/birdclef2026-exp028-pseudo-hgnetv2-r1    -> pseudo_exp013.csv   (Babych b3 R1, HGNetV2-B0)

Analysis:
1. Pair-wise correlation (Pearson, mean over rows)
2. PCA 2D of teacher-vectors (each teacher = flattened predictions)
3. Per-class disagreement (variance across teachers per species)
4. Recommended R3 subset (low correlation, high coverage)

Output (slim):
- /kaggle/working/pseudo_diversity_summary.parquet
- /kaggle/working/correlation_matrix.npz
- /kaggle/working/pseudo_pca_2d.parquet
- /kaggle/working/per_class_variance.parquet
- /kaggle/working/r3_recommended.json
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "nb_diversity.ipynb"


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown", "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code", "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src.splitlines(keepends=True),
    }


cells = []

# ── Header
cells.append(md_cell("""# Pseudo Diversity Analysis (6 teachers)

Analyze which of our 6 already-generated pseudo-label sets are diverse vs redundant.

**Inputs:**
- exp017 R2 (eca_nfnet_l0 single fold)
- exp020 R2 (eca_nfnet_l0 5-fold)
- exp015 R2 (convnext_pico)
- exp016 R2 (regnety_008)
- Tucker SED public
- exp013 R1 (Babych b3 base / HGNetV2-B0)

**Goal:** Select a minimal subset for Multi-iter NS R3 with high diversity, low correlation."""))

# ── Setup
cells.append(code_cell("""import os, gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)

# V8: Use dataset_sources (mounted at /kaggle/input/{dataset-slug}/)
# Datasets updated 2026-05-19 with correct pseudo outputs
DATASET_TO_TAG = {
    'exp028-pseudo-eca-nfnet-l0-e17':    'nfnet_e17',     # eca_nfnet_l0 R2 single fold
    'exp028-pseudo-eca-nfnet-l0-r2':     'nfnet_5fold',   # eca_nfnet_l0 R2 5-fold (修正版)
    'exp028-pseudo-convnext-pico':       'convnext',      # convnext_pico R2
    'exp028-pseudo-regnety-008':         'regnety',       # regnety_008 R2
    'exp028-pseudo-hgnetv2-tucker':      'tucker',        # Tucker SED public
    'exp028-pseudo-hgnetv2-r1':          'hgnet_r1',      # Babych HGNetV2-B0 R1 (LB 0.729 = low!)
    'exp028-pseudo-eca-nfnet-l1-exp029': 'l1_exp029',     # exp029 l1 R3 fold0 (NEW)
}

KAGGLE_INPUT = Path('/kaggle/input')

print('Directory listing /kaggle/input/ for our 7 pseudo datasets:')
for slug in DATASET_TO_TAG:
    d = KAGGLE_INPUT / slug
    if not d.exists():
        alt = KAGGLE_INPUT / 'datasets' / 'maekeso' / slug
        if alt.exists():
            d = alt
    print(f'  {slug}/  exists={d.exists()}  path={d}')
    if d.exists():
        for f in sorted(d.iterdir())[:4]:
            sz = f.stat().st_size if f.is_file() else '-'
            print(f'    {f.name}  ({sz})')
print()

# Discover each teacher's pseudo CSV — pick LARGEST .csv per dir
PSEUDO_SOURCES = {}
for slug, tag in DATASET_TO_TAG.items():
    candidates = [
        KAGGLE_INPUT / slug,
        KAGGLE_INPUT / 'datasets' / 'maekeso' / slug,
    ]
    d = next((c for c in candidates if c.exists()), None)
    if d is None:
        print(f'  {tag:<12s} DIR MISSING  {slug}')
        continue
    csvs = [(f, f.stat().st_size) for f in d.glob('*.csv')]
    csvs.sort(key=lambda x: -x[1])
    if not csvs:
        print(f'  {tag:<12s} NO CSV       {d}')
        continue
    chosen = csvs[0][0]
    if csvs[0][1] < 100_000_000:
        print(f'  {tag:<12s} TOO SMALL    {chosen.name} ({csvs[0][1]/1e6:.1f} MB)')
        continue
    PSEUDO_SOURCES[tag] = str(chosen)
    print(f'  {tag:<12s} OK           {chosen.name} ({csvs[0][1]/1e6:.1f} MB)')

print(f'\\n{len(PSEUDO_SOURCES)} teachers available')"""))

# ── Load + align
cells.append(md_cell("""## 1. Load + align by row_id"""))
cells.append(code_cell("""# Load each pseudo, find species columns + row_id columns
data = {}
species_cols_per = {}
for name, path in PSEUDO_SOURCES.items():
    if not Path(path).exists():
        print(f'SKIP {name}: file missing')
        continue
    t0 = time.time()
    df = pd.read_csv(path)
    # Identify columns
    meta_cols = [c for c in df.columns if c in ('filename', 'start_sec', 'end_sec', 'row_id')]
    species_cols = [c for c in df.columns if c not in meta_cols]
    species_cols_per[name] = species_cols
    # Construct row_id if missing
    if 'row_id' not in df.columns:
        if 'filename' in df.columns and 'end_sec' in df.columns:
            df['row_id'] = df['filename'].str.replace('.ogg','',regex=False) + '_' + df['end_sec'].astype(int).astype(str)
    data[name] = df
    print(f'{name:<10s}  rows={len(df):>7d}  species_cols={len(species_cols):>3d}  meta={meta_cols}  load={time.time()-t0:.1f}s')
print()
# Check species columns agreement
all_species_sets = {name: set(cols) for name, cols in species_cols_per.items()}
common_species = set.intersection(*all_species_sets.values()) if all_species_sets else set()
print(f'Common species columns: {len(common_species)}')
for name, cols in all_species_sets.items():
    extra = sorted(cols - common_species)
    missing = sorted(common_species - cols)
    if extra:
        print(f'  {name} has extra: {extra[:5]}{\"...\" if len(extra)>5 else \"\"} ({len(extra)})')
    if missing:
        print(f'  {name} missing: {missing[:5]} ({len(missing)})')"""))

# ── Align row_id
cells.append(code_cell("""# Common row_ids
row_id_sets = {name: set(df['row_id']) for name, df in data.items() if 'row_id' in df.columns}
common_rows = set.intersection(*row_id_sets.values()) if row_id_sets else set()
print(f'Common row_ids: {len(common_rows)} across {len(row_id_sets)} teachers')
for name, s in row_id_sets.items():
    print(f'  {name}: {len(s)} rows  ({len(common_rows)/len(s):.1%} usable)')

# Sort row_ids deterministically
SPECIES = sorted(common_species)
ROWS = sorted(common_rows)
print(f'\\nFinal align: {len(ROWS)} rows x {len(SPECIES)} species')"""))

cells.append(code_cell("""# Build aligned matrix: (n_teachers, n_rows, n_species) — float32 for memory
teacher_names = sorted(data.keys())
n_t, n_r, n_s = len(teacher_names), len(ROWS), len(SPECIES)
print(f'Allocating matrix: ({n_t}, {n_r}, {n_s}) float32 = {n_t*n_r*n_s*4/1e9:.2f} GB')

M = np.zeros((n_t, n_r, n_s), dtype=np.float32)
for ti, name in enumerate(teacher_names):
    df = data[name]
    df_sel = df[df['row_id'].isin(common_rows)].set_index('row_id').loc[ROWS]
    M[ti] = df_sel[SPECIES].values.astype(np.float32)
    print(f'{name}: loaded into M[{ti}], mean={M[ti].mean():.4f}, std={M[ti].std():.4f}')

print(f'\\nM shape: {M.shape}, dtype: {M.dtype}, size: {M.nbytes/1e9:.2f} GB')

# Free original dataframes
del data
gc.collect()"""))

# ── Correlation
cells.append(md_cell("""## 2. Pair-wise correlation matrix (Pearson)"""))
cells.append(code_cell("""# Pearson correlation between teacher prediction vectors (flatten rows*species)
def teacher_corr(M):
    n_t = M.shape[0]
    flat = M.reshape(n_t, -1)  # (n_t, n_r*n_s)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.sqrt((flat**2).sum(axis=1))
    return (flat @ flat.T) / (norm[:, None] * norm[None, :] + 1e-12)

corr = teacher_corr(M)
print('Pair-wise Pearson correlation:')
corr_df = pd.DataFrame(corr, index=teacher_names, columns=teacher_names).round(3)
print(corr_df)

# Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, vmin=0.5, vmax=1.0, cmap='RdYlGn_r')
ax.set_xticks(range(n_t)); ax.set_xticklabels(teacher_names, rotation=45, ha='right')
ax.set_yticks(range(n_t)); ax.set_yticklabels(teacher_names)
for i in range(n_t):
    for j in range(n_t):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                color='white' if corr[i,j]>0.85 else 'black', fontsize=9)
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title('Teacher Pseudo Correlation (Pearson)')
plt.tight_layout()
plt.savefig('/kaggle/working/correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.show()

np.savez('/kaggle/working/correlation_matrix.npz', corr=corr, teachers=np.array(teacher_names))"""))

# ── Pair table
cells.append(code_cell("""# Pair-wise correlations as sortable table
pairs = []
for i in range(n_t):
    for j in range(i+1, n_t):
        pairs.append((teacher_names[i], teacher_names[j], float(corr[i, j])))
pairs_df = pd.DataFrame(pairs, columns=['teacher_a', 'teacher_b', 'pearson_r']).sort_values('pearson_r', ascending=False)
print('Pair-wise correlations (sorted):')
print(pairs_df.to_string(index=False))
pairs_df.to_parquet('/kaggle/working/pseudo_diversity_pairs.parquet', index=False)"""))

# ── PCA
cells.append(md_cell("""## 3. PCA 2D of teacher vectors"""))
cells.append(code_cell("""# Each teacher = 1 point in (n_rows*n_species)-dim space
flat = M.reshape(n_t, -1).astype(np.float32)
# Center
flat_c = flat - flat.mean(axis=0, keepdims=True)

# Robust to small n_t (e.g., only 2 teachers loaded)
n_components = max(1, min(n_t - 1, 2))
pca = PCA(n_components=n_components)
pca_proj = pca.fit_transform(flat_c)
# Pad to 2D if only 1 PC
if pca_proj.shape[1] == 1:
    pca_2d = np.column_stack([pca_proj[:, 0], np.arange(n_t).astype(np.float32)])  # y = ordinal
    var_ratios = [pca.explained_variance_ratio_[0], 0.0]
    pc2_label = 'ordinal (n_t too small for real PC2)'
else:
    pca_2d = pca_proj
    var_ratios = list(pca.explained_variance_ratio_)
    pc2_label = f'PC2 ({var_ratios[1]:.1%})'
print(f'PCA explained variance: {[f\"{v:.3f}\" for v in var_ratios]}')

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pca_2d[:, 0], pca_2d[:, 1], s=200, c=range(n_t), cmap='tab10', edgecolors='black')
for i, name in enumerate(teacher_names):
    ax.annotate(name, (pca_2d[i, 0], pca_2d[i, 1]),
                xytext=(8, 5), textcoords='offset points', fontsize=11, fontweight='bold')
ax.set_xlabel(f'PC1 ({var_ratios[0]:.1%})')
ax.set_ylabel(pc2_label)
ax.set_title('Teacher PCA (2D projection of pseudo predictions)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/pseudo_pca_2d.png', dpi=100, bbox_inches='tight')
plt.show()

pca_df = pd.DataFrame({
    'teacher': teacher_names,
    'pc1': pca_2d[:, 0],
    'pc2': pca_2d[:, 1],
})
print(pca_df)
pca_df.to_parquet('/kaggle/working/pseudo_pca_2d.parquet', index=False)"""))

# ── Hierarchical
cells.append(code_cell("""# Hierarchical clustering on correlation distance
dist = 1 - corr  # cosine-ish distance
n_clusters = max(1, min(3, n_t - 1))
if n_t >= 2:
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels_clust = agg.fit_predict(dist)
else:
    labels_clust = np.zeros(n_t, dtype=int)
clust_df = pd.DataFrame({'teacher': teacher_names, 'cluster': labels_clust})
print(f'Teacher clusters (n={n_clusters}) by correlation similarity:')
print(clust_df.sort_values('cluster').to_string(index=False))"""))

# ── Per-class disagreement
cells.append(md_cell("""## 4. Per-class disagreement"""))
cells.append(code_cell("""# Variance across teachers per (row, species), then average per species
# But row count is huge; use a more meaningful metric:
# per-species: mean over rows of (var across teachers)
per_species_var = M.var(axis=0).mean(axis=0)  # (n_species,)
per_species_mean = M.mean(axis=(0, 1))         # (n_species,)
per_species_max  = M.mean(axis=1).max(axis=0)  # (n_species,)
n_files = M.shape[1]

per_species_df = pd.DataFrame({
    'species': SPECIES,
    'teacher_variance': per_species_var,
    'mean_pred': per_species_mean,
    'max_teacher_mean': per_species_max,
}).sort_values('teacher_variance', ascending=False)

print('Top 30 species by teacher disagreement (high variance = teachers split):')
print(per_species_df.head(30).round(4).to_string(index=False))

print('\\nBottom 30 species (low variance = all teachers agree):')
print(per_species_df.tail(30).round(4).to_string(index=False))

per_species_df.to_parquet('/kaggle/working/per_class_variance.parquet', index=False)"""))

# ── Class-level disagreement
cells.append(code_cell("""# Class-level (Aves/Insecta/...) disagreement
TAX_PATH = '/kaggle/input/birdclef-2026/taxonomy.csv'
if not Path(TAX_PATH).exists():
    TAX_PATH = '/kaggle/input/competitions/birdclef-2026/taxonomy.csv'
tax = pd.read_csv(TAX_PATH)
per_species_df = per_species_df.merge(tax[['primary_label','class_name']],
                                        left_on='species', right_on='primary_label', how='left')

class_summary = per_species_df.groupby('class_name').agg(
    n_species=('species', 'count'),
    mean_variance=('teacher_variance', 'mean'),
    median_variance=('teacher_variance', 'median'),
).round(5).sort_values('mean_variance', ascending=False)
print('Class-level disagreement:')
print(class_summary)"""))

# ── Recommendations
cells.append(md_cell("""## 5. R3 subset recommendation"""))
cells.append(code_cell("""# Strategy: pick a subset of teachers such that
# - max pair correlation < threshold (e.g., 0.95)
# - includes at least 1 from each cluster
# Greedy: sort teachers by uniqueness (1 - mean corr with others)

mean_corr_excl_self = (corr.sum(axis=1) - 1) / (n_t - 1)  # exclude self-correlation
uniqueness = 1 - mean_corr_excl_self
ranked = sorted(enumerate(teacher_names), key=lambda x: uniqueness[x[0]], reverse=True)
print('Teacher uniqueness ranking (high = different from others):')
for idx, name in ranked:
    print(f'  {name:<10s}  mean_corr_with_others={mean_corr_excl_self[idx]:.3f}  uniqueness={uniqueness[idx]:.3f}')

# Suggest R3 subset: top 3-4 unique teachers
print('\\nSuggested R3 subsets:')
for k in [3, 4, 5]:
    subset = [name for idx, name in ranked[:k]]
    subset_idxs = [i for i, n in enumerate(teacher_names) if n in subset]
    sub_corr = corr[np.ix_(subset_idxs, subset_idxs)]
    sub_max = sub_corr[~np.eye(k, dtype=bool)].max()
    sub_mean = sub_corr[~np.eye(k, dtype=bool)].mean()
    print(f'  K={k}: {subset}  max_pair_r={sub_max:.3f}  mean_r={sub_mean:.3f}')

recommended = {
    'subset_k3': [name for _, name in ranked[:3]],
    'subset_k4': [name for _, name in ranked[:4]],
    'subset_k5': [name for _, name in ranked[:5]],
    'uniqueness': {name: float(uniqueness[i]) for i, name in enumerate(teacher_names)},
    'mean_corr_with_others': {name: float(mean_corr_excl_self[i]) for i, name in enumerate(teacher_names)},
}
with open('/kaggle/working/r3_recommended.json', 'w') as f:
    json.dump(recommended, f, indent=2)
print('\\nSaved: /kaggle/working/r3_recommended.json')"""))

# ── Summary parquet (slim, for local Streamlit if wanted)
cells.append(code_cell("""# Slim summary parquet: per-teacher overview
summary = pd.DataFrame({
    'teacher': teacher_names,
    'mean_pred': M.mean(axis=(1, 2)),
    'std_pred': M.std(axis=(1, 2)),
    'pc1': pca_2d[:, 0],
    'pc2': pca_2d[:, 1],
    'cluster': labels_clust,
    'mean_corr_with_others': mean_corr_excl_self,
    'uniqueness': uniqueness,
}).round(4)
print(summary.to_string(index=False))
summary.to_parquet('/kaggle/working/pseudo_diversity_summary.parquet', index=False)
print('\\nAll outputs saved to /kaggle/working/')
print('  - correlation_heatmap.png')
print('  - correlation_matrix.npz')
print('  - pseudo_diversity_pairs.parquet')
print('  - pseudo_pca_2d.png, pseudo_pca_2d.parquet')
print('  - per_class_variance.parquet')
print('  - r3_recommended.json')
print('  - pseudo_diversity_summary.parquet')"""))

# Write notebook
import json as _json
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(_json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {OUT}")
print(f"  {len(cells)} cells")
