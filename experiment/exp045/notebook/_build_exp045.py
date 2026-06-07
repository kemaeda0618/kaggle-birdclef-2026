"""Modify nb_train_r3_l1_filtered.ipynb in place:
1. hdr: rewrite to ablation purpose
2. setup: exp029 -> exp045 in DRIVE_EXP_DIR
3. config: add CHUNK_FILTER_MAX_PROB = 0.5
4. fold_loop: add chunk filter after pseudo_df_r1 = pd.read_csv(...)
5. upload: change SLUG/TITLE to exp045
"""
import json
from pathlib import Path

NB_PATH = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb")

nb = json.load(open(NB_PATH, encoding="utf-8"))


def get_cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def replace_cell_source(cid, new_src):
    c = get_cell(cid)
    c["source"] = new_src.splitlines(keepends=True)


def patch_cell_source(cid, old, new, must_exist=True):
    c = get_cell(cid)
    src = "".join(c["source"])
    if old not in src:
        if must_exist:
            raise ValueError(f"pattern not found in cell {cid}:\n{old[:200]}")
        return
    src = src.replace(old, new, 1)
    c["source"] = src.splitlines(keepends=True)


# ---- 1. hdr ----
NEW_HDR = """# exp045: exp029 R3 + Chunk Filter Ablation (l1 single fold)

**目的**: paper 256 流 chunk filter (max_prob > 0.5 で chunk drop) の効果を実測。
exp029 R3 (val_ns22 fold 0 = 0.9177、LB 0.923 5-fold ensemble) を base に、**唯一の variable** として teacher pseudo に chunk filter を適用。

## 設計

- **base**: exp029 R3 spec を完全複製 (eca_nfnet_l1 + exp017 R2 teacher pseudo + Perch distill + 20 ep + same config)
- **唯一の変更**: `CHUNK_FILTER_MAX_PROB = 0.5` を pseudo CSV 読込直後に適用
- **fold**: FOLDS=[0] single fold
- **比較**: 既存 exp029 R3 fold 0 val_ns22 = **0.9177** vs exp045 (with filter) val_ns22

## 期待結果

| Outcome | 解釈 |
|---|---|
| delta > +0.003 | filter 有効、exp029 R3 5-fold 再学習 (15h) で展開 |
| -0.003 ≤ delta ≤ +0.003 | noise floor、無効と判定 |
| delta < -0.003 | filter は drag、学習データが減って悪化 |

## 出力

- Drive: `output/exp045/fold0/r3/ckpt_best_ns22.pth`
- Kaggle Dataset: `maekeso/birdclef2026-exp045-l1-filtered` (新規)

## 前提

- `pseudo_e17.csv` (exp028 e17 pseudo、exp017 R2 teacher) を Drive `kaggle/birdclef2026/pseudo_e17.csv` に DL 済 or NB 内 selective DL
- Perch cache `kaggle/birdclef2026/perch-cache/{emb.npy, meta.csv}` を Drive に存在
"""
replace_cell_source("hdr", NEW_HDR)
print("[hdr] OK rewritten")


# ---- 2. setup: exp029 -> exp045 ----
patch_cell_source(
    "setup",
    'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029"',
    'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp045"',
)
print("[setup] OK DRIVE_EXP_DIR -> exp045")


# ---- 3. config: add CHUNK_FILTER_MAX_PROB ----
# Insert after SHARES_R2 line
patch_cell_source(
    "config",
    'SHARES_R2 = {"focal": 0.65, "labeled_sc": 0.10, "pseudo_sc": 0.25}   # ★ exp029 Option C: pseudo 0.20 → 0.25、distill 強化\nSOURCE_WEIGHTS',
    'SHARES_R2 = {"focal": 0.65, "labeled_sc": 0.10, "pseudo_sc": 0.25}   # ★ exp029 Option C: pseudo 0.20 → 0.25、distill 強化\n\n# ★ exp045 Ablation: paper 256 流 chunk filter (max_prob > 0.5 で teacher pseudo の chunk drop)\n# CHUNK_FILTER_MAX_PROB = 0.0 で disable (exp029 R3 baseline と同等)\n# CHUNK_FILTER_MAX_PROB = 0.5 で paper 256 spec (Sydorskyi 2nd place の primary_label_min_prob)\nCHUNK_FILTER_MAX_PROB = 0.5\n\nSOURCE_WEIGHTS',
)
print("[config] OK added CHUNK_FILTER_MAX_PROB")


# ---- 4. fold_loop: add chunk filter ----
patch_cell_source(
    "fold_loop",
    """    print(f"  Loading teacher pseudo (exp017 R2): {DRIVE_TEACHER_PSEUDO}")
    pseudo_df_r1 = pd.read_csv(DRIVE_TEACHER_PSEUDO)
    print(f"    {len(pseudo_df_r1)} rows")""",
    """    print(f"  Loading teacher pseudo (exp017 R2): {DRIVE_TEACHER_PSEUDO}")
    pseudo_df_r1 = pd.read_csv(DRIVE_TEACHER_PSEUDO)
    print(f"    {len(pseudo_df_r1)} rows (raw)")

    # ★ exp045: paper 256 流 chunk filter (max_prob > threshold で chunk drop)
    if CHUNK_FILTER_MAX_PROB > 0:
        prob_cols = [c for c in pseudo_df_r1.columns if c in PRIMARY_LABELS]
        assert len(prob_cols) == NUM_CLASSES, f"Expected {NUM_CLASSES} prob cols, got {len(prob_cols)}"
        max_prob_per_row = pseudo_df_r1[prob_cols].max(axis=1)
        keep_mask = max_prob_per_row > CHUNK_FILTER_MAX_PROB
        n_before = len(pseudo_df_r1)
        n_after = int(keep_mask.sum())
        pseudo_df_r1 = pseudo_df_r1[keep_mask].reset_index(drop=True)
        print(f"    [chunk filter > {CHUNK_FILTER_MAX_PROB}] {n_before} -> {n_after} ({100*n_after/n_before:.1f}%)")
        print(f"      max_prob 50%ile={np.percentile(max_prob_per_row, 50):.3f} 90%ile={np.percentile(max_prob_per_row, 90):.3f}")
    else:
        print(f"    [chunk filter DISABLED] CHUNK_FILTER_MAX_PROB={CHUNK_FILTER_MAX_PROB}")
""",
)
print("[fold_loop] OK added chunk filter")


# ---- 5. upload: SLUG/TITLE -> exp045 ----
patch_cell_source(
    "upload",
    'SLUG  = "birdclef2026-exp029-l1-single"\nTITLE = "birdclef2026 exp029 l1 single (R3 from e17)"',
    'SLUG  = "birdclef2026-exp045-l1-filtered"\nTITLE = "birdclef2026 exp045 l1 filtered ablation"',
)
patch_cell_source(
    "upload",
    'version_notes = f"exp029 l1 single (R3 from e17) | {summary}"[:498]',
    'version_notes = f"exp045 l1 chunk-filter ablation (filter={CHUNK_FILTER_MAX_PROB}) | {summary}"[:498]',
)
print("[upload] OK SLUG -> exp045")


# Save
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nDONE: saved {NB_PATH}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c.get('cell_type'):8s} id={c.get('id'):20s} L={n}")
