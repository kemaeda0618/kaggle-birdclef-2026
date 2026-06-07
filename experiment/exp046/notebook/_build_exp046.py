"""Modify nb_train_r3_l1_gamma12.ipynb in place:
1. hdr: rewrite for γ=1.2 Power Transform ablation
2. setup: exp045 -> exp046 in DRIVE_EXP_DIR
3. config: CHUNK_FILTER_MAX_PROB -> POWER_GAMMA = 1.2
4. fold_loop: replace chunk filter logic with Power Transform
5. upload: change SLUG/TITLE to exp046

Key: metrics output unchanged (per_taxon + per_class distribution remain from exp045 patches).
"""
import json
from pathlib import Path

NB_PATH = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp046\notebook\nb_train_r3_l1_gamma12.ipynb")
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
            raise ValueError(f"pattern not found in cell {cid}:\n  {old[:200]}")
        return False
    src = src.replace(old, new, 1)
    c["source"] = src.splitlines(keepends=True)
    return True


# ---- 1. hdr ----
NEW_HDR = """# exp046: exp029 R3 + Power Transform γ=1.2 Ablation (l1 single fold)

**目的**: BC25 5th 流 / exp017/exp020 標準の **Power Transform γ=1.2** を pseudo CSV に適用、Aves negative signal を維持しつつ weak noise suppress。exp045 (chunk filter > 0.2) の drag (-0.0062) を回避する soft alternative。

## 設計動機 (exp045 失敗からの設計)

- **exp045 結果**: chunk filter > 0.2 で val 0.9347 (baseline 0.9409 -0.0062)、Aves AUC plateau 0.85 = silence chunk drop で 140 種 Aves negative signal 損失
- **γ=1.2 解**: silence chunk **保持** (prob 0.05 → 0.029、依然 silence 値)、weak noise **soft suppress**、Aves negative 維持
- **lineage 整合**: exp017/exp020 self-distill lineage 全て γ=1.2、唯一 exp029 R3 が γ=1.0 raw = 設計漏れ補正

## 設定

- **base**: exp029 R3 spec 完全複製 (eca_nfnet_l1 + exp017 R2 teacher pseudo + Perch distill + 20 ep + same config)
- **唯一の変更**: `POWER_GAMMA = 1.2` を pseudo CSV 読込直後に適用 (`prob_mat = prob_mat ^ 1.2`)
- **chunk drop なし**: 全 chunk 保持、silence chunk も学習に使用
- **fold**: FOLDS=[0] single fold
- **比較**: 既存 exp029 R3 fold 0 val_ns22 = **0.9409** (ep15、history.json 確定) vs exp046 (γ=1.2)

## 期待結果

| Outcome | 解釈 |
|---|---|
| delta > +0.003 (val ≥ 0.9439) | **γ=1.2 有効**、exp029 R3 5-fold 再学習で +0.005-0.010 LB |
| -0.003 ≤ delta ≤ +0.003 | noise floor、効果不明 (filter ablation と同様) |
| delta < -0.003 | drag、撤退 |

## 期待 LB lift シナリオ

| exp046 単 fold val (best) | 5-fold avg val 推定 | LB 推定 |
|---|---|---|
| 0.9419 (+0.001) | 0.9368 | 0.924 (+0.001) |
| 0.9439 (+0.003 有効) | 0.9388 | 0.926 (+0.003) |
| 0.9459 (+0.005 楽観) | 0.9408 | 0.928 (+0.005) |
| 0.9509 (+0.010 極楽観) | 0.9458 | 0.933 |
| 0.9409 (変化なし) | 0.9358 | 0.923 |

## 出力

- Drive: `output/exp046/fold0/r3/ckpt_best_ns22.pth`
- Kaggle Dataset: `maekeso/birdclef2026-exp046-l1-gamma12`

## 観察 metrics (exp045 と同)

- val_ns22 / val_macro (epoch 別)
- per_taxon (Aves/Amphibia/Insecta/Mammalia/Reptilia) ← **Aves が baseline 並 0.92+ に達するか要監視**
- per_class distribution (n_valid, median, p25, p75, #>0.5/0.7/0.9/perfect)
- train_loss / cls_loss / dist_loss
- epoch time

## 前提

- `pseudo_e17.csv` (exp028 e17 pseudo、exp017 R2 teacher、raw γ=1.0) を Drive `kaggle/birdclef2026/pseudo_e17.csv` に配置済
- Perch cache `kaggle/birdclef2026/perch-cache/{emb.npy, meta.csv}`
"""
replace_cell_source("hdr", NEW_HDR)
print("[hdr] OK rewritten for γ=1.2 ablation")


# ---- 2. setup: exp045 -> exp046 ----
patch_cell_source(
    "setup",
    'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp045"',
    'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp046"',
)
print("[setup] OK DRIVE_EXP_DIR -> exp046")


# ---- 3. config: CHUNK_FILTER_MAX_PROB -> POWER_GAMMA ----
# Replace the CHUNK_FILTER_MAX_PROB block with POWER_GAMMA block
config_old = """# ★ exp045 Ablation: paper 256 流 chunk filter (max_prob > 0.5 で teacher pseudo の chunk drop)
# CHUNK_FILTER_MAX_PROB = 0.0 で disable (exp029 R3 baseline と同等)
# CHUNK_FILTER_MAX_PROB = 0.2 で paper 256 spec (Sydorskyi 2nd place の primary_label_min_prob)
CHUNK_FILTER_MAX_PROB = 0.2"""

config_new = """# ★ exp046 Ablation: Power Transform γ=1.2 (exp017/exp020/BC25 5th 標準、self-distill lineage 整合)
# POWER_GAMMA = 1.0 で identity (exp029 R3 baseline と同等)
# POWER_GAMMA = 1.2 で BC25 5th flavor (silence 保持 soft suppress、Aves negative 維持)
# 注: chunk drop は使わない (exp045 で drag 確認、silence 必要)
POWER_GAMMA = 1.2"""

patch_cell_source("config", config_old, config_new)
print("[config] OK CHUNK_FILTER_MAX_PROB -> POWER_GAMMA = 1.2")


# ---- 4. fold_loop: replace chunk filter with Power Transform ----
filter_old = """    print(f"  Loading teacher pseudo (exp017 R2): {DRIVE_TEACHER_PSEUDO}")
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
"""

filter_new = """    print(f"  Loading teacher pseudo (exp017 R2): {DRIVE_TEACHER_PSEUDO}")
    pseudo_df_r1 = pd.read_csv(DRIVE_TEACHER_PSEUDO)
    print(f"    {len(pseudo_df_r1)} rows (raw)")

    # ★ exp046: Power Transform γ (BC25 5th 流、exp017/exp020 標準)
    # silence chunk 保持で Aves negative signal 維持 (exp045 で chunk drop が drag)
    if POWER_GAMMA != 1.0:
        prob_cols = [c for c in pseudo_df_r1.columns if c in PRIMARY_LABELS]
        assert len(prob_cols) == NUM_CLASSES, f"Expected {NUM_CLASSES} prob cols, got {len(prob_cols)}"
        prob_mat_before = pseudo_df_r1[prob_cols].values.astype(np.float32)
        max_before_50 = np.percentile(prob_mat_before.max(axis=1), 50)
        max_before_90 = np.percentile(prob_mat_before.max(axis=1), 90)
        # Power Transform: prob^γ
        prob_mat_after = np.power(prob_mat_before, POWER_GAMMA).astype(np.float32)
        max_after_50 = np.percentile(prob_mat_after.max(axis=1), 50)
        max_after_90 = np.percentile(prob_mat_after.max(axis=1), 90)
        pseudo_df_r1[prob_cols] = prob_mat_after
        print(f"    [Power Transform γ={POWER_GAMMA}] all {len(pseudo_df_r1)} rows kept (no chunk drop)")
        print(f"      max_prob 50%ile {max_before_50:.3f} -> {max_after_50:.3f}")
        print(f"      max_prob 90%ile {max_before_90:.3f} -> {max_after_90:.3f}")
        print(f"      effect: {'sharpen (suppress weak)' if POWER_GAMMA > 1.0 else 'smooth (amplify weak)'}")
    else:
        print(f"    [Power Transform DISABLED] POWER_GAMMA={POWER_GAMMA} (identity)")
"""

patch_cell_source("fold_loop", filter_old, filter_new)
print("[fold_loop] OK chunk filter -> Power Transform")


# ---- 5. upload: SLUG/TITLE -> exp046 ----
patch_cell_source(
    "upload",
    'SLUG  = "birdclef2026-exp045-l1-filtered"\nTITLE = "birdclef2026 exp045 l1 filtered ablation"',
    'SLUG  = "birdclef2026-exp046-l1-gamma12"\nTITLE = "birdclef2026 exp046 l1 gamma12 ablation"',
)
patch_cell_source(
    "upload",
    'version_notes = f"exp045 l1 chunk-filter ablation (filter={CHUNK_FILTER_MAX_PROB}) | {summary}"[:498]',
    'version_notes = f"exp046 l1 power-transform ablation (gamma={POWER_GAMMA}) | {summary}"[:498]',
)
print("[upload] OK SLUG -> exp046, version_notes -> gamma=...")


# ---- Save ----
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nDONE: saved {NB_PATH}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c.get('cell_type'):8s} id={c.get('id'):20s} L={n}")
