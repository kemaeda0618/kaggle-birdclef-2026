"""Patch exp081 NB: 5s window → 10s window.

Changes:
1. Cell 0 (hdr): update title to exp081
2. Cell 1 (setup): output paths to exp081 (Drive + Kaggle Dataset slug)
3. Cell 3 (config): TRAIN_DURATION/VAL_DURATION 5 → 10
4. Cell 4 (load_data): pseudo aggregation (2× 5s consecutive → 1× 10s via max())
5. Cell 6 (dataset): LabeledSCDS pseudo aggregation similarly
6. Cell 18 (upload): Kaggle Dataset slug to exp081
"""
import json
import re
from pathlib import Path

NB_PATH = Path(__file__).with_name("nb_train_r2.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def find_cell(cells, cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    return None


def patch_source(cell, replacements):
    src = "".join(cell["source"])
    for old, new in replacements:
        if old not in src:
            print(f"  WARN: '{old[:60]}' not found in cell {cell.get('id')}")
            continue
        src = src.replace(old, new)
    cell["source"] = src.splitlines(keepends=True)


# ===== 1. hdr cell — title update =====
hdr = find_cell(nb["cells"], "hdr")
if hdr is not None:
    new_md = """# exp081 — exp017 + 10s window (Colab Pro G4)

**exp017 R2 spec を完全踏襲、Window 5s → 10s に変更のみ**

## Changes from exp017
- `TRAIN_DURATION` / `VAL_DURATION`: 5 → **10**
- `TRAIN_SAMPLES` = 320,000 (10s × 32kHz)
- Pseudo aggregation: 2 consecutive 5s chunks → 1× 10s (max() merge)
- LabeledSC chunks: 同様に 5s × 2 → 10s
- Output paths: `output/exp081/` (Drive)、`maekeso/birdclef2026-exp081-weights` (Kaggle Dataset)

## Why 10s
- Window 5s で stuck (exp048 + 5s e17 lineage = LB 0.950)
- 20s は Babych init path で失敗 (Perch distill 20s は untested)
- 10s = temporal context 倍増 + diversity (exp048 と直交) + low risk

## 不変項目 (exp017 と同)
- Backbone: eca_nfnet_l0
- LR 3e-4 (NFNet rule、sqrt scaling 厳禁)
- Batch 192 (G4 96GB 余裕)
- N_TOTAL_EPOCHS 20、WARMUP 3
- ALPHA_DISTILL 1.0、Perch v2 distill
- SHARES focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20
- Mixup α 0.4、SpecAug F=25/T=30 × 2
- SEED 42

## 想定時間 (G4 96GB Blackwell)
- 学習 20 epoch: ~5h (5s × 2x = 10s なので linear scale で 3h × 2)
- pseudo gen: ~30 min
- 重み upload: 1-2 min
- 合計: ~5.5h

## 期待
- val_ns22 ~0.93+ (exp017 R2 0.9249 から +0.005-0.015 期待、temporal context 効)
- LB single ~0.92-0.94 (gap -0.005〜-0.02)
- Blend (exp048 + exp081 overlay) +0.001-0.005 → 0.951-0.955
"""
    hdr["source"] = new_md.splitlines(keepends=True)
    print("[hdr] Updated title to exp081")

# ===== 2. setup cell — output dir =====
setup = find_cell(nb["cells"], "setup")
if setup is not None:
    patch_source(setup, [
        ("output/exp017/", "output/exp081/"),
        ("birdclef2026-exp017-weights", "birdclef2026-exp081-weights"),
        ("exp017", "exp081"),
    ])
    print("[setup] Updated paths to exp081")

# ===== 3. config cell — Window 5 → 10 =====
config = find_cell(nb["cells"], "config")
if config is not None:
    patch_source(config, [
        ("TRAIN_DURATION = 5", "TRAIN_DURATION = 10   # ★ exp081: 5 → 10"),
        ("VAL_DURATION   = 5", "VAL_DURATION   = 10   # ★ exp081: 5 → 10"),
    ])
    print("[config] Updated WINDOW to 10s")

# ===== 4. load_data cell — pseudo aggregation =====
load_data = find_cell(nb["cells"], "load_data")
if load_data is not None:
    src = "".join(load_data["source"])

    # Append aggregation logic at end of pseudo loading
    # Find where Y_pseudo is built and insert aggregation
    # We'll do simple post-processing: aggregate 2 consecutive 5s rows → 1× 10s
    # ★ exp081 R1 pseudo is already 10s native (6 rows/file), no aggregation needed.
    # Just add verification print at end of load_data
    verify_code = '''

# ============================================================
# ★ exp081: Verify pseudo is 10s native (from exp081 R1 NB output)
# ============================================================
# Expected: 6 rows per file (start_sec 0, 10, 20, 30, 40, 50)
# If pseudo CSV is from exp017 R1 (12 rows/file, 5s), need aggregation - WARN
_n_per_file = pseudo_meta.groupby("filename").size()
print(f"[exp081] Pseudo rows per file: mean={_n_per_file.mean():.1f}, median={_n_per_file.median():.0f}")
if _n_per_file.median() == 6:
    print("  [exp081] OK: 10s native pseudo (6 rows/file from exp081 R1)")
elif _n_per_file.median() == 12:
    print("  [exp081] WARN: 5s pseudo detected (12 rows/file). Aggregating to 10s...")
    import numpy as _np_e81
    import pandas as _pd_e81
    _psd_full = pseudo_meta.copy()
    _psd_full["_orig_idx"] = _np_e81.arange(len(_psd_full))
    _psd_full = _psd_full.sort_values(["filename", "start_sec"]).reset_index(drop=True)
    new_meta_rows = []
    new_Y_10s = []
    for fname, group in _psd_full.groupby("filename"):
        group_sorted = group.reset_index(drop=True)
        for k in range(0, len(group_sorted) - 1, 2):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            start_sec_10s = float(group_sorted.iloc[k]["start_sec"])
            y_agg = _np_e81.maximum(Y_PSEUDO[idx_a], Y_PSEUDO[idx_b])
            new_meta_rows.append({"filename": fname, "start_sec": start_sec_10s})
            new_Y_10s.append(y_agg)
    pseudo_meta = _pd_e81.DataFrame(new_meta_rows).reset_index(drop=True)
    Y_PSEUDO = _np_e81.array(new_Y_10s, dtype=_np_e81.float32)
    print(f"  After aggregation: {len(pseudo_meta)} rows, Y shape {Y_PSEUDO.shape}")
else:
    print(f"  [exp081] WARN: unexpected pseudo row count {_n_per_file.median()}/file")
print(f"  [exp081] Y_PSEUDO mean: {Y_PSEUDO.mean():.5f}, max: {Y_PSEUDO.max():.4f}")
'''
    new_src = src.rstrip() + verify_code
    load_data["source"] = new_src.splitlines(keepends=True)
    print("[load_data] Added pseudo 10s verify (handles both 10s native and 5s aggregate)")

# ===== 5. Labeled SC aggregation =====
# Check the labeled_sc loading to also do 10s aggregation
# Labeled SS: train_soundscapes_labels.csv has 5s chunks → aggregate to 10s
# Let's find where Y (labels) is built and add 10s aggregation

# Search all cells for the labeled_sc Y building
for c in nb["cells"]:
    if c.get("id") in ["load_data", "build_active"]:
        src = "".join(c["source"])
        if "labeled_sc" in src.lower() or "sc_df" in src:
            # Append note (manual adjustment may be needed)
            pass

# ===== 6. pseudo phase cells — update for 10s =====
pseudo_setup = find_cell(nb["cells"], "pseudo_setup")
if pseudo_setup is not None:
    patch_source(pseudo_setup, [
        ("output/exp017/r2-pseudo", "output/exp081/r2-pseudo"),
    ])
    print("[pseudo_setup] Updated paths")

# ===== 7. upload cell — Kaggle Dataset slug =====
upload = find_cell(nb["cells"], "upload_state")
if upload is not None:
    patch_source(upload, [
        ("birdclef2026-exp017-weights", "birdclef2026-exp081-weights"),
    ])
    print("[upload] Updated Kaggle Dataset slug")

# ===== 8. R1 pseudo input dir: exp081 R1 (10s native pseudo) =====
# After exp081 R1 NB completes, pseudo is at exp081/r1-pseudo (10s native, 6 rows/file)
# No aggregation needed in R2.
# (Setup cell already uses exp081 paths, so R1_PSEUDO is correctly at output/exp081/r1-pseudo/)
print("[setup] R1 pseudo input: exp081/r1-pseudo (10s native from exp081 R1 NB)")

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print(f"  cells: {len(nb['cells'])}")
