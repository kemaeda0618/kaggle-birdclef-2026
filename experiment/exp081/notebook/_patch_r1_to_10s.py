"""Patch exp081 R1 NB: 5s → 10s window (exp017 R1 base).

R1 differences from R2:
- No PseudoScDS (R1 doesn't use pseudo)
- SHARES = {focal: 0.85, labeled_sc: 0.15} (no pseudo_sc)
- Pseudo_infer cell generates R1 pseudo (input for R2)
- No pseudo aggregation needed in load_data

Changes:
1. Cell 0 (hdr): update title to exp081 R1
2. Cell 1 (setup): output paths to exp081/r1/ + Kaggle Dataset slug
3. Cell 3 (config): TRAIN_DURATION/VAL_DURATION 5 → 10
4. Cell 16 (pseudo_infer): generate 10s pseudo for next R2 round
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("nb_train_r1.ipynb")
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


# ===== 1. hdr cell =====
hdr = find_cell(nb["cells"], "hdr")
if hdr is not None:
    new_md = """# exp081 R1 — exp017 R1 + 10s window (Colab Pro G4)

**exp017 R1 spec を完全踏襲、Window 5s → 10s に変更のみ**

## Changes from exp017 R1
- `TRAIN_DURATION` / `VAL_DURATION`: 5 → **10**
- `TRAIN_SAMPLES` = 320,000 (10s × 32kHz)
- Output paths: `output/exp081/r1/` (Drive)、`maekeso/birdclef2026-exp081-weights` (Kaggle Dataset)
- Pseudo gen: 10s native (12 windows per file with stride 5s + overlap)

## 不変項目 (exp017 R1 と同)
- Backbone: eca_nfnet_l0
- LR 3e-4 (NFNet rule、sqrt scaling 厳禁)
- Batch 192 (G4 96GB 余裕)
- N_TOTAL_EPOCHS 25 (R1 は longer)
- WARMUP 3
- ALPHA_DISTILL 1.0、Perch v2 distill
- SHARES focal 0.85 / labeled_sc 0.15 (★ R1 = no pseudo)
- Mixup α 0.4、SpecAug F=25/T=30 × 2

## R1 → R2 flow

```
R1: focal + labeled_sc only (no pseudo) → exp081 R1 ckpt
R1 pseudo gen: model.predict(soundscape) → pseudo_labels.csv (10s native)
R2 (別 NB): focal + labeled_sc + R1 pseudo → exp081 R2 ckpt (final)
```

## 想定時間 (G4 96GB Blackwell)
- 学習 25 epoch: ~5-6h (5s exp017 R1 43min の 10x、focal load も 2x)
- pseudo gen on soundscape: ~30 min
- 重み upload: 1-2 min
- 合計: ~6-7h

## 期待
- val_ns22 ~0.91-0.93 (exp017 R1 0.9166 から +0.005-0.015 期待、temporal context 効)
- R1 単独 sub はしない (R2 input 用)
"""
    hdr["source"] = new_md.splitlines(keepends=True)
    print("[hdr] Updated to exp081 R1")

# ===== 2. setup cell =====
setup = find_cell(nb["cells"], "setup")
if setup is not None:
    patch_source(setup, [
        ("exp017", "exp081"),
    ])
    print("[setup] Updated paths to exp081")

# ===== 3. config cell =====
config = find_cell(nb["cells"], "config")
if config is not None:
    patch_source(config, [
        ("TRAIN_DURATION = 5", "TRAIN_DURATION = 10   # ★ exp081: 5 → 10"),
        ("VAL_DURATION   = 5", "VAL_DURATION   = 10   # ★ exp081: 5 → 10"),
    ])
    print("[config] WINDOW 5 → 10")

# ===== 4. upload cell =====
upload = find_cell(nb["cells"], "upload")
if upload is not None:
    patch_source(upload, [
        ("birdclef2026-exp017-weights", "birdclef2026-exp081-weights"),
        ("exp017", "exp081"),
    ])
    print("[upload] Kaggle Dataset slug updated")

# ===== 5. pseudo_infer cell: ★ 10s native pseudo gen =====
pseudo_infer = find_cell(nb["cells"], "pseudo_infer")
if pseudo_infer is not None:
    patch_source(pseudo_infer, [
        # N_WINDOWS = 12 (5s × 12 = 60s) → 6 (10s × 6 = 60s)
        ("N_WINDOWS = 12", "N_WINDOWS = 6   # ★ exp081: 10s window で 6 chunks/file"),
    ])
    print("[pseudo_infer] N_WINDOWS 12 → 6 (10s native pseudo)")

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print(f"  cells: {len(nb['cells'])}")
