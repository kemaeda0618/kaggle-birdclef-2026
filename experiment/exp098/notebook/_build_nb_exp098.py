"""Build exp098 NB: exp090 base + exp029 R3 slot → exp095 R3 (l0 + R2 spec) ckpt swap.

Changes (e29-infer cell only):
  1. Dataset path: birdclef2026-exp029-l1-single → birdclef2026-exp095-l0-r2spec
  2. BACKBONE   : eca_nfnet_l1 → eca_nfnet_l0
  3. Comments updated for clarity

Other layers (gem_freq, dense hidden_dim=512, att/cla, distill_head)
use backbone_dim auto-computed via dummy forward -> auto-adapt to l0.
ckpt filename "r3_fold0_ckpt_best_ns22.pth" is same (training template identical).

This is a diagnostic submit to measure whether l0 R3 (val 0.9187)
strongly drags vs exp029 R3 (val 0.9358 / LB 0.923) in the exp090 stack.
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp098_e095_swap.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# === Find e29-infer cell ===
target = None
for c in nb["cells"]:
    if c.get("id") == "e29-infer":
        target = c
        break
assert target is not None, "e29-infer cell not found"

src = "".join(target["source"])
print(f"e29-infer cell original length: {len(src)}")

patches = [
    # 1. Dataset paths (TWO occurrences)
    ('Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single")',
     'Path("/kaggle/input/datasets/maekeso/birdclef2026-exp095-l0-r2spec")'),
    ('Path("/kaggle/input/birdclef2026-exp029-l1-single")',
     'Path("/kaggle/input/birdclef2026-exp095-l0-r2spec")'),

    # 2. BACKBONE
    ('E17_BACKBONE = "eca_nfnet_l1"  # exp029 student (l0 から +30% params)',
     'E17_BACKBONE = "eca_nfnet_l0"  # ★ exp098: exp095 R3 (l0 + R2 spec, val 0.9187)'),

    # 3. Comment header
    ('# === exp029 R3 (eca_nfnet_l1 + Perch distill) inference ===',
     '# === exp098: exp095 R3 (eca_nfnet_l0 + R2 spec + Perch distill) inference ===  # ckpt swap from exp029 R3'),

    # 4. ckpt name comment
    ('"r3_fold0_ckpt_best_ns22.pth",   # exp029 R3 fold0 (val 0.9409, LB 0.923)',
     '"r3_fold0_ckpt_best_ns22.pth",   # ★ exp098: exp095 R3 fold0 (l0, val 0.9187 @ ep8)'),
]

n_applied = 0
for old, new in patches:
    if old in src:
        n_applied += src.count(old)
        src = src.replace(old, new)
        print(f"  [OK] {old[:80]!r}")
    else:
        print(f"  [MISS] {old[:80]!r}")
        # Show closest for debug
        for ln in src.splitlines():
            if any(kw in ln for kw in old.split()[:3]):
                print(f"    cand: {ln.strip()[:120]}")
print(f"Total patches applied: {n_applied}/{len(patches)}")
assert n_applied == len(patches), "Some patches missed!"

target["source"] = src.splitlines(keepends=True)
print(f"e29-infer cell new length: {len(src)}")

# === Update header ===
hdr = None
for c in nb["cells"]:
    if c.get("id") == "hdr":
        hdr = c
        break
assert hdr is not None

new_hdr = '''# exp098 — exp090 stack で exp029 R3 (l1) を exp095 R3 (l0 + R2 spec) に差し替え (diagnostic sub)

## 目的
- exp095 R3 = l0 + R2 exact spec ckpt の val_ns22 = 0.9187 (exp029 R3 0.9358 から -0.017)
- これが exp090 stack 内 (w=0.25 slot) で **どれくらい LB を drag するか** を実測
- l0 R3 path の死活判定

## 変更点 (vs exp090, LB 0.951)
- **唯一の差**: e29-infer cell の ckpt swap
  - Dataset:  `maekeso/birdclef2026-exp029-l1-single` → `maekeso/birdclef2026-exp095-l0-r2spec`
  - BACKBONE: `eca_nfnet_l1` → `eca_nfnet_l0`
  - ckpt file 名は同じ (`r3_fold0_ckpt_best_ns22.pth`)
- その他 (PP, blend weights, lambda_prior 0.65, rank_aware 0.6, TAX_SMOOTHING 0.15/0.05): 不変

## Expected
- LB **0.940-0.948** (exp090 0.951 から -0.003 〜 -0.011 と予想)
- 改善確率: **15%** (val gap -0.017 を blend で吸収できる僅かな可能性)
- drag 確率: **80%**

## Diagnostic 価値
- もし LB ≥ 0.948 なら l0 R3 path は dilute されつつ生存 → 4-way blend で diversity 候補
- LB < 0.945 なら l0 R3 path 棄却確定、exp029 R3 (l1) のみ採用

## Inputs (exp090 から `exp029-l1-single` を `exp095-l0-r2spec` に置換)
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`
- **`maekeso/birdclef2026-exp095-l0-r2spec`** ★ swap target
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
'''
hdr["source"] = new_hdr.splitlines(keepends=True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[OK] Saved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
