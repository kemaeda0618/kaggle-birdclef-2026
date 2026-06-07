"""Build exp101: exp090 base + exp029 R3 slot → exp100 R3 (l1 + Babych spec) ckpt swap.

Both are l1 backbone, so BACKBONE unchanged. Only dataset path differs.

Changes vs exp090:
  - e29-infer cell dataset path:
    birdclef2026-exp029-l1-single -> birdclef2026-exp100-l1-babych
  - ckpt filename glob unchanged: r3_fold*_ckpt_best_ns22.pth
  - Header / comments updated

Honest expectation:
  exp100 fold 0 val 0.9251 vs exp029 R3 5-fold avg 0.9358 = -0.011 lower
  Stack LB likely -0.001 to -0.003 drag (0.948-0.950 predicted)
  Diagnostic value: measure exp100 spec effect at stream level
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp101_e100_swap.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

target = None
for c in nb["cells"]:
    if c.get("id") == "e29-infer":
        target = c
        break
assert target is not None, "e29-infer cell not found"

src = "".join(target["source"])
print(f"e29-infer cell original length: {len(src)}")

patches = [
    ('Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single")',
     'Path("/kaggle/input/datasets/maekeso/birdclef2026-exp100-l1-babych")'),
    ('Path("/kaggle/input/birdclef2026-exp029-l1-single")',
     'Path("/kaggle/input/birdclef2026-exp100-l1-babych")'),
    ('# === exp029 R3 (eca_nfnet_l1 + Perch distill) inference ===',
     '# === exp101: exp100 R3 (eca_nfnet_l1 + Babych spec + Perch distill) inference ===  # ckpt swap from exp029 R3'),
    ('"r3_fold0_ckpt_best_ns22.pth",   # exp029 R3 fold0 (val 0.9409, LB 0.923)',
     '"r3_fold0_ckpt_best_ns22.pth",   # ★ exp101: exp100 R3 fold0 (l1+Babych, val 0.9251 @ ep20)'),
]
n_applied = 0
for old, new in patches:
    if old in src:
        n_applied += src.count(old)
        src = src.replace(old, new)
        print(f"  [OK] {old[:80]!r}")
    else:
        print(f"  [MISS] {old[:80]!r}")
assert n_applied == len(patches), f"Some patches missed: {n_applied}/{len(patches)}"

target["source"] = src.splitlines(keepends=True)
print(f"e29-infer cell new length: {len(src)}")

# Update header
hdr = None
for c in nb["cells"]:
    if c.get("id") == "hdr":
        hdr = c
        break
assert hdr is not None
new_hdr = '''# exp101 — exp090 stack で exp029 R3 (5-fold avg) を exp100 R3 (l1 + Babych spec、single fold) に差し替え

## 目的
- exp100 = l1 + Babych spec (mixup α=1.0, drop_path 0.15, SHARES 0.60/0.10/0.30, per-class WRS) で train した R3 single fold ckpt
- 同 fold 0 比較で exp100 val 0.9251 vs exp029 R3 fold 0 0.9177 = **+0.0074** (Babych spec の clean effect)
- ただし exp090 stack で使われていた exp029 R3 は 5-fold avg (val 0.9358)、single fold 置換で **stream level -0.011 drop**

## 変更点 (vs exp090, LB 0.951)
- **唯一の差**: e29-infer cell の ckpt swap
  - Dataset:  `birdclef2026-exp029-l1-single` → `birdclef2026-exp100-l1-babych`
  - BACKBONE: `eca_nfnet_l1` (両者同じ、変更なし)
  - ckpt file 名は同じ (`r3_fold0_ckpt_best_ns22.pth`)
- PP / blend weights / lambda_prior 0.65 / rank_aware 0.6 / TAX_SMOOTHING 0.15/0.05: 不変

## Expected
- LB **0.948-0.950** (exp090 0.951 から -0.001 〜 -0.003 と予想)
- 改善確率: **15-20%** (Babych spec の test-domain robustness が val gap を縮める可能性)
- drag 確率: **65%** (single fold で variance 大きい)

## Diagnostic 価値
- もし LB ≥ 0.950 なら exp100 spec の effect が test 域でも有効 → **5-fold ensemble で確実 gold 射程**
- LB < 0.948 なら exp100 spec は val gap 大、5-fold ensemble でも頭打ち可能性

## Inputs (exp090 から exp029 → exp100 置換)
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`
- **`maekeso/birdclef2026-exp100-l1-babych`** ★ swap target
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
'''
hdr["source"] = new_hdr.splitlines(keepends=True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[OK] Saved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
