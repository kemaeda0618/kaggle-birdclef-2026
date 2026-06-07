"""Build exp097 NB by tuning tax_smoothing alphas on exp090.

Change: f_tax_smoothing call uses genus_alpha=0.25 (was 0.15), class_alpha=0.10 (was 0.05).
Single 1-line edit; all other PP unchanged.

This is a defensive ablation around tax smoothing strength.
- exp090 (LB 0.951) baseline = 0.15/0.05
- Larger α = more aggressive cross-genus/class probabilistic mixing
- Risk: over-flatten species-level signals
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp097_tax_alpha.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Find blend cell
blend_cell = None
for c in nb["cells"]:
    if c.get("id") == "blend":
        blend_cell = c
        break
assert blend_cell is not None

blend_src = "".join(blend_cell["source"])
print(f"blend cell original length: {len(blend_src)}")

OLD = "submission = f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05)"
NEW = "submission = f_tax_smoothing(submission, genus_alpha=0.25, class_alpha=0.10)  # ★ exp097: alpha extended"

if OLD not in blend_src:
    raise RuntimeError(f"Anchor not found: {OLD}")
new_blend_src = blend_src.replace(OLD, NEW)
assert new_blend_src != blend_src
blend_cell["source"] = new_blend_src.splitlines(keepends=True)
print(f"blend cell new length: {len(new_blend_src)}")
print(f"diff: +{len(new_blend_src) - len(blend_src)} chars")

# Update header
hdr_cell = None
for c in nb["cells"]:
    if c.get("id") == "hdr":
        hdr_cell = c
        break
assert hdr_cell is not None

new_hdr = '''# exp097 — exp090 + Tax smoothing α 拡張 (0.25 / 0.10)

## 変更点 (vs exp090)
- exp090 (LB 0.951) stack 完全踏襲: NB4 v7 0.35 / Tucker SED 0.40 / exp029 R3 0.25 + lambda_prior 0.65 + rank_aware power 0.6
- **唯一の差**: `f_tax_smoothing(genus_alpha=0.25, class_alpha=0.10)` (exp090 は 0.15 / 0.05)

## Rationale
- Tax smoothing は **multi-column** operation (per-genus / per-class 平均との mix) なので AUC を変える
- exp090 の α=0.15/0.05 が必ずしも optimum ではない、aggressive 側に動かして ablation
- 1 行差なので drag/lift が PP 効果として cleanly isolate される

## Expected
- LB **0.949-0.953** (exp090 から +0.000-0.002 or -0.002)
- 改善確率: **40%**
- drag 確率: **30%** (species 信号の over-flatten で)
- ablation 価値: **HIGH** (どちら方向に動くか分かれば次の tuning 判断材料に)

## Inputs (exp090 と完全同一)
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`
- `maekeso/birdclef2026-exp029-l1-single`
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
'''
hdr_cell["source"] = new_hdr.splitlines(keepends=True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[OK] Saved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
