"""Build exp090 NB by modifying exp078:
1. exp037_full_pipeline cell:
   - apply_prior: lambda_prior=0.4 → lambda_prior=0.65 (2 calls)
   - rank_aware_scaling: add power=0.6 (default 0.4)
2. blend cell:
   - Add f_tax_smoothing after Sonotype mirror, before submission.to_csv
3. Update hdr cell

FCS is NOT touched (memory warning about overcalibration).
"""
import json, io, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp090_tuned_tax.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Apply changes to exp037_full_pipeline cell
# ============================================================
target_cell = None
for c in nb["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        target_cell = c
        break
assert target_cell is not None, "exp037_full_pipeline cell not found"

src = "".join(target_cell["source"])
print("Original cell length:", len(src))

# Change 1: lambda_prior=0.4 → 0.65 (2 calls)
new_src = src.replace("lambda_prior=0.4", "lambda_prior=0.65")
n_replaced_lambda = src.count("lambda_prior=0.4")
print(f"lambda_prior=0.4 → 0.65: replaced {n_replaced_lambda} occurrences")
assert n_replaced_lambda == 2, f"Expected 2 lambda_prior calls, found {n_replaced_lambda}"

# Change 2: rank_aware_scaling — add power=0.6
# Original: rank_aware_scaling(probs, n_windows=N_WINDOWS,
# Target:   rank_aware_scaling(probs, n_windows=N_WINDOWS, power=0.6,
old_rank = "rank_aware_scaling(   probs, n_windows=N_WINDOWS,"
new_rank = "rank_aware_scaling(   probs, n_windows=N_WINDOWS, power=0.6,"
if old_rank in new_src:
    new_src = new_src.replace(old_rank, new_rank)
    print("rank_aware_scaling: added power=0.6")
else:
    # Try alternative pattern
    alt_old = "rank_aware_scaling(probs, n_windows=N_WINDOWS,"
    alt_new = "rank_aware_scaling(probs, n_windows=N_WINDOWS, power=0.6,"
    if alt_old in new_src:
        new_src = new_src.replace(alt_old, alt_new)
        print("rank_aware_scaling (alt pattern): added power=0.6")
    else:
        # Print actual call for debug
        for ln in src.split("\n"):
            if "rank_aware_scaling(" in ln and "def " not in ln:
                print(f"DEBUG actual call: {repr(ln)}")
        raise RuntimeError("rank_aware_scaling call not found in expected pattern")

target_cell["source"] = new_src.splitlines(keepends=True)
print(f"Updated exp037_full_pipeline cell, new length: {len(new_src)}")

# ============================================================
# Apply changes to blend cell (add TAX_SMOOTHING)
# ============================================================
blend_cell = None
for c in nb["cells"]:
    if c.get("id") == "blend":
        blend_cell = c
        break
assert blend_cell is not None, "blend cell not found"

blend_src = "".join(blend_cell["source"])
print(f"\nblend cell original length: {len(blend_src)}")

# Insert TAX_SMOOTHING right before submission.to_csv
# Original: submission.to_csv("submission.csv", index=False)
old_save = 'submission.to_csv("submission.csv", index=False)'

tax_smoothing_code = '''# === ★ exp090: Taxonomy smoothing post-processing (from anthonytherrien 0.950 NB) ===
# Genus-level smoothing (α=0.15) + Class-level smoothing (α=0.05)
# Expected lift: +0.001-0.002 (independent post-process after blend)
def f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05):
    """Smooth predictions within taxonomic genera and classes."""
    tax_paths = [
        Path("/kaggle/input/competitions/birdclef-2026/taxonomy.csv"),
        Path("/kaggle/input/birdclef-2026/taxonomy.csv"),
    ]
    taxonomy_df = None
    for p in tax_paths:
        if p.exists():
            taxonomy_df = pd.read_csv(p)
            break
    if taxonomy_df is None:
        print("[tax_smooth] taxonomy.csv not found — skip")
        return submission

    species_to_genus = {}
    species_to_class = {}
    for _, row in taxonomy_df.iterrows():
        label = str(row["primary_label"])
        sci = str(row.get("scientific_name", ""))
        cls = str(row.get("class_name", ""))
        genus = sci.split(" ")[0] if " " in sci else sci
        species_to_genus[label] = genus
        species_to_class[label] = cls

    label_cols = [c for c in submission.columns if c != "row_id"]
    genus_groups = {}
    class_groups = {}
    for col in label_cols:
        g = species_to_genus.get(col, col)
        c = species_to_class.get(col, "")
        genus_groups.setdefault(g, []).append(col)
        if c:
            class_groups.setdefault(c, []).append(col)
    multi_genus = {k: v for k, v in genus_groups.items() if len(v) > 1}
    multi_class = {k: v for k, v in class_groups.items() if len(v) > 1}
    print(f"[tax_smooth] pre: mean={submission[label_cols].values.mean():.6f}")
    print(f"[tax_smooth] multi-genus groups: {len(multi_genus)}, multi-class: {len(multi_class)}")

    probs = submission[label_cols].values.astype(np.float32).copy()
    col_to_idx = {c: i for i, c in enumerate(label_cols)}

    # Genus smoothing
    for _, members in multi_genus.items():
        idx = [col_to_idx[m] for m in members]
        g_mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1 - genus_alpha) * probs[:, idx] + genus_alpha * g_mean

    # Class smoothing
    for _, members in multi_class.items():
        idx = [col_to_idx[m] for m in members]
        c_mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1 - class_alpha) * probs[:, idx] + class_alpha * c_mean

    submission[label_cols] = probs
    print(f"[tax_smooth] post: mean={submission[label_cols].values.mean():.6f}")
    return submission

# Apply tax smoothing
submission = f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05)

''' + old_save

if old_save not in blend_src:
    raise RuntimeError(f"Save pattern not found: {old_save}")

new_blend_src = blend_src.replace(old_save, tax_smoothing_code)
blend_cell["source"] = new_blend_src.splitlines(keepends=True)
print(f"blend cell new length: {len(new_blend_src)}")
print(f"TAX_SMOOTHING insertion: +{len(new_blend_src) - len(blend_src)} chars")

# ============================================================
# Update header cell
# ============================================================
hdr_cell = None
for c in nb["cells"]:
    if c.get("id") == "hdr":
        hdr_cell = c
        break
assert hdr_cell is not None

new_hdr_src = '''# exp090 — exp078 (LB 0.950) + lambda_prior 0.65 + rank_aware power 0.6 + TAX_SMOOTHING

★ **exp078 (LB 0.950) を base に、anthonytherrien 0.950 NB で実証された 3 改善を適用**

## 変更点 (vs exp078)

| 項目 | exp078 (現 best) | exp090 (改善版) | 出典 |
|---|---|---|---|
| `lambda_prior` (Hour/Site prior) | 0.4 (default) | **0.65** ★ | anthony Model_51 |
| `rank_aware_scaling(power)` | 0.4 (default) | **0.6** ★ | anthony Model_51 |
| `file_confidence_scale(power)` | 0.4 | **0.4 維持** | memory 警告 (overcalibration risk) |
| **TAX_SMOOTHING** (genus+class smoothing) | なし | **追加 (α=0.15/0.05)** ★ | anthony Cell 24 |

## Stack (exp078 と同)

| # | Model | Weight |
|---|---|---|
| 1 | exp037 v313 (LightProtoSSM + MLP + ResSSM) ★ hyperparameter tuned | 0.35 |
| 2 | Tucker SED 5-fold ONNX | 0.40 |
| 3 | exp029 R3 (eca_nfnet_l1) | 0.25 |

## Expected
- LB **0.950-0.953** (exp078 から +0.000-0.003)
- 改善確率: **70%**
- gold (0.954) 確率: **10-15%**
- drag 確率: **15%**

## 根拠
- anthony Model_22 (lambda 0.4) LB 0.928 → Model_51 (lambda 0.65 + rank 0.6) LB 0.949 = +0.021 standalone
- 我々 exp037 v313 が同 paradigm (mtoshidesu yaroslav v221 系)、同じ tuning で similar lift 期待
- TAX_SMOOTHING は anthony 0.950 NB の必須 component (Cell 24)
- 3-way blend 内 weight 0.35 で dilute → 期待 +0.001-0.003

## Risk
- exp037 内部 PP が default value で local optimum の可能性 (~20%)
- TAX_SMOOTHING の α (0.15/0.05) が我々の predictions に最適でない可能性 (~15%)
- FCS は memory 警告のため変更せず

## Inputs (exp078 と完全同一)
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`
- `maekeso/birdclef2026-exp029-l1-single`
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
'''
hdr_cell["source"] = new_hdr_src.splitlines(keepends=True)

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[OK] Saved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
