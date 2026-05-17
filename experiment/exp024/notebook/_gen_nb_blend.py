"""exp024 v1: exp019 base (NB4 0.35 / Tucker 0.40 / e17 0.25) + ProtoSSM v5 architecture upgrade.

NB4 v7 の ProtoSSM 内部 architecture を **公開 NB v5 仕様** に上書き:
- D_MODEL: 128 → 256
- N_SSM_LAYERS: 2 → 3
- DROPOUT: 0.10 → 0.15

ResidualSSM と Tweak E は別 step (exp024 v2, v3) で incremental 投入。
LB 効果が isolate できる。

期待 LB: 0.948-0.950 (公開 NB ProtoSSM v5 単独効果 +0.001-0.003)
リスク: d_model 倍化で labeled SS 66 file 過学習 → -0.001 〜 -0.002 ありうる
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp024/notebook
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

EXP018_GEN_PATH = HERE.parent.parent / "exp018" / "notebook" / "_gen_nb_blend.py"
_EXP018_SRC = EXP018_GEN_PATH.read_text(encoding="utf-8")

_E17_START = _EXP018_SRC.index("E17_INFER = r'''")
_E17_BODY_START = _E17_START + len("E17_INFER = r'''")
_E17_END = _EXP018_SRC.index("'''", _E17_BODY_START)
E17_INFER = _EXP018_SRC[_E17_BODY_START:_E17_END]

# ============================================================================
# BLEND cell: exp019 と同一 ratio (w35-40-25)、ProtoSSM v5 効果のみ isolate
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 v8 [ProtoSSM v5] + Tucker + exp017) + Sonotype mirror + submission ===
# exp024 v1 ratio (w35-40-25, exp019 と同): ProtoSSM v5 architecture 効果 isolate
BLEND_W_E10  = 0.35   # NB4 v8 (ProtoSSM v5)
BLEND_W_SED  = 0.40   # Tucker public SED
BLEND_W_E17  = 0.25   # exp017 R2 (eca_nfnet_l0)
USE_SED_PRE_AVG = False

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6, \
    "blend weights must sum to 1.0"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])

rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
blend_flat = (BLEND_W_E10 * rank_e10
              + BLEND_W_SED * rank_sed
              + BLEND_W_E17 * rank_e17)
print(f"  3-way rank blend: NB4 v8={BLEND_W_E10} / Tucker={BLEND_W_SED} / e17={BLEND_W_E17}")

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
print(f"  Sonotype mirror applied to {mirror_count} columns")

probs_blend = blend_flat.reshape(probs_exp010.shape)
print(f"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")

preds_array = probs_blend.reshape(-1, N_CLASSES)
_n, _c = preds_array.shape
_view = preds_array.reshape(-1, N_WINDOWS, _c)
_sorted = np.sort(_view, axis=1)
_topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
_scale = np.power(_topk_mean, FCS_POWER)
preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)

submission = pd.DataFrame(preds_array, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)
extra = our_ids - expected_ids
if extra:
    submission = submission[submission["row_id"].isin(expected_ids)]
submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

total = time.time() - START
print(f"\nSubmission: {submission.shape}, total {total:.0f}s ({total/60:.1f} min)")
print(f"Mean pred: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"Max pred:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head())
'''

# ============================================================================
# Patch source: 1) override ProtoSSM v5 CONFIG, 2) insert e17 inference cell, 3) replace BLEND
# ============================================================================

# === Step 1: Override ProtoSSM CONFIG (D_MODEL, N_SSM_LAYERS, DROPOUT) ===
# These are defined in _gen_nb4_blend.py CONFIG string, embedded in SRC via from import.
# We patch by string replacement on the embedded CONFIG content.
PROTO_V5_PATCHES = [
    ("D_MODEL = 128", "D_MODEL = 256   # exp024 v1: ProtoSSM v5 upgrade (128→256)"),
    ("N_SSM_LAYERS = 2", "N_SSM_LAYERS = 3   # exp024 v1: ProtoSSM v5 upgrade (2→3)"),
    ("DROPOUT = 0.1\n", "DROPOUT = 0.15  # exp024 v1: ProtoSSM v5 upgrade (0.10→0.15)\n"),
]

# CONFIG patches apply ONLY to _gen_nb4_blend.py (which holds CONFIG string).
# tucker SRC just imports CONFIG via `from _gen_nb4_blend import CONFIG`, no inline values.
NB4_BASE_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb4_blend.py"
NB4_BASE_SRC = NB4_BASE_PATH.read_text(encoding="utf-8")
for _old, _new in PROTO_V5_PATCHES:
    assert _old in NB4_BASE_SRC, f"Patch target not found in _gen_nb4_blend.py: {_old!r}"
    NB4_BASE_SRC = NB4_BASE_SRC.replace(_old, _new)
    print(f"[gen] Patched _gen_nb4_blend.py: {_old.split(chr(10))[0][:50]}")

# Write patched NB4 base to a temp location, switch import path
_PATCHED_NB4_PATH = HERE / "_gen_nb4_blend_v8.py"
_PATCHED_NB4_PATH.write_text(NB4_BASE_SRC, encoding="utf-8")
print(f"[gen] Wrote patched NB4 base: {_PATCHED_NB4_PATH}")

# Modify SRC to import from our patched module instead of _gen_nb4_blend
SRC = SRC.replace(
    "from _gen_nb4_blend import (",
    "from _gen_nb4_blend_v8 import (",
)

# === Step 2: Insert e17 inference cell ===
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e17-infer", E17_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in tucker source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E17 inference cell (exp024 v1: ProtoSSM v5) ===\n'
    'E17_INFER = ' + repr(E17_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, E17_CONST_BLOCK)

# === Step 3: Replace BLEND with our 3-way (same ratio as exp019) ===
OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_protossm_v5.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp024 v1 = NB4 v8 (ProtoSSM v5: d=256/L=3/dropout=0.15) + Tucker + e17 R2 (3-way rank blend, exp019 ratio)\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp019 (w35-40-25 LB 0.947 silver peak) の NB4 内部 ProtoSSM を v5 (d_model 128→256, layers 2→3, dropout 0.10→0.15) に upgrade、ratio 同一で ProtoSSM v5 効果 isolate。Tucker 公開 5-fold ONNX",
)

import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
_EXP024_NB_DIR = str(HERE)
if _EXP024_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP024_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
