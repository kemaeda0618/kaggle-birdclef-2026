"""exp019: 3-way blend = NB4 v7 + Tucker public SED + exp017 R2 (eca_nfnet_l0 SED).

exp018 v1 (NB4 0.40 / Tucker 0.45 / e17 0.15) = LB 0.946 (silver タッチ) からの e17 寄せ。
- e17 weight を 0.15 → 0.25 に増、NB4/Tucker を対称に -0.05 ずつ削減
- 命名規約: birdclef2026-exp{NNN}-blend-w{NB4*100}-{Tucker*100}-{e17*100}
  - 今回: w35-40-25 (NB4 0.35 / Tucker 0.40 / e17 0.25)
- 過去 e14 R2 weight 0.10→0.15 で blend +0.001 出た実績を参考に、
  e17 の質的優位 (gap -0.004) を考慮して step を +0.10 に倍化
- 期待 LB: 0.947-0.948 (silver 確定)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp019/notebook
# Tucker base generator lives under exp010
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

# ============================================================================
# Reuse E17_INFER from exp018 (same model, same ckpt source, same architecture)
# ============================================================================
EXP018_GEN_PATH = HERE.parent.parent / "exp018" / "notebook" / "_gen_nb_blend.py"
_EXP018_SRC = EXP018_GEN_PATH.read_text(encoding="utf-8")

# Extract E17_INFER constant from exp018 generator
_E17_START = _EXP018_SRC.index("E17_INFER = r'''")
_E17_END_MARKER = "'''"
_E17_BODY_START = _E17_START + len("E17_INFER = r'''")
_E17_END = _EXP018_SRC.index(_E17_END_MARKER, _E17_BODY_START)
E17_INFER = _EXP018_SRC[_E17_BODY_START:_E17_END]

# ============================================================================
# New BLEND cell — exp019 v1 ratio = 0.35 / 0.40 / 0.25
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 + Tucker + exp017) + Sonotype mirror + submission ===
# exp019 ratio config (w35-40-25):
BLEND_W_E10  = 0.35   # NB4 v7 weight (v1: 0.40 → -0.05)
BLEND_W_SED  = 0.40   # Tucker public SED weight (v1: 0.45 → -0.05)
BLEND_W_E17  = 0.25   # exp017 R2 (eca_nfnet_l0) weight (v1: 0.15 → +0.10)
USE_SED_PRE_AVG = False

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6, \
    "blend weights must sum to 1.0"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])

if USE_SED_PRE_AVG:
    _eps = 1e-7
    _l_sed = np.log(np.clip(flat_sed, _eps, 1 - _eps)) - np.log1p(-np.clip(flat_sed, _eps, 1 - _eps))
    _l_e17 = np.log(np.clip(flat_e17, _eps, 1 - _eps)) - np.log1p(-np.clip(flat_e17, _eps, 1 - _eps))
    _w_sed = BLEND_W_SED / (BLEND_W_SED + BLEND_W_E17)
    _w_e17 = BLEND_W_E17 / (BLEND_W_SED + BLEND_W_E17)
    _l_sed_avg = _w_sed * _l_sed + _w_e17 * _l_e17
    flat_sed_combined = (1.0 / (1.0 + np.exp(-np.clip(_l_sed_avg, -50, 50)))).astype(np.float32)
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed_combined).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    _w_nb4 = BLEND_W_E10
    _w_sed_total = BLEND_W_SED + BLEND_W_E17
    blend_flat = _w_nb4 * rank_e10 + _w_sed_total * rank_sed
    print(f"  SED-pre-avg mode: NB4={_w_nb4:.2f} / (Tucker+e17)={_w_sed_total:.2f} "
          f"(Tucker:e17 internal = {_w_sed:.2f}:{_w_e17:.2f})")
else:
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    blend_flat = (BLEND_W_E10 * rank_e10
                  + BLEND_W_SED * rank_sed
                  + BLEND_W_E17 * rank_e17)
    print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e17={BLEND_W_E17}")

# === Sonotype mirror (公開 0.946 NB Cell 40) ===
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
# Patch source: insert e17-infer cell append + replace BLEND constant.
# ============================================================================
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e17-infer", E17_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in tucker source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E17 inference cell (exp019 3-way blend, ratio w35-40-25) ===\n'
    'E17_INFER = ' + repr(E17_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, E17_CONST_BLOCK)

OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_w35-40-25.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp019 = NB4 v7 (0.35) + Tucker SED (0.40) + exp017 R2 eca_nfnet_l0 (0.25), 3-way rank blend\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp018 v1 (NB4+Tucker+e17 R2 LB 0.946) からの **e17 寄せ (0.15→0.25)** で silver 確定狙い。Tucker 公開 5-fold ONNX",
)

# exec under exp019/notebook so HERE / out_path resolve here.
import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
