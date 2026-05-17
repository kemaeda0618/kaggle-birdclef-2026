"""exp021: 3-way blend = NB4 v7 + Tucker public SED + exp017 R2 (eca_nfnet_l0 SED).

exp019 (w35-40-25) = LB 0.947 からの **Tucker 復元** 実験。
- Tucker weight を 0.40 → 0.45 (= exp018 v1 level)、NB4 を 0.35 → 0.30 で吸収
- e17 0.25 は維持 (exp019 と同)
- 目的: exp019 で Tucker を削った (-0.05) のが cost だったかを isolate 判定
  - +0.001 出れば「Tucker は強軸、復元すべき」→ 次戦略 (l1 込み 4-way) で Tucker 高めに振る
  - ±0 なら「NB4=Tucker 等価」→ 自由度ある
  - -0.001 なら「NB4 が binding lever」→ exp019 ratio が最適、他軸に振る
- 命名: birdclef2026-exp021-blend-w30-45-25
- 期待 LB: 0.947-0.948 (改善確率 25%)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp021/notebook
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

EXP018_GEN_PATH = HERE.parent.parent / "exp018" / "notebook" / "_gen_nb_blend.py"
_EXP018_SRC = EXP018_GEN_PATH.read_text(encoding="utf-8")

_E17_START = _EXP018_SRC.index("E17_INFER = r'''")
_E17_BODY_START = _E17_START + len("E17_INFER = r'''")
_E17_END = _EXP018_SRC.index("'''", _E17_BODY_START)
E17_INFER = _EXP018_SRC[_E17_BODY_START:_E17_END]

NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 + Tucker + exp017) + Sonotype mirror + submission ===
# exp021 ratio (w30-45-25): Tucker 復元、NB4 削減、e17 維持
BLEND_W_E10  = 0.30   # NB4 v7 weight (exp019: 0.35 → -0.05)
BLEND_W_SED  = 0.45   # Tucker public SED weight (exp019: 0.40 → +0.05, exp018 v1 復元)
BLEND_W_E17  = 0.25   # exp017 R2 weight (exp019 維持)
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

CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e17-infer", E17_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in tucker source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E17 inference cell (exp021 3-way blend, ratio w30-45-25) ===\n'
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
    'out_path = HERE / "nb_blend_w30-45-25.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp021 = NB4 v7 (0.30) + Tucker SED (0.45) + exp017 R2 eca_nfnet_l0 (0.25), Tucker 復元 isolate\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp019 (w35-40-25 LB 0.947) からの **Tucker 復元** (0.40→0.45) で Tucker 強軸度を isolate 判定。Tucker 公開 5-fold ONNX",
)

import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
