"""exp031: 3-way blend = NB4 v7 + Tucker public SED + exp029 (eca_nfnet_l1 R3 student).

exp019 (NB4 0.35 / Tucker 0.40 / **e17** 0.25) = LB 0.947 silver からの **e17 -> exp029 swap**。
- e17 (eca_nfnet_l0 R2 single, LB 0.921) → exp029 (eca_nfnet_l1 R3 distill single, LB 0.923)
- e29 は e17 の上位互換: 単 LB +0.002、gap -0.006 改善、l1 +30% params
- weight ratio は exp019 と同じ 0.35/0.40/0.25 維持 (e29 単体は e17 と類似分布想定)

期待 LB: 0.947-0.949 (中央 0.948、+0.001 寄り)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp031/notebook
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

# ============================================================================
# E29 inference cell — rewrite of E17_INFER (exp018) with exp029 ckpt+backbone
# ============================================================================
EXP018_GEN_PATH = HERE.parent.parent / "exp018" / "notebook" / "_gen_nb_blend.py"
_EXP018_SRC = EXP018_GEN_PATH.read_text(encoding="utf-8")

_E17_START = _EXP018_SRC.index("E17_INFER = r'''")
_E17_BODY_START = _E17_START + len("E17_INFER = r'''")
_E17_END = _EXP018_SRC.index("'''", _E17_BODY_START)
E17_INFER_BASE = _EXP018_SRC[_E17_BODY_START:_E17_END]

# Patch: swap e17 (exp017 l0 R2) → e29 (exp029 l1 R3)
E17_INFER = E17_INFER_BASE
E17_INFER = E17_INFER.replace(
    'Path("/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights"),',
    'Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single"),',
)
E17_INFER = E17_INFER.replace(
    'Path("/kaggle/input/birdclef2026-exp017-weights"),',
    'Path("/kaggle/input/birdclef2026-exp029-l1-single"),',
)
# Ckpt name priority: exp029 R3 fold0 first
E17_INFER = E17_INFER.replace(
    '"r2_ckpt_best_ns22.pth",   # R2 best (val 0.9249) — preferred',
    '"r3_fold0_ckpt_best_ns22.pth",   # exp029 R3 fold0 (val 0.9409, LB 0.923)',
)
E17_INFER = E17_INFER.replace(
    '"r2_ckpt_best_macro.pth",',
    '"r3_fold0_ckpt_best_macro.pth",',
)
E17_INFER = E17_INFER.replace(
    '"r2_ckpt_latest.pth",',
    '"r3_fold0_ckpt_latest.pth",',
)

# Patch the initial dataset existence check to also look for r3_fold0_*
E17_INFER = E17_INFER.replace(
    '''    if _p.exists() and (any(_p.rglob("r2_ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_latest*.pth"))):''',
    '''    if _p.exists() and (any(_p.rglob("r3_fold0_ckpt_best_ns22.pth"))
                        or any(_p.rglob("r2_ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_latest*.pth"))):''',
)

# Patch the fallback rglob search to look for r3_fold0 first
E17_INFER = E17_INFER.replace(
    '''if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r2_ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
assert E17_STATE_DIR is not None, "exp017 ckpt not found"''',
    '''if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r3_fold0_ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r2_ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
assert E17_STATE_DIR is not None, "exp031: exp029 ckpt not found (looked for r3_fold0_ckpt_best_ns22.pth)"''',
)

E17_INFER = E17_INFER.replace(
    'print(f"exp017 state dir: {E17_STATE_DIR}")',
    'print(f"exp031 (exp029 ckpt) state dir: {E17_STATE_DIR}")',
)
# Backbone: l0 → l1
E17_INFER = E17_INFER.replace(
    'E17_BACKBONE = "eca_nfnet_l0"',
    'E17_BACKBONE = "eca_nfnet_l1"  # exp029 student (l0 から +30% params)',
)
# Cell header comment
E17_INFER = E17_INFER.replace(
    '# === exp017 R2 (eca_nfnet_l0 + Perch distill) inference ===',
    '# === exp029 R3 (eca_nfnet_l1 + Perch distill) inference ===',
)

# Assertion: should now refer to exp029
assert "birdclef2026-exp029-l1-single" in E17_INFER, "swap failed"
assert "eca_nfnet_l1" in E17_INFER, "backbone swap failed"
assert "r3_fold0_ckpt_best_ns22.pth" in E17_INFER, "ckpt swap failed"

# ============================================================================
# Blend cell — exp031 ratio = 0.35 / 0.40 / 0.25 (same as exp019, but slot3 = e29)
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 + Tucker + exp029) + Sonotype mirror + submission ===
# exp031 ratio config (w35-40-25, slot3 = exp029 instead of e17):
BLEND_W_E10  = 0.35   # NB4 v7 weight
BLEND_W_SED  = 0.40   # Tucker public SED weight
BLEND_W_E17  = 0.25   # exp029 R3 (eca_nfnet_l1) weight — note var name kept "E17" for code reuse
USE_SED_PRE_AVG = False

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6, \
    "blend weights must sum to 1.0"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])  # ← actually exp029

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
    print(f"  SED-pre-avg mode: NB4={_w_nb4:.2f} / (Tucker+e29)={_w_sed_total:.2f} "
          f"(Tucker:e29 internal = {_w_sed:.2f}:{_w_e17:.2f})")
else:
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    blend_flat = (BLEND_W_E10 * rank_e10
                  + BLEND_W_SED * rank_sed
                  + BLEND_W_E17 * rank_e17)
    print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e29={BLEND_W_E17}")

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
# Patch source — same pattern as exp019
# ============================================================================
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e29-infer", E17_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E29 inference cell (exp031 3-way blend, slot3 = exp029) ===\n'
    'E17_INFER = ' + repr(E17_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod
mod = mod.replace(BLEND_CONST_MARKER, E17_CONST_BLOCK)

OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod)
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp031 = NB4 v7 (0.35) + Tucker SED (0.40) + exp029 R3 eca_nfnet_l1 (0.25), 3-way rank blend\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp019 (NB4+Tucker+e17 R2 LB 0.947) からの **e17 → exp029 swap** で +0.001-0.002 狙い",
)

import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
