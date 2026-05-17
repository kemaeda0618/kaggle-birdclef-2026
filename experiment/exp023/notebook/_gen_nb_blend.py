"""exp023: exp019 base (NB4 0.35 / Tucker 0.40 / e17 0.25) + species-level wet/dry prior.

exp019 (w35-40-25) = LB 0.947 silver peak からの **post-FCS multiplicative prior** 追加実験。
- labeled SS から 234×2 (species × season) empirical 共起表を inline 構築 (NPZ 不要)
- Empirical Bayes shrinkage (ALPHA=50)、boost clamp [0.7, 1.5]、n<5 種は中立
- post-FCS の最終 preds に season-aware multiplicative apply
- ratio (w35-40-25) は exp019 と同一 → wet/dry prior 効果を isolate

命名: birdclef2026-exp023-blend-w35-40-25-wd
期待 LB: 0.949-0.952 (silver 完全確保、+0.002-0.005)
リスク: labeled SS と test の seasonal mix がズレた場合 -0.001 〜 -0.003
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp023/notebook
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

EXP018_GEN_PATH = HERE.parent.parent / "exp018" / "notebook" / "_gen_nb_blend.py"
_EXP018_SRC = EXP018_GEN_PATH.read_text(encoding="utf-8")

_E17_START = _EXP018_SRC.index("E17_INFER = r'''")
_E17_BODY_START = _E17_START + len("E17_INFER = r'''")
_E17_END = _EXP018_SRC.index("'''", _E17_BODY_START)
E17_INFER = _EXP018_SRC[_E17_BODY_START:_E17_END]

# ============================================================================
# BLEND cell: exp019 ratio + species-level wet/dry prior (post-FCS)
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend + Sonotype mirror + FCS + wet/dry species prior ===
# exp023 ratio (w35-40-25, exp019 と同): wet/dry prior 効果 isolate
BLEND_W_E10  = 0.35   # NB4 v7
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
print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e17={BLEND_W_E17}")

# === Sonotype mirror ===
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

# === FCS (file_confidence_scale) ===
preds_array = probs_blend.reshape(-1, N_CLASSES)
_n, _c = preds_array.shape
_view = preds_array.reshape(-1, N_WINDOWS, _c)
_sorted = np.sort(_view, axis=1)
_topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
_scale = np.power(_topk_mean, FCS_POWER)
preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)

# === NEW: species-level wet/dry empirical prior ===
# Build prior table inline from train_soundscapes_labels.csv (no extra dataset needed)
WD_ALPHA       = 50.0
WD_CLAMP_MIN   = 0.7
WD_CLAMP_MAX   = 1.5
WD_MIN_SAMPLES = 5

_labels_path = BASE / "train_soundscapes_labels.csv"
print(f"\n[wd-prior] Loading {_labels_path}")
_labels_df = pd.read_csv(_labels_path)
print(f"[wd-prior] {len(_labels_df)} labeled segments")

# Parse season from filename
_FN_RE = re.compile(r"BC2026_(?:Train|Test)_\d+_S(\d+)_(\d{8})_(\d{6})")

def _parse_season(fn):
    _m = _FN_RE.match(str(fn))
    if not _m: return None
    _mm = int(_m.group(2)[4:6])
    return "wet" if _mm in (11, 12, 1, 2, 3, 4) else "dry"

_labels_df["_season"] = _labels_df["filename"].apply(_parse_season)
_n_wet_segs = int((_labels_df["_season"] == "wet").sum())
_n_dry_segs = int((_labels_df["_season"] == "dry").sum())
print(f"[wd-prior] wet={_n_wet_segs} segs, dry={_n_dry_segs} segs")

# Build per-species co-occurrence
_co_wet = np.zeros(N_CLASSES, dtype=np.float64)
_co_dry = np.zeros(N_CLASSES, dtype=np.float64)
_label_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
for _, _row in _labels_df.iterrows():
    _seas = _row["_season"]
    if _seas not in ("wet", "dry") or pd.isna(_row["primary_label"]):
        continue
    for _lab in str(_row["primary_label"]).split(";"):
        _lab = _lab.strip()
        if not _lab: continue
        _idx = _label_to_idx.get(_lab)
        if _idx is None: continue
        if _seas == "wet": _co_wet[_idx] += 1
        else:              _co_dry[_idx] += 1

# Empirical Bayes shrinkage
_n_total = _n_wet_segs + _n_dry_segs
_uniform = 1.0 / N_CLASSES
_p_c_global = (_co_wet + _co_dry + WD_ALPHA * _uniform) / (_n_total + WD_ALPHA)
_p_c_wet    = (_co_wet + WD_ALPHA * _p_c_global) / (_n_wet_segs + WD_ALPHA)
_p_c_dry    = (_co_dry + WD_ALPHA * _p_c_global) / (_n_dry_segs + WD_ALPHA)
_boost_wet  = np.clip(_p_c_wet / _p_c_global, WD_CLAMP_MIN, WD_CLAMP_MAX).astype(np.float32)
_boost_dry  = np.clip(_p_c_dry / _p_c_global, WD_CLAMP_MIN, WD_CLAMP_MAX).astype(np.float32)

# Mask low-sample species (n<5) → neutral
_n_obs = _co_wet + _co_dry
_mask_low = _n_obs < WD_MIN_SAMPLES
_boost_wet[_mask_low] = 1.0
_boost_dry[_mask_low] = 1.0

_n_active = int((~_mask_low).sum())
print(f"[wd-prior] active species (n>={int(WD_MIN_SAMPLES)}): {_n_active} / {N_CLASSES}")
print(f"[wd-prior] boost_wet: min={_boost_wet.min():.3f}, max={_boost_wet.max():.3f}, mean={_boost_wet.mean():.3f}")
print(f"[wd-prior] boost_dry: min={_boost_dry.min():.3f}, max={_boost_dry.max():.3f}, mean={_boost_dry.mean():.3f}")

# Parse season per row_id (test filename)
def _row_season(rid):
    # row_id format: BC2026_Test_NNNN_SXX_YYYYMMDD_HHMMSS_OFFSET
    _parts = str(rid).split("_")
    if len(_parts) < 6: return "wet"  # fallback (most common)
    try:
        _mm = int(_parts[4][4:6])
        return "wet" if _mm in (11, 12, 1, 2, 3, 4) else "dry"
    except Exception:
        return "wet"

_row_seas = np.array([_row_season(_rid) for _rid in all_row_ids])
_n_wet_rows = int((_row_seas == "wet").sum())
_n_dry_rows = int((_row_seas == "dry").sum())
print(f"[wd-prior] test rows: wet={_n_wet_rows}, dry={_n_dry_rows}")

# Apply multiplicative prior
_wet_mask = (_row_seas == "wet")
_dry_mask = (_row_seas == "dry")
if _wet_mask.any():
    preds_array[_wet_mask] = preds_array[_wet_mask] * _boost_wet[None, :]
if _dry_mask.any():
    preds_array[_dry_mask] = preds_array[_dry_mask] * _boost_dry[None, :]
print(f"[wd-prior] applied. preds mean before/after change: ...")

# === Build submission ===
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
# Patch source: insert e17-infer cell + replace BLEND constant body
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
    '# === E17 inference cell (exp023 3-way + wet/dry prior) ===\n'
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
    'out_path = HERE / "nb_blend_w35-40-25_wd.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp023 = exp019 base (w35-40-25) + species-level wet/dry empirical prior (post-FCS)\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp019 (w35-40-25 LB 0.947 silver peak) に **species-level wet/dry empirical prior** を post-FCS で重ねる、ratio 同一で prior 効果 isolate。Tucker 公開 5-fold ONNX",
)

import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
