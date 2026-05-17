"""v11: v8 + G26 (Per-class Isotonic regression + F1-optimal threshold sharpening).

公開 0.946 NB (yaroslavkholmirzayev) などで使われる post-hoc calibration。
本実装は MVP: NB4 が学習に使った labeled SS を OOF proxy として用いる
(in-sample bias あり; 公開 NB は proper OOF 基盤を持つ)。

3 新セル + BLEND 修正:
  Cell A (g26_nb4_oof): NB4 v7 を labeled SS embedding に適用 → probs_nb4_oof
  Cell B (g26_tucker_oof): Tucker SED を labeled SS audio に適用 → probs_tucker_oof
  Cell C (g26_calibrate): Cell A/B を BLEND と同じロジックで結合 → per-class
                          IsotonicRegression + F1-optimal threshold をフィット
  BLEND (修正): Sonotype mirror の直後、file_confidence_scale の前で
                calibrators[c].transform → threshold-based sharpening を適用

期待 LB: +0.004-0.008 (in-sample bias で減衰、最悪悪化もあり得る)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# ----------------------------------------------------------------------------
# Cell A: NB4 v7 predictions on labeled SS (OOF proxy, IN-SAMPLE!)
# Reuses _ensemble_one / compute_retrieval_logit / compute_retrieval_ta_logit
# / compute_prior_logit / proto_models / mlp_models / E19 / TA retrieval logic.
# ----------------------------------------------------------------------------
G26_NB4_OOF = r'''# === G26 Cell A: NB4 v7 OOF predictions on labeled SS (IN-SAMPLE bias!) ===
# WARNING: NB4 was trained on these labeled SS files. Predictions here are
# in-sample and biased optimistically. Multi-seed=5 averaging reduces but
# does NOT eliminate this bias. The Isotonic calibrator fit on these will
# under-correct true OOD test predictions. Public 0.946 NB uses proper OOF.
print(f"[G26 Cell A] NB4 v7 on {lab_emb_files.shape[0]} labeled SS files (IN-SAMPLE)...")
import gc as _gc_g26
_t0_g26a = time.time()
probs_nb4_oof = []

# Pre-compute prior logits per file (already in lab_prior_files)
for _fi in range(lab_emb_files.shape[0]):
    _emb_np    = lab_emb_files[_fi]                          # (12, 1536)
    _scores_np = lab_scores_files[_fi]                       # (12, N_CLASSES)
    _site_id   = int(lab_site_ids[_fi])
    _hour      = int(lab_hours[_fi])
    _prior_vec = lab_prior_files[_fi]                        # (N_CLASSES,)

    _site_t  = torch.tensor([_site_id], dtype=torch.long)
    _hour_t  = torch.tensor([_hour], dtype=torch.long)
    _prior_t = torch.tensor(_prior_vec, dtype=torch.float32).unsqueeze(0)
    _emb_t   = torch.tensor(_emb_np, dtype=torch.float32).unsqueeze(0)
    _scores_t = torch.tensor(_scores_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        _p_proto = _ensemble_one(proto_models, _emb_t, _scores_t,
                                  _site_t, _hour_t, _prior_t)
        _p_mlp   = _ensemble_one(mlp_models,   _emb_t, _scores_t,
                                  _site_t, _hour_t, _prior_t)
        _p_proto_c = _p_proto.clamp(min=1e-7, max=1 - 1e-7)
        _p_mlp_c   = _p_mlp.clamp(min=1e-7, max=1 - 1e-7)
        _l_proto = torch.log(_p_proto_c) - torch.log1p(-_p_proto_c)
        _l_mlp   = torch.log(_p_mlp_c)   - torch.log1p(-_p_mlp_c)
        _l_blend = W_PROTO * _l_proto + W_MLP * _l_mlp

        if LAMBDA_RETRIEVAL > 0:
            _ret_logit = compute_retrieval_logit(_emb_np, _site_id, _hour)
            _ret_logit_t = torch.tensor(_ret_logit, dtype=torch.float32).unsqueeze(0)
            _l_blend = _l_blend + LAMBDA_RETRIEVAL * _ret_logit_t

        if USE_TA_RETRIEVAL and ta_pool_emb_norm is not None:
            _ta_ret_logit = compute_retrieval_ta_logit(_emb_np)
            _ta_ret_logit_t = torch.tensor(_ta_ret_logit, dtype=torch.float32).unsqueeze(0)
            _lam_t = torch.tensor(LAMBDA_TA_VEC, dtype=torch.float32)
            _l_blend = _l_blend + _lam_t * _ta_ret_logit_t

        if USE_E19 and E19_BETA > 0:
            if E19_AGG == "max":
                _file_sig = _l_blend.max(dim=1, keepdim=True).values
            elif E19_AGG == "mean":
                _file_sig = _l_blend.mean(dim=1, keepdim=True)
            elif E19_AGG == "median":
                _file_sig = _l_blend.median(dim=1, keepdim=True).values
            else:
                _file_sig = None
            if _file_sig is not None:
                _l_blend = (1.0 - E19_BETA) * _l_blend + E19_BETA * _file_sig

        _agg = torch.sigmoid(_l_blend)
        _probs = _agg.squeeze(0).numpy()
    probs_nb4_oof.append(_probs)

    if (_fi + 1) % 10 == 0 or _fi == lab_emb_files.shape[0] - 1:
        print(f"  NB4 OOF [{_fi+1}/{lab_emb_files.shape[0]}] {time.time()-_t0_g26a:.0f}s")

probs_nb4_oof = np.stack(probs_nb4_oof).astype(np.float32)   # (N_lab, 12, N_CLASSES)
print(f"NB4 OOF done: {probs_nb4_oof.shape} in {time.time()-_t0_g26a:.0f}s")
_gc_g26.collect()
'''

# ----------------------------------------------------------------------------
# Cell B: Tucker SED predictions on labeled SS audio (OOF proxy)
# Tucker was trained on its own training data (not necessarily our labeled SS),
# so this is mostly proper-OOF for Tucker. But the calibrator fit will use the
# NB4 OOF (in-sample biased) so the combined target is still biased.
# ----------------------------------------------------------------------------
G26_TUCKER_OOF = r'''# === G26 Cell B: Tucker SED on labeled SS audio (proper OOF for SED branch) ===
# Tucker SED was not trained on our labeled SS, so this is honest OOF for SED.
# Reuses sed_sessions / audio_to_mel / sigmoid_sed / smoothing from SED_INFER cell.
print(f"[G26 Cell B] Tucker SED on {len(lab_file_list)} labeled SS audio files...")
_t0_g26b = time.time()
probs_tucker_oof = []

for _fi, _fn in enumerate(lab_file_list):
    _fpath = TRAIN_SC_DIR / _fn
    if not _fpath.exists():
        print(f"  WARN: {_fpath} not found, filling 0.5")
        probs_tucker_oof.append(np.full((N_WINDOWS, N_CLASSES), 0.5, dtype=np.float32))
        continue
    _raw_60s = read_audio(str(_fpath), target_samples=SR * 60)
    _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
    _mel = audio_to_mel(_chunks)
    _p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    for _sess in sed_sessions:
        _outs = _sess.run(None, {_sess.get_inputs()[0].name: _mel})
        _clip_logits = _outs[0]
        _frame_max = _outs[1].max(axis=1)
        _p_sum += 0.5 * sigmoid_sed(_clip_logits) + 0.5 * sigmoid_sed(_frame_max)
    _p_mean = _p_sum / len(sed_sessions)
    _p_mean = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_tucker_oof.append(_p_mean)
    if (_fi + 1) % 10 == 0 or _fi == len(lab_file_list) - 1:
        print(f"  Tucker OOF [{_fi+1}/{len(lab_file_list)}] {time.time()-_t0_g26b:.0f}s")

probs_tucker_oof = np.stack(probs_tucker_oof).astype(np.float32)  # (N_lab, 12, N_CLASSES)
print(f"Tucker OOF done: {probs_tucker_oof.shape} in {time.time()-_t0_g26b:.0f}s")
'''

# ----------------------------------------------------------------------------
# Cell C: Combine OOF (same logic as BLEND) and fit per-class Isotonic + F1 threshold
# ----------------------------------------------------------------------------
G26_CALIBRATE = r'''# === G26 Cell C: Fit per-class Isotonic + F1-optimal threshold ===
# Replicate BLEND logic (50:50 rank blend + Sonotype mirror) on OOF preds,
# then fit IsotonicRegression per class against labeled targets.
print("[G26 Cell C] Fitting per-class Isotonic + F1 threshold on OOF blend...")
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

# Flatten (N_lab, 12, N_CLASSES) -> (N_lab * 12, N_CLASSES)
_flat_nb4    = probs_nb4_oof.reshape(-1, N_CLASSES)
_flat_tucker = probs_tucker_oof.reshape(-1, N_CLASSES)
_flat_Y      = lab_labels_files.reshape(-1, N_CLASSES)   # (N_lab * 12, N_CLASSES)

# Same rank-space blend as BLEND cell
_rank_nb4    = pd.DataFrame(_flat_nb4).rank(axis=0, pct=True).to_numpy().astype(np.float32)
_rank_tucker = pd.DataFrame(_flat_tucker).rank(axis=0, pct=True).to_numpy().astype(np.float32)
_BW_NB4_OOF, _BW_SED_OOF = 0.5, 0.5
blend_oof = _BW_NB4_OOF * _rank_nb4 + _BW_SED_OOF * _rank_tucker

# Same Sonotype mirror as BLEND
_MIRROR_PAIRS_OOF = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)
_col_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
for _group in _MIRROR_PAIRS_OOF:
    _valid_idx = [_col_to_idx[s] for s in _group if s in _col_to_idx]
    if len(_valid_idx) >= 2:
        _gmax = blend_oof[:, _valid_idx].max(axis=1, keepdims=True)
        blend_oof[:, _valid_idx] = _gmax

print(f"blend_oof shape: {blend_oof.shape}, mean={blend_oof.mean():.4f}")

# Per-class IsotonicRegression + F1 threshold grid
calibrators = []
thresholds = []
classes_with_calibration = []
THR_GRID = np.linspace(0.20, 0.80, 25)

for _c in range(N_CLASSES):
    _y_c = _flat_Y[:, _c]
    if _y_c.sum() < 3:   # too few positives, skip
        calibrators.append(None)
        thresholds.append(0.5)
        continue
    _x_c = blend_oof[:, _c]
    _ir = IsotonicRegression(out_of_bounds="clip")
    _ir.fit(_x_c, _y_c)
    calibrators.append(_ir)
    _cal_pred = _ir.predict(_x_c)
    _best_f1, _best_thr = -1.0, 0.5
    for _thr in THR_GRID:
        _pred_bin = (_cal_pred >= _thr).astype(np.int32)
        _f1 = f1_score(_y_c, _pred_bin, zero_division=0)
        if _f1 > _best_f1:
            _best_f1 = _f1
            _best_thr = _thr
    thresholds.append(_best_thr)
    classes_with_calibration.append(_c)

thresholds = np.array(thresholds, dtype=np.float32)
print(f"Calibrated {len(classes_with_calibration)}/{N_CLASSES} classes "
      f"(rest had <3 positive labels)")
print(f"Threshold distribution: min={thresholds[classes_with_calibration].min():.3f} "
      f"max={thresholds[classes_with_calibration].max():.3f} "
      f"mean={thresholds[classes_with_calibration].mean():.3f}")
'''

# ----------------------------------------------------------------------------
# Patch 1: Insert 3 new cells BEFORE the BLEND cell append.
# ----------------------------------------------------------------------------
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("g26_nb4_oof", G26_NB4_OOF))\n'
    'cells.append(code_cell("g26_tucker_oof", G26_TUCKER_OOF))\n'
    'cells.append(code_cell("g26_calibrate", G26_CALIBRATE))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in v8 source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

# ----------------------------------------------------------------------------
# Patch 2: Inject the G26 cell-source constants right before BLEND constant definition.
# ----------------------------------------------------------------------------
BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
G26_CONST_BLOCK = (
    '# G26 OOF + calibration cells (inserted between SED_INFER and BLEND)\n'
    'G26_NB4_OOF = ' + repr(G26_NB4_OOF) + '\n\n'
    'G26_TUCKER_OOF = ' + repr(G26_TUCKER_OOF) + '\n\n'
    'G26_CALIBRATE = ' + repr(G26_CALIBRATE) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, G26_CONST_BLOCK)

# ----------------------------------------------------------------------------
# Patch 3: Modify BLEND cell to apply calibrators after Sonotype mirror,
# BEFORE file_confidence_scale.
# ----------------------------------------------------------------------------
OLD_BLEND_TAIL_MARKER = """probs_blend = blend_flat.reshape(probs_exp010.shape)
print(f"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
print(f"  ratios (rank-space): NB4={BLEND_W_EXP010} / Tucker={BLEND_W_SED}")"""

NEW_BLEND_TAIL = """# === G26: per-class Isotonic + F1-threshold sharpening ===
# Apply post-hoc calibration learned from labeled SS OOF preds.
# WARNING: NB4 OOF was IN-SAMPLE so calibration is biased optimistically.
print("Applying G26 per-class Isotonic + F1 threshold sharpening...")
blend_calibrated = blend_flat.copy()
_sharp_above = 1.3   # above threshold: stretch upward (sharpen positive)
_sharp_below = 0.7   # below threshold: shrink toward zero (sharpen negative)
for _c in classes_with_calibration:
    _cal = calibrators[_c].transform(blend_flat[:, _c])
    _thr = thresholds[_c]
    _sharpened = np.where(
        _cal >= _thr,
        _thr + (_cal - _thr) * _sharp_above,
        _cal * _sharp_below,
    )
    blend_calibrated[:, _c] = np.clip(_sharpened, 0.0, 1.0)
blend_flat = blend_calibrated
print(f"  G26 applied to {len(classes_with_calibration)} classes "
      f"(sharp_above={_sharp_above}, sharp_below={_sharp_below})")

probs_blend = blend_flat.reshape(probs_exp010.shape)
print(f"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
print(f"  ratios (rank-space): NB4={BLEND_W_EXP010} / Tucker={BLEND_W_SED}")"""

assert OLD_BLEND_TAIL_MARKER in mod, "Could not find BLEND tail in v8 source"
mod = mod.replace(OLD_BLEND_TAIL_MARKER, NEW_BLEND_TAIL)

# ----------------------------------------------------------------------------
# Patch 4: Header comment + output filename + markdown
# ----------------------------------------------------------------------------
mod = mod.replace(
    "# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===",
    "# === Stage C: rank blend 50:50 + Sonotype mirror + G26 calibration (v11) + submission ==="
)
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_tucker_v11_g26.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# v11 = v8 + G26 (Per-class Isotonic + F1 threshold sharpening, MVP in-sample)\\n",
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_tucker_v11_g26_inner.py")})
