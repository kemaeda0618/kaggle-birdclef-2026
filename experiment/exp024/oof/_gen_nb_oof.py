"""OOF validation NB: run exp019 blend pipeline on labeled_ss 66 files.

入出力:
- 入力: train_soundscapes/{66 labeled files}.ogg (test_soundscapes でなく)
- 出力: per-model probs (NB4 / Tucker / e17) + per-class AUC + labels
  - oof_probs_e10.npy   (n_rows, 234) float32
  - oof_probs_sed.npy   (n_rows, 234) float32
  - oof_probs_e17.npy   (n_rows, 234) float32
  - oof_labels.npy      (n_rows, 234) int8 multi-hot
  - oof_row_meta.npy    (n_rows,) row_id string array
  - oof_auc_summary.json per-model and blend AUC

設計:
- exp019 generator base、test_files をだ labeled_ss 66 file に強制切替
- 推論経路は exp019 と完全同一 (NB4 inline + Tucker + e17 同じ)
- BLEND/submission cells は AUC 計算 + npz save に書き換え

slug: birdclef2026-exp024-oof-exp019-base
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp024/oof
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

EXP018_GEN_PATH = HERE.parent.parent / "exp018" / "notebook" / "_gen_nb_blend.py"
_EXP018_SRC = EXP018_GEN_PATH.read_text(encoding="utf-8")

_E17_START = _EXP018_SRC.index("E17_INFER = r'''")
_E17_BODY_START = _E17_START + len("E17_INFER = r'''")
_E17_END = _EXP018_SRC.index("'''", _E17_BODY_START)
E17_INFER = _EXP018_SRC[_E17_BODY_START:_E17_END]

# ============================================================================
# Replace BLEND cell with OOF computation + save
# ============================================================================
OOF_BLEND = r'''# === OOF Stage C: compute per-class AUC + save predictions for offline analysis ===
import json
from sklearn.metrics import roc_auc_score

print(f"\n[OOF] Computing per-model probabilities and per-class AUCs on labeled SS ({len(audio_cache)} files)")

# Flatten per-model probs to (n_rows, n_classes)
oof_probs_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1]).astype(np.float32)
oof_probs_sed = probs_sed.reshape(-1, probs_sed.shape[-1]).astype(np.float32)
oof_probs_e17 = probs_e17.reshape(-1, probs_e17.shape[-1]).astype(np.float32)
oof_row_ids = np.array(all_row_ids, dtype=object)

print(f"[OOF] probs shapes: e10={oof_probs_e10.shape}, sed={oof_probs_sed.shape}, e17={oof_probs_e17.shape}")
print(f"[OOF] row_ids: {len(oof_row_ids)}")

# === Build labels matrix from train_soundscapes_labels.csv ===
_labels_csv = BASE / "train_soundscapes_labels.csv"
print(f"\n[OOF] Loading labels: {_labels_csv}")
_labels_df = pd.read_csv(_labels_csv)
print(f"[OOF] {len(_labels_df)} labeled segments across {_labels_df['filename'].nunique()} files")

# Parse end_sec from start/end fields ("00:00:00", "00:00:05")
def _parse_end_sec(end_str):
    h, m, s = end_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

_labels_df["end_sec"] = _labels_df["end"].apply(_parse_end_sec)
_labels_df["stem"] = _labels_df["filename"].str.replace(".ogg", "", regex=False)
_labels_df["row_id"] = _labels_df["stem"] + "_" + _labels_df["end_sec"].astype(str)

# Build species → idx map
_label_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}

# Build (row_id → multi-hot label) dict
_rid_to_labels = {}
for _, _row in _labels_df.iterrows():
    _rid = _row["row_id"]
    if pd.isna(_row["primary_label"]):
        _labels_set = set()
    else:
        _labels_set = set(str(_row["primary_label"]).split(";"))
    _rid_to_labels[_rid] = _labels_set

# Build oof_labels matrix aligned with oof_row_ids
oof_labels = np.zeros((len(oof_row_ids), N_CLASSES), dtype=np.int8)
n_matched = 0
n_unmatched = 0
for _ri, _rid in enumerate(oof_row_ids):
    _lset = _rid_to_labels.get(str(_rid))
    if _lset is None:
        n_unmatched += 1
        continue
    n_matched += 1
    for _lab in _lset:
        _idx = _label_to_idx.get(_lab.strip())
        if _idx is not None:
            oof_labels[_ri, _idx] = 1

print(f"[OOF] rows matched: {n_matched}, unmatched: {n_unmatched} / {len(oof_row_ids)}")
print(f"[OOF] positive labels: {oof_labels.sum()} (mean {oof_labels.mean():.4f})")

# === Compute per-class AUC for each model + key blends ===
def _macro_auc(probs, labels):
    # ROC AUC macro-averaged across classes with at least 1 positive
    aucs = []
    valid_classes = []
    for ci in range(labels.shape[1]):
        if labels[:, ci].sum() == 0 or labels[:, ci].sum() == labels.shape[0]:
            continue
        try:
            auc = roc_auc_score(labels[:, ci], probs[:, ci])
            aucs.append(auc)
            valid_classes.append(ci)
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)

# Per-model
auc_e10, n_e10 = _macro_auc(oof_probs_e10, oof_labels)
auc_sed, n_sed = _macro_auc(oof_probs_sed, oof_labels)
auc_e17, n_e17 = _macro_auc(oof_probs_e17, oof_labels)
print(f"\n[OOF] Per-model macro AUC (n_valid_classes):")
print(f"  NB4 (e10):      {auc_e10:.5f} (n={n_e10})")
print(f"  Tucker (sed):   {auc_sed:.5f} (n={n_sed})")
print(f"  e17 (NFNet):    {auc_e17:.5f} (n={n_e17})")

# Rank-blend (exp019 ratio)
_rank_e10 = pd.DataFrame(oof_probs_e10).rank(axis=0, pct=True).to_numpy(np.float32)
_rank_sed = pd.DataFrame(oof_probs_sed).rank(axis=0, pct=True).to_numpy(np.float32)
_rank_e17 = pd.DataFrame(oof_probs_e17).rank(axis=0, pct=True).to_numpy(np.float32)

BLENDS = {
    "exp018_v1_w40-45-15": (0.40, 0.45, 0.15),
    "exp019_w35-40-25":    (0.35, 0.40, 0.25),
    "exp021_w30-45-25":    (0.30, 0.45, 0.25),
    "equal_w33-33-33":     (1/3, 1/3, 1/3),
    "nb4_only_w100-0-0":   (1.0, 0.0, 0.0),
    "tucker_only_w0-100-0":(0.0, 1.0, 0.0),
    "e17_only_w0-0-100":   (0.0, 0.0, 1.0),
    "nb4_tucker_w50-50":   (0.50, 0.50, 0.00),
    "nb4_e17_w50-50":      (0.50, 0.00, 0.50),
    "tucker_e17_w50-50":   (0.00, 0.50, 0.50),
}

blend_results = {}
print(f"\n[OOF] Blend AUC (rank-space):")
for _name, (_w10, _wsed, _w17) in BLENDS.items():
    _blend = _w10 * _rank_e10 + _wsed * _rank_sed + _w17 * _rank_e17
    _auc, _n = _macro_auc(_blend, oof_labels)
    blend_results[_name] = {"auc": _auc, "n_classes": _n, "weights": [_w10, _wsed, _w17]}
    print(f"  {_name:30s}  {_auc:.5f}  (n={_n})")

# Per-class AUC (granular for analysis)
print(f"\n[OOF] Computing per-class AUC...")
per_class_auc = {}
for _name, _data in [
    ("e10", oof_probs_e10), ("sed", oof_probs_sed), ("e17", oof_probs_e17)
]:
    aucs_c = np.zeros(N_CLASSES, dtype=np.float32)
    has_pos = np.zeros(N_CLASSES, dtype=np.int32)
    for ci in range(N_CLASSES):
        if oof_labels[:, ci].sum() == 0:
            continue
        has_pos[ci] = int(oof_labels[:, ci].sum())
        try:
            aucs_c[ci] = roc_auc_score(oof_labels[:, ci], _data[:, ci])
        except Exception:
            pass
    per_class_auc[_name] = {"auc": aucs_c, "n_pos": has_pos}

# === Save artifacts ===
out_dir = Path(".")
np.save(out_dir / "oof_probs_e10.npy", oof_probs_e10)
np.save(out_dir / "oof_probs_sed.npy", oof_probs_sed)
np.save(out_dir / "oof_probs_e17.npy", oof_probs_e17)
np.save(out_dir / "oof_labels.npy", oof_labels)
np.save(out_dir / "oof_row_ids.npy", oof_row_ids)
np.save(out_dir / "oof_per_class_auc_e10.npy", per_class_auc["e10"]["auc"])
np.save(out_dir / "oof_per_class_auc_sed.npy", per_class_auc["sed"]["auc"])
np.save(out_dir / "oof_per_class_auc_e17.npy", per_class_auc["e17"]["auc"])
np.save(out_dir / "oof_per_class_n_pos.npy", per_class_auc["e10"]["n_pos"])

summary = {
    "n_files": len(audio_cache),
    "n_rows": int(len(oof_row_ids)),
    "n_matched": int(n_matched),
    "n_unmatched": int(n_unmatched),
    "n_positive_labels": int(oof_labels.sum()),
    "per_model_auc": {
        "nb4_e10": {"auc": auc_e10, "n_classes": n_e10},
        "tucker_sed": {"auc": auc_sed, "n_classes": n_sed},
        "e17": {"auc": auc_e17, "n_classes": n_e17},
    },
    "blend_auc": blend_results,
}
with open(out_dir / "oof_auc_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[OOF] Saved artifacts to {out_dir.resolve()}:")
for p in sorted(out_dir.glob("oof_*")):
    print(f"  {p.name}  {p.stat().st_size/1024:.1f}KB")

total = time.time() - START
print(f"\nOOF total {total:.0f}s ({total/60:.1f} min)")
'''

# ============================================================================
# Patch source: insert e17 inference cell + replace BLEND with OOF.
# Also override test_files to use labeled_ss 66 files.
# ============================================================================

# Override test file loading: use labeled SS files (66 files, glob from train_soundscapes/)
TEST_FILES_OVERRIDE_OLD = 'test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))'
TEST_FILES_OVERRIDE_NEW = '''# OOF override: use labeled SS 66 files instead of test_soundscapes
_labels_df_for_files = pd.read_csv(BASE / "train_soundscapes_labels.csv")
_oof_filenames = sorted(_labels_df_for_files["filename"].unique().tolist())
test_files = []
for _fn in _oof_filenames:
    _p = TRAIN_SC_DIR / _fn
    if _p.exists():
        test_files.append(str(_p))
print(f"[OOF] Using {len(test_files)} labeled SS files (of {len(_oof_filenames)} in labels CSV)")
# original test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))'''

assert TEST_FILES_OVERRIDE_OLD in SRC, "Could not find test_files glob to override"
mod = SRC.replace(TEST_FILES_OVERRIDE_OLD, TEST_FILES_OVERRIDE_NEW)

# Insert e17 inference cell + replace BLEND
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e17-infer", E17_INFER))\n'
    'cells.append(code_cell("oof", BLEND))'  # rename cell tag
)
assert CELL_APPEND_MARKER in mod, "Could not find blend cells.append() in tucker source"
mod = mod.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

# Inject E17_INFER constant + replace BLEND constant body
BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E17 inference cell + OOF computation cell ===\n'
    'E17_INFER = ' + repr(E17_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, E17_CONST_BLOCK)

OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + OOF_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

# Output filename
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_oof.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp024 OOF: exp019 base blend pipeline on labeled_ss 66 files (per-class AUC validation)\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp019 blend pipeline (NB4+Tucker+e17) を labeled_ss 66 file に対して推論、per-class AUC + blend AUC を測定して以後の実験基盤とする。Tucker 公開 5-fold ONNX",
)

import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_oof_inner.py")})
