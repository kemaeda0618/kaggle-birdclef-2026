"""Patch exp090 NB V3 to:
1. Override test_paths to use labeled SS files (66 files / 792 windows)
2. Override sample_submission to match labeled SS row_ids
3. Save intermediate stream + final blend outputs to /kaggle/working/

This lets us get exp090 BLEND predictions on labeled SS for analysis.
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp090_oof_v3.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Patch 1: bridge cell — override test_paths with labeled SS files
# ============================================================
override_bridge_addon = '''
# ============================================================
# ★ V3 PATCH: Override test_paths with labeled SS files (66 files / 792 windows)
# This lets exp090 blend run on labeled SS instead of empty test_soundscapes.
# ============================================================
TRAIN_SC_LABELS_PATH = BASE / "train_soundscapes_labels.csv"
_sc_labels = pd.read_csv(TRAIN_SC_LABELS_PATH)
_labeled_files = sorted(_sc_labels["filename"].unique().tolist())
print(f"\\n[V3] Override test_paths with {len(_labeled_files)} labeled SS files")

_new_test_paths = []
for fn in _labeled_files:
    fn_str = str(fn)
    if not fn_str.endswith(".ogg"):
        fn_str = fn_str + ".ogg"
    p = (BASE / "train_soundscapes") / fn_str
    if p.exists():
        _new_test_paths.append(p)
print(f"[V3] Resolved {len(_new_test_paths)} existing labeled SS file paths")

# Override
test_paths = _new_test_paths
test_files = [str(p) for p in test_paths]

# Rebuild meta_te + all_row_ids for these files
_new_meta_rows = []
_new_row_ids = []
import re
for fp in test_paths:
    fname = fp.stem
    m = re.search(r"_S(\\d+)_\\d{8}_(\\d{6})", fname)
    site = "S" + m.group(1) if m else "UNK"
    hour = int(m.group(2)[:2]) if m else -1
    for k in range(N_WINDOWS):
        rid = f"{fname}_{(k+1)*5}"
        _new_row_ids.append(rid)
        _new_meta_rows.append({
            "row_id": rid,
            "filename": fname,
            "site": site,
            "hour_utc": hour,
        })
meta_te = pd.DataFrame(_new_meta_rows)
all_row_ids = _new_row_ids
n_test_rows = len(all_row_ids)
print(f"[V3] meta_te: {len(meta_te)} rows, row_ids unique: {len(set(all_row_ids))}")

# Rebuild audio_cache with the new test_paths (replace what bridge cell built)
print(f"[V3] Rebuilding audio_cache from {len(test_paths)} labeled SS files...")
import soundfile as _sf_v3
audio_cache = []
for fp in test_paths:
    try:
        wav, sr_read = _sf_v3.read(str(fp), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr_read != SR:
            import librosa as _lb_v3
            wav = _lb_v3.resample(wav, orig_sr=sr_read, target_sr=SR)
    except Exception as e:
        wav = np.zeros(SR * 60, dtype=np.float32)
    target = SR * 60
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    audio_cache.append(wav.astype(np.float32))
print(f"[V3] audio_cache rebuilt: {len(audio_cache)} files")

# Also re-do probs_exp010 if probs_exp037 exists (it was computed on the original test_paths)
# Need to rerun exp037 inference on the new test set — but exp037 is in cell exp037_full_pipeline
# This patch must happen BEFORE exp037 prediction. So put this cell BEFORE exp037 if possible.
# For now, we'll handle this in a later cell (after exp037 runs).
print("[V3] Note: exp037 needs to re-run on the new test_paths (handled by override above)")
'''

# Save the override addon as a separate string to insert into bridge cell
# Find bridge cell and append
for c in nb["cells"]:
    if c.get("id") == "bridge":
        src = "".join(c["source"])
        # Insert override BEFORE the audio_cache loop (need to make exp037 use this test_paths)
        # Actually the bridge cell uses test_paths AFTER exp037 already ran (in exp037_full_pipeline)
        # So we need to:
        # 1. Override test_paths BEFORE exp037 runs (preferably in v313_03 or earlier)
        # 2. OR rerun exp037 inference on the new test_paths
        # For simplicity: do the override BEFORE exp037 by inserting before v313_03 (Cell 3: Data loading)
        # But exp037 pipeline cell is exp037_full_pipeline (which is after bridge anyway)
        # Wait, bridge cell happens AFTER exp037_full_pipeline. Let me re-check cell order.
        # Cell order from earlier inspection:
        # 24: v313_23 Test inference
        # 25: exp037_full_pipeline
        # 26: bridge
        # 27: sed-infer
        # 28: e29-infer
        # 30: blend
        # So exp037 runs BEFORE bridge. exp037 produces probs_exp037 on its own test_paths.
        # bridge cell THEN uses test_paths to build audio_cache for Tucker SED + exp029 R3.
        #
        # If we override test_paths in bridge cell, only Tucker + exp029 use the new paths.
        # probs_exp037 is already computed on the OLD test_paths.
        # So we need to ALSO override probs_exp037 to be on the new (labeled SS) paths.
        # This means we need to rerun exp037 inference on the new test_paths.
        # That's complex. Let me think of another approach.
        #
        # Alternative: override test_paths in v313_03 (data loading cell) where test_paths is first determined.
        # Then ALL cells will use the new test_paths.
        print(f"[INFO] Bridge cell found. Will need different approach.")
        break

# Actually, the test_paths is likely determined in v313_03 (Data loading & label parsing)
# Let me check that cell
for c in nb["cells"]:
    if c.get("id") == "v313_03":
        src = "".join(c["source"])
        if "test_paths" in src or "test_files" in src:
            print(f"[INFO] v313_03 has test_paths/test_files. Should override here.")
            # Find the right insertion point — after test_paths is set
            for ln in src.split("\n"):
                if "test_paths" in ln and "=" in ln:
                    print(f"  found: {ln.strip()[:120]}")
        break

# Plan B: prepend a new cell right after v313_03 that overrides test_paths
# Then all subsequent cells use the new test_paths
override_cell_src = '''# ============================================================
# ★ V3 PATCH: Override test_paths with labeled SS files (66 / 792 windows)
# Inserted right after v313_03 so all downstream cells (exp037, Tucker, exp029, blend) use labeled SS.
# ============================================================
import pandas as _pd_v3
import re as _re_v3
from pathlib import Path as _Path_v3

_BASE = BASE if "BASE" in dir() else _Path_v3("/kaggle/input/competitions/birdclef-2026")
TRAIN_SC_LABELS_PATH_V3 = _BASE / "train_soundscapes_labels.csv"
_sc_labels = _pd_v3.read_csv(TRAIN_SC_LABELS_PATH_V3)
_labeled_files = sorted(_sc_labels["filename"].unique().tolist())
print(f"\\n[V3] Override: {len(_labeled_files)} labeled SS files")

_new_test_paths = []
for fn in _labeled_files:
    fn_str = str(fn)
    if not fn_str.endswith(".ogg"):
        fn_str = fn_str + ".ogg"
    p = (_BASE / "train_soundscapes") / fn_str
    if p.exists():
        _new_test_paths.append(p)
print(f"[V3] Resolved {len(_new_test_paths)} existing labeled SS paths")

# Override test_paths (used by exp037 + bridge cells downstream)
test_paths = _new_test_paths
test_files = [str(p) for p in test_paths]

# Also rebuild meta_te (filename, site, hour_utc, row_id) for these files
_new_meta_rows = []
N_WINDOWS_V3 = 12
for fp in test_paths:
    fname = fp.stem
    m = _re_v3.search(r"_S(\\d+)_\\d{8}_(\\d{6})", fname)
    site = "S" + m.group(1) if m else "UNK"
    hour = int(m.group(2)[:2]) if m else -1
    for k in range(N_WINDOWS_V3):
        _new_meta_rows.append({
            "row_id": f"{fname}_{(k+1)*5}",
            "filename": fname,
            "site": site,
            "hour_utc": hour,
        })
meta_te = _pd_v3.DataFrame(_new_meta_rows)
print(f"[V3] meta_te overridden: {len(meta_te)} rows")
'''

# Save override cell to insert
override_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": override_cell_src.splitlines(keepends=True),
    "id": "v3_override",
}

# Find index of v313_03 and insert override AFTER it
cells = nb["cells"]
v303_idx = None
for i, c in enumerate(cells):
    if c.get("id") == "v313_03":
        v303_idx = i
        break
assert v303_idx is not None, "v313_03 not found"
print(f"\n[INFO] Inserting V3 override cell after v313_03 (idx {v303_idx})")
cells.insert(v303_idx + 1, override_cell)

# ============================================================
# Patch 2: blend cell — save labeled SS predictions
# ============================================================
save_addon = '''

# ============================================================
# ★ V3: Save labeled SS predictions for analysis
# ============================================================
_out_dir_v3 = Path("/kaggle/working")

# Save per-stream predictions on labeled SS (shape: (792, 234) = 66 files * 12 windows)
np.savez_compressed(_out_dir_v3 / "exp037_oof_ss.npz", probs_exp010.reshape(-1, probs_exp010.shape[-1]))
np.savez_compressed(_out_dir_v3 / "tucker_sed_ss.npz", probs_sed.reshape(-1, probs_sed.shape[-1]))
np.savez_compressed(_out_dir_v3 / "exp029_r3_ss.npz", probs_e17.reshape(-1, probs_e17.shape[-1]))
np.savez_compressed(_out_dir_v3 / "exp090_blend_ss.npz", preds_array)

# Build + save truth (832, 234) — labeled SS truth aligned with the 66-file × 12-window meta_te
_sc_lbl = pd.read_csv(BASE / "train_soundscapes_labels.csv").drop_duplicates()
if _sc_lbl["start"].dtype == object:
    _sc_lbl["start_sec"] = pd.to_timedelta(_sc_lbl["start"]).dt.total_seconds().astype(int)
else:
    _sc_lbl["start_sec"] = _sc_lbl["start"].astype(int)

_truth_ss = np.zeros((len(meta_te), N_CLASSES), dtype=np.float32)
for _, _row in _sc_lbl.iterrows():
    _fn = _row["filename"]
    _start = int(_row["start_sec"])
    _win_idx = _start // 5
    _rid = f"{_fn}_{(_win_idx + 1) * 5}"
    _match = meta_te[meta_te["row_id"] == _rid].index
    if len(_match) == 0:
        continue
    _w = _match[0]
    for _lbl in str(_row["primary_label"]).split(";"):
        _lbl = _lbl.strip()
        if _lbl in [str(c) for c in PRIMARY_LABELS]:
            _bc_i = PRIMARY_LABELS.index(_lbl)
            _truth_ss[_w, _bc_i] = 1.0
np.savez_compressed(_out_dir_v3 / "truth_ss.npz", _truth_ss)

# Meta
meta_te.to_parquet(_out_dir_v3 / "meta_ss.parquet")
# species labels
with open(_out_dir_v3 / "primary_labels.json", "w") as _f:
    json.dump(PRIMARY_LABELS, _f)

print(f"\\n[V3 SAVE] Outputs in {_out_dir_v3}:")
for _p in sorted(_out_dir_v3.glob("*.npz")) + sorted(_out_dir_v3.glob("*.parquet")) + sorted(_out_dir_v3.glob("*.json")):
    print(f"  {_p.name}  {_p.stat().st_size/1e6:.2f}MB")

# Sanity check
print(f"\\n[V3 SANITY]")
print(f"  exp037_oof: shape {probs_exp010.reshape(-1, probs_exp010.shape[-1]).shape}, mean {probs_exp010.mean():.4f}")
print(f"  tucker_sed: shape {probs_sed.reshape(-1, probs_sed.shape[-1]).shape}, mean {probs_sed.mean():.4f}")
print(f"  exp029_r3: shape {probs_e17.reshape(-1, probs_e17.shape[-1]).shape}, mean {probs_e17.mean():.4f}")
print(f"  exp090_blend: shape {preds_array.shape}, mean {preds_array.mean():.4f}")
print(f"  truth_ss: shape {_truth_ss.shape}, positives {int(_truth_ss.sum())}, active species {int((_truth_ss.sum(axis=0)>0).sum())}")
'''

for c in cells:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Insert save_addon AFTER the submission save line
        target = 'submission.to_csv("submission.csv", index=False)'
        if target in src:
            new_src = src.replace(target, target + save_addon)
            c["source"] = new_src.splitlines(keepends=True)
            print(f"[OK] Added save logic to blend cell")
            break
        else:
            print(f"[WARN] target 'submission.to_csv' not found in blend cell")

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")

# Compile check
print("\n=== Compile check (strict) ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} (id={c.get('id','?')}): {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
