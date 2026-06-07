"""v2 patch for exp069a Babych pseudo gen NB.

Fixes from user feedback:
  1. Path: /kaggle/input/birdclef-2026 → /kaggle/input/competitions/birdclef-2026 (with fallback)
  2. Save BOTH per-model (7) AND ensemble outputs (Babych pseudo can be used per-model)
  3. Remove api.dataset_create_new() — kernel auth 不可、/kaggle/working/ に save するだけ
"""
import json, sys, io, os
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def set_code(i, text):
    nb["cells"][i]["cell_type"] = "code"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None


# ─────────── Cell 11: Fix DATA_PATH (path bug) ───────────
set_code(11, """# BC2026 data paths (Kaggle new env では competitions/ 階層追加)
_data_path_candidates = [
    "/kaggle/input/competitions/birdclef-2026",   # New Kaggle env (2026-05+)
    "/kaggle/input/birdclef-2026",                 # Old Kaggle env (fallback)
]
DATA_PATH = None
for _p in _data_path_candidates:
    if Path(_p).exists():
        DATA_PATH = _p; break
assert DATA_PATH is not None, f"BC2026 data not found in any of: {_data_path_candidates}"
TRAIN_SC_DIR = f"{DATA_PATH}/train_soundscapes"

# Babych weights from public Kaggle Dataset
_babych_candidates = [
    "/kaggle/input/birdclef2025-1st-place-ensemble",
    "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble",
]
BABYCH_DIR = None
for _p in _babych_candidates:
    if Path(_p).exists():
        BABYCH_DIR = _p; break
assert BABYCH_DIR is not None, f"Babych weights not found in any of: {_babych_candidates}"

# Output → /kaggle/working/ (kernel output 経由で M-R3 から参照、Dataset upload は不要)
OUT_DIR = Path("/kaggle/working")

assert Path(TRAIN_SC_DIR).exists(), f"train_soundscapes not found at {TRAIN_SC_DIR}"
ts_files = sorted(Path(TRAIN_SC_DIR).glob("*.ogg"))
print(f"DATA_PATH: {DATA_PATH}")
print(f"BABYCH_DIR: {BABYCH_DIR}")
print(f"train_soundscapes files: {len(ts_files)}")
print(f"Babych weights: {len(list(Path(BABYCH_DIR).glob('*.pt')))} .pt files")
""")


# ─────────── Cell 16: Save per-model + ensemble, no dataset upload ───────────
set_code(16, """# ─── Inference: run all 7 models on all train_soundscapes chunks ───
# 出力: per-model probability (n_files, 12 chunks, 7 models, 206 class) + ensemble (n_files, 12, 206)

N_FILES = len(all_waves)
N_SEGMENTS = InferenceConfig.num_segments_sample   # 12
N_CLASSES_BC25 = InferenceConfig.num_classes        # 206
N_MODELS = len(all_models)                          # 7

# Build dataset (re-uses model 0's mel_spec_generator)
dataset = InferenceDataset(
    all_waves=all_waves,
    feature_extractor=mel_spec_generator,
    full_signal=ModelsGroupConfig.full_signal_to_spectr,
    hop_length=ModelsGroupConfig.hop_length,
    img_size=ModelsGroupConfig.img_size,
    duration_sec=ModelsGroupConfig.duration,
    slice_step_sec=ModelsGroupConfig.slice_step_sec,
    num_segments_sample=N_SEGMENTS,
    separate_norm=InferenceConfig.separate_norm,
)


def gauss_convolve_np(arr, weights=np.array([0.1, 0.2, 0.4, 0.2, 0.1])):
    from scipy.ndimage import convolve1d
    return convolve1d(arr, weights, axis=0, mode='nearest')


def multilabel_to_train(preds_segwise, multilabel_to_train_labels):
    if isinstance(multilabel_to_train_labels, dict):
        y = np.zeros((N_SEGMENTS, N_CLASSES_BC25), dtype=np.float32)
        for multi_ind, train_ind in multilabel_to_train_labels.items():
            y[:, train_ind] = preds_segwise[:, multi_ind]
        return y
    return preds_segwise


PREDS_WEIGHTS = InferenceConfig.preds_weights  # (7,)
print(f"Ensemble weights: {PREDS_WEIGHTS}")

# Storage:
#   per_model_probs: (n_files, 12, 7, 206) float16  → ~120MB × 7 = ~840MB
#   ensemble_probs:  (n_files, 12, 206) float16     → ~120MB
per_model_probs = np.zeros((N_FILES, N_SEGMENTS, N_MODELS, N_CLASSES_BC25), dtype=np.float16)
ensemble_probs  = np.zeros((N_FILES, N_SEGMENTS, N_CLASSES_BC25), dtype=np.float16)
file_ids = []

t0 = time.time()
with torch.no_grad():
    for sample_ind in tqdm.tqdm(range(N_FILES), desc="Inference"):
        mel_spec, filename = dataset[sample_ind]
        # mel_spec: (12, n_mels, time_frames) → 3-channel
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.unsqueeze(1).expand(-1, 3, -1, -1)
        mel_spec = mel_spec.to(DEVICE).to(torch.float16)

        per_segs = []
        for model_ind, model in enumerate(all_models):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = model.backbone(mel_spec)[-1]
                segwise = model.get_head_preds(feat)   # (12, num_class) numpy
            mapping = getattr(model, 'multilabel_to_train_labels', 'one2one')
            segwise = multilabel_to_train(segwise, mapping)
            per_segs.append(segwise.astype(np.float32))
            per_model_probs[sample_ind, :, model_ind, :] = segwise.astype(np.float16)

        per_arr = np.stack(per_segs, axis=0)  # (7, 12, 206)
        ensemble = (per_arr * PREDS_WEIGHTS.reshape(-1, 1, 1)).sum(0)  # (12, 206)
        ensemble = gauss_convolve_np(ensemble)
        ensemble_probs[sample_ind] = ensemble.astype(np.float16)
        file_ids.append(filename)

        if (sample_ind + 1) % 200 == 0 or sample_ind == N_FILES - 1:
            elapsed = time.time() - t0
            rate = (sample_ind + 1) / elapsed
            eta = (N_FILES - sample_ind - 1) / rate / 60
            print(f"  [{sample_ind+1}/{N_FILES}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"Inference done in {(time.time()-t0)/60:.1f} min")


# ─── Save outputs to /kaggle/working (kernel output 経由で M-R3 から参照) ───
np.savez_compressed(
    OUT_DIR / "babych_per_model_206.npz",
    probs=per_model_probs,
    file_ids=np.array(file_ids),
)
print(f"Saved per-model NPZ: {(OUT_DIR / 'babych_per_model_206.npz').stat().st_size / 1e6:.1f} MB")

np.savez_compressed(
    OUT_DIR / "babych_ensemble_206.npz",
    probs=ensemble_probs,
    file_ids=np.array(file_ids),
)
print(f"Saved ensemble NPZ: {(OUT_DIR / 'babych_ensemble_206.npz').stat().st_size / 1e6:.1f} MB")

# Save BC25 species index → label mapping
with open(OUT_DIR / "babych_label2ind.json", "w") as f:
    json.dump(InferenceConfig.label2ind, f, indent=2)
print(f"Saved label2ind: {len(InferenceConfig.label2ind)} BC25 species")

# Save model name list (for per-model identification)
_model_names = []
for wname, _cfg in MODELS_GROUP_META_1:
    _model_names.append(wname.split("_")[0] + "_" + wname.split("_")[1])  # rough id
with open(OUT_DIR / "babych_model_names.json", "w") as f:
    json.dump(_model_names, f, indent=2)
print(f"Saved model_names: {_model_names}")

# Save file_id → row index
file_to_row = {fid: i for i, fid in enumerate(file_ids)}
with open(OUT_DIR / "file_index.json", "w") as f:
    json.dump(file_to_row, f, indent=2)

# NOTE: api.dataset_create_new() を削除
#  Kaggle kernel 内では auth 不可。/kaggle/working/ に save した時点で
#  Save Version 完了時に kernel output として保存される。M-R3 NB は kernel_sources
#  で maekeso/birdclef2026-exp069a-babych-pseudo-gen を attach すれば参照可能。

print(f"OK exp069a pseudo gen DONE")
""")


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nv2 patched: {NB_PATH}")
print(f"Cells: {len(nb['cells'])}")
