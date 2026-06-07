"""v3 patch for exp069a: Stream audio loading to avoid OOM.

V2 でも 10,658 files × 60s × 32000Hz × float32 = ~80GB を all_waves dict に pre-load しようとして
4000 files 時点 (~30GB) で Kaggle OOM killer に殺された。

Fix:
  - Cell 12: load_all_samples() 削除、ts_paths だけ保持
  - Cell 16: inference loop 内で 1 file ずつ load → mel → 7 model forward → discard
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


# ─────────── Cell 12: Don't preload all audio (just save file list) ───────────
set_code(12, """# Skip preloading all audio (OOM avoidance: 10,658 files × 60s × 32kHz × float32 ≈ 80GB)
# Will stream load one file at a time in the inference loop (Cell 16)

ts_paths = ts_files  # already sorted Path objects from Cell 11
print(f"Will stream {len(ts_paths)} train_soundscape files (no preload)")
""")


# ─────────── Cell 16: Streaming inference loop ───────────
set_code(16, """# ─── Streaming inference: load 1 file at a time, run 7 models, accumulate output ───
# 出力: per-model (n_files, 12, 7, 206) + ensemble (n_files, 12, 206)

N_FILES = len(ts_paths)
N_SEGMENTS = InferenceConfig.num_segments_sample   # 12
N_CLASSES_BC25 = InferenceConfig.num_classes        # 206
N_MODELS = len(all_models)                          # 7
SR = InferenceConfig.sample_rate                    # 32000

# Storage:
per_model_probs = np.zeros((N_FILES, N_SEGMENTS, N_MODELS, N_CLASSES_BC25), dtype=np.float16)
ensemble_probs  = np.zeros((N_FILES, N_SEGMENTS, N_CLASSES_BC25), dtype=np.float16)
file_ids = []


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


def load_one_60s(fp, sr=SR):
    wave, _ = librosa.load(str(fp), sr=sr, mono=True)
    target_len = sr * 60
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    elif len(wave) > target_len:
        wave = wave[:target_len]
    return wave.astype(np.float32)


PREDS_WEIGHTS = InferenceConfig.preds_weights  # (7,)
print(f"Ensemble weights: {PREDS_WEIGHTS}")

t0 = time.time()
with torch.no_grad():
    for sample_ind, fp in enumerate(tqdm.tqdm(ts_paths, desc="Stream inference")):
        # Load + compute mel for THIS file only
        wave = load_one_60s(fp)
        single_waves = {fp.stem: wave}
        ds_single = InferenceDataset(
            all_waves=single_waves,
            feature_extractor=mel_spec_generator,
            full_signal=ModelsGroupConfig.full_signal_to_spectr,
            hop_length=ModelsGroupConfig.hop_length,
            img_size=ModelsGroupConfig.img_size,
            duration_sec=ModelsGroupConfig.duration,
            slice_step_sec=ModelsGroupConfig.slice_step_sec,
            num_segments_sample=N_SEGMENTS,
            separate_norm=InferenceConfig.separate_norm,
        )
        mel_spec, filename = ds_single[0]
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.unsqueeze(1).expand(-1, 3, -1, -1)
        mel_spec = mel_spec.to(DEVICE).to(torch.float16)

        per_segs = []
        for model_ind, model in enumerate(all_models):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = model.backbone(mel_spec)[-1]
                segwise = model.get_head_preds(feat)
            mapping = getattr(model, 'multilabel_to_train_labels', 'one2one')
            segwise = multilabel_to_train(segwise, mapping)
            per_segs.append(segwise.astype(np.float32))
            per_model_probs[sample_ind, :, model_ind, :] = segwise.astype(np.float16)

        per_arr = np.stack(per_segs, axis=0)  # (7, 12, 206)
        ensemble = (per_arr * PREDS_WEIGHTS.reshape(-1, 1, 1)).sum(0)
        ensemble = gauss_convolve_np(ensemble)
        ensemble_probs[sample_ind] = ensemble.astype(np.float16)
        file_ids.append(filename)

        # Explicit cleanup to keep RAM low
        del wave, single_waves, ds_single, mel_spec, per_segs, per_arr, ensemble

        if (sample_ind + 1) % 200 == 0 or sample_ind == N_FILES - 1:
            elapsed = time.time() - t0
            rate = (sample_ind + 1) / elapsed
            eta = (N_FILES - sample_ind - 1) / rate / 60
            print(f"  [{sample_ind+1}/{N_FILES}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"Stream inference done in {(time.time()-t0)/60:.1f} min")


# ─── Save outputs to /kaggle/working ───
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

with open(OUT_DIR / "babych_label2ind.json", "w") as f:
    json.dump(InferenceConfig.label2ind, f, indent=2)

_model_names = [wname.split("_")[0] + "_" + wname.split("_")[1] for wname, _cfg in MODELS_GROUP_META_1]
with open(OUT_DIR / "babych_model_names.json", "w") as f:
    json.dump(_model_names, f, indent=2)

file_to_row = {fid: i for i, fid in enumerate(file_ids)}
with open(OUT_DIR / "file_index.json", "w") as f:
    json.dump(file_to_row, f, indent=2)

print(f"OK exp069a streaming inference DONE")
""")


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v3 patched (streaming): {NB_PATH}")
