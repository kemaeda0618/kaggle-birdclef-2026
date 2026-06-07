"""Patch Babych reference inference NB → exp069a pseudo gen NB for BC2026 train_soundscapes.

Changes:
  - Cell 0:  Title update
  - Cell 1:  Remove openvino install, use Kaggle env defaults + ensure timm
  - Cell 3:  Imports: drop openvino
  - Cell 7:  Drop VINOEngine class (PyTorch GPU direct)
  - Cell 11: DATA_PATH to BC2026, load train_soundscapes file list
  - Cell 12: load_all_samples adapted for train_soundscapes (10,658 files)
  - Cell 14: prepare_models — PyTorch CUDA FP16 (no OpenVINO)
  - Cell 16: Inference loop → save NPZ + upload to Kaggle Dataset
"""
import json, sys, io, os
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def set_md(i, text):
    nb["cells"][i]["cell_type"] = "markdown"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    if "outputs" in nb["cells"][i]: del nb["cells"][i]["outputs"]
    if "execution_count" in nb["cells"][i]: del nb["cells"][i]["execution_count"]


def set_code(i, text):
    nb["cells"][i]["cell_type"] = "code"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None


# ─────────── Cell 0: Title ───────────
set_md(0, """# exp069a: Babych 7-ensemble pseudo gen on BC2026 train_soundscapes

**Purpose**: Babych BC25 1位 7-model ensemble で BC2026 train_soundscapes (~10,658 files) に推論、
raw probability (n_files, 12 chunks, 206 BC25 species) を NPZ 保存して Kaggle Dataset upload。

**Output**: maekeso/birdclef2026-exp069a-babych-pseudo
  - babych_raw_206.npz: ensemble probability (n_files × 12 × 206)
  - babych_label2ind.json: BC25 species → 206-col index mapping
  - file_index.json: file_id → row index in NPZ

**後段**: exp069c で BC25 → BC26 species mapping + exp048 と merge → 3 pseudo CSV
""")


# ─────────── Cell 1: pip install (Kaggle env用に簡素化) ───────────
set_code(1, """# Kaggle env (Internet=True 想定) でデフォルト available なライブラリで動く
# OpenVINO は使わない、PyTorch GPU で直接推論
import sys
print("Python:", sys.version)

# Ensure timm + torch + librosa available
import timm, torch, librosa
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
print(f"timm  {timm.__version__}")
print(f"librosa {librosa.__version__}")
""")


# ─────────── Cell 3: Imports ───────────
set_code(3, """import os, math, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
import timm
import tqdm.auto as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
""")


# ─────────── Cell 7: Drop VINOEngine, keep InferenceDataset only ───────────
set_code(7, """class InferenceDataset:
    \"\"\"Babych InferenceDataset adapted from reference NB (no VINOEngine).\"\"\"
    def __init__(self, all_waves, feature_extractor, full_signal=True, hop_length=1252,
                 img_size=(224, 512), sample_rate=32000, duration_sec=5, slice_step_sec=5,
                 num_segments_sample=12, separate_norm=False):
        self.all_waves = all_waves
        self.all_filenames = list(all_waves.keys())
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.duration_timestamps = sample_rate * duration_sec
        self.slice_step_timestamps = sample_rate * slice_step_sec
        self.num_segments_sample = num_segments_sample
        self.full_signal = full_signal
        self.hop_length = hop_length
        self.img_size = img_size
        self.separate_norm = separate_norm

    def __len__(self):
        return len(self.all_filenames)

    def split_wave(self, wave):
        if self.slice_step_timestamps != self.duration_timestamps:
            pad_side_length = int((self.duration_timestamps - self.slice_step_timestamps) / 2)
            wave = np.pad(wave, (pad_side_length, pad_side_length))
            segments = [librosa.util.normalize(wave[i*self.slice_step_timestamps: i*self.slice_step_timestamps + self.duration_timestamps]) for i in range(self.num_segments_sample)]
        else:
            segments = [librosa.util.normalize(wave[i*self.duration_timestamps: (i+1)*self.duration_timestamps]) for i in range(self.num_segments_sample)]
        segments = np.stack(segments, 0)
        return segments

    def prepare_mel_specs_from_full_wave(self, wave):
        def get_wave_len(hop_length, time_bins):
            return (time_bins - 1) * hop_length
        total_time_bins = self.img_size[1] + (self.num_segments_sample - 1) * (self.img_size[1] * self.slice_step_timestamps / self.duration_timestamps)
        if not self.separate_norm:
            total_wave_len = get_wave_len(hop_length=self.hop_length, time_bins=total_time_bins)
            pad_side_length = math.ceil(((total_wave_len - len(wave)) / 2))
            wave = np.pad(wave, (pad_side_length, pad_side_length))
        mel_spec_full = self.feature_extractor(torch.tensor(wave.reshape(1, -1)), without_norm=self.separate_norm)
        if self.separate_norm:
            pad_side_length = int((total_time_bins - mel_spec_full.shape[-1]) // 2)
            extra_pad = 0
            if (total_time_bins - mel_spec_full.shape[-1]) % 2 != 0:
                extra_pad = 1
            mel_spec_full = F.pad(mel_spec_full, pad=(pad_side_length, pad_side_length + extra_pad, 0, 0), mode='constant', value=mel_spec_full.min())
        slice_step = int((self.img_size[1] * self.slice_step_timestamps / self.duration_timestamps))
        mel_specs = []
        for i in range(self.num_segments_sample):
            mel_specs.append(mel_spec_full[0, :, i*slice_step: (i*slice_step) + self.img_size[1]])
        mel_specs = torch.stack(mel_specs, 0)
        if self.separate_norm:
            mel_specs = self.feature_extractor.norm(mel_specs)
        return mel_specs

    def __getitem__(self, ind):
        filename = self.all_filenames[ind]
        wave = self.all_waves[filename]
        if self.full_signal:
            mel_specs = self.prepare_mel_specs_from_full_wave(wave=wave)
        else:
            # fallback to per-segment
            segments = self.split_wave(wave=wave)
            mel_specs = self.feature_extractor(torch.tensor(segments))
        return mel_specs, filename
""")


# ─────────── Cell 11: DATA_PATH BC2026 + train_soundscapes ───────────
set_code(11, """# BC2026 data paths
DATA_PATH = "/kaggle/input/birdclef-2026"
TRAIN_SC_DIR = f"{DATA_PATH}/train_soundscapes"

# Babych weights from public Kaggle Dataset
BABYCH_DIR = "/kaggle/input/birdclef2025-1st-place-ensemble"

# Output paths
OUT_DIR = Path("/kaggle/working/exp069a_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sanity check
assert Path(TRAIN_SC_DIR).exists(), f"train_soundscapes not found at {TRAIN_SC_DIR}"
assert Path(BABYCH_DIR).exists(), f"Babych weights not found at {BABYCH_DIR}, attach nikitababich/birdclef2025-1st-place-ensemble"

ts_files = sorted(Path(TRAIN_SC_DIR).glob("*.ogg"))
print(f"train_soundscapes files: {len(ts_files)}")
print(f"Babych weights dir: {len(list(Path(BABYCH_DIR).glob('*.pt')))} .pt files")
""")


# ─────────── Cell 12: load_all_samples for BC2026 train_soundscapes ───────────
set_code(12, """def load_all_samples(file_paths, sr=32000, batch_log=500):
    \"\"\"Load 60s audio from each train_soundscape file → dict {file_id: ndarray}.\"\"\"
    all_waves = {}
    t0 = time.time()
    for i, fp in enumerate(file_paths):
        wave, _ = librosa.load(str(fp), sr=sr, mono=True)
        # Pad/trim to 60s
        target_len = sr * 60
        if len(wave) < target_len:
            wave = np.pad(wave, (0, target_len - len(wave)))
        elif len(wave) > target_len:
            wave = wave[:target_len]
        all_waves[fp.stem] = wave.astype(np.float32)
        if (i + 1) % batch_log == 0 or i == len(file_paths) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  loaded [{i+1}/{len(file_paths)}] {elapsed:.0f}s ({rate:.1f} files/s)")
    return all_waves


# Load all train_soundscapes audio (~10,658 files × 60s × 32000Hz = ~7.6 GB float32 in RAM)
# Optimization: keep float16 in RAM, convert to float32 just before mel
# Or process in chunks to limit memory
print("Loading train_soundscapes audio...")
all_waves = load_all_samples(ts_files)
print(f"Loaded {len(all_waves)} files, total memory: ~{sum(w.nbytes for w in all_waves.values())/1e9:.1f} GB")
""")


# ─────────── Cell 14: prepare_models (PyTorch CUDA FP16) ───────────
set_code(14, """def prepare_models_pytorch(model_meta_list, models_dir, device, dtype=torch.float16):
    \"\"\"Load Babych weights as PyTorch models, push to CUDA + FP16. No OpenVINO.\"\"\"
    models = []
    for weights_name, model_config in tqdm.tqdm(model_meta_list, desc="Loading Babych models"):
        # Build model architecture (uses cell 5 CLEFClassifierSED + cell 9 configs)
        model = CLEFClassifierSED(config=model_config)
        pt_path = os.path.join(models_dir, weights_name)
        state = torch.load(pt_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        model.to(device).to(dtype)
        models.append(model)
        print(f"  loaded {weights_name[:60]}... ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return models


# Build & load all 7 Babych models
all_models = prepare_models_pytorch(MODELS_GROUP_META_1, BABYCH_DIR, DEVICE, dtype=torch.float16)
print(f"Loaded {len(all_models)} Babych models on {DEVICE} (FP16)")

# Reference mel spectrogram feature extractor from model 0 (all share same mel spec params)
mel_spec_generator = all_models[0].mel_spectr_generator
""")


# ─────────── Cell 16: Inference loop + NPZ save + upload ───────────
set_code(16, """# ─── Inference: run all 7 models on all train_soundscapes chunks ───
N_FILES = len(all_waves)
N_SEGMENTS = InferenceConfig.num_segments_sample   # 12
N_CLASSES_BC25 = InferenceConfig.num_classes        # 206

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
    \"\"\"Re-project sub-model output (e.g., Insecta/Amphibia 700 species) → 206 BC25 class space.\"\"\"
    if isinstance(multilabel_to_train_labels, dict):
        y = np.zeros((N_SEGMENTS, N_CLASSES_BC25), dtype=np.float32)
        for multi_ind, train_ind in multilabel_to_train_labels.items():
            y[:, train_ind] = preds_segwise[:, multi_ind]
        return y
    return preds_segwise


PREDS_WEIGHTS = InferenceConfig.preds_weights  # (7,)
print(f"Ensemble weights: {PREDS_WEIGHTS}")

# Storage for ensemble outputs: (n_files, 12, 206)
ensemble_probs = np.zeros((N_FILES, N_SEGMENTS, N_CLASSES_BC25), dtype=np.float16)
file_ids = []

t0 = time.time()
with torch.no_grad():
    for sample_ind in tqdm.tqdm(range(N_FILES), desc="Inference"):
        mel_spec, filename = dataset[sample_ind]
        # mel_spec: (12, n_mels, time_frames) → add channel dim and repeat to 3-channel
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.unsqueeze(1).expand(-1, 3, -1, -1)
        mel_spec = mel_spec.to(DEVICE).to(torch.float16)

        # Per-file ensemble
        per_model_segs = []  # list of (12, 206) np arrays
        for model_ind, model in enumerate(all_models):
            # Forward backbone → head
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = model.backbone(mel_spec)[-1]
                segwise = model.get_head_preds(feat)   # (12, num_class) numpy
            # Apply sub-model mapping if any
            mapping = getattr(model, 'multilabel_to_train_labels', 'one2one')
            segwise = multilabel_to_train(segwise, mapping)
            per_model_segs.append(segwise.astype(np.float32))

        # Weighted ensemble
        per_model_arr = np.stack(per_model_segs, axis=0)  # (7, 12, 206)
        ensemble = (per_model_arr * PREDS_WEIGHTS.reshape(-1, 1, 1)).sum(0)  # (12, 206)
        # Gaussian smooth across 12 segments
        ensemble = gauss_convolve_np(ensemble)
        ensemble_probs[sample_ind] = ensemble.astype(np.float16)
        file_ids.append(filename)

        if (sample_ind + 1) % 100 == 0 or sample_ind == N_FILES - 1:
            elapsed = time.time() - t0
            rate = (sample_ind + 1) / elapsed
            eta = (N_FILES - sample_ind - 1) / rate / 60
            print(f"  [{sample_ind+1}/{N_FILES}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"Inference done in {(time.time()-t0)/60:.1f} min")


# ─── Save outputs to /kaggle/working ───
np.savez_compressed(
    OUT_DIR / "babych_raw_206.npz",
    probs=ensemble_probs,
    file_ids=np.array(file_ids),
)
print(f"Saved probs NPZ: {(OUT_DIR / 'babych_raw_206.npz').stat().st_size / 1e6:.1f} MB")

# Save BC25 species index → label mapping
with open(OUT_DIR / "babych_label2ind.json", "w") as f:
    json.dump(InferenceConfig.label2ind, f, indent=2)
print(f"Saved label2ind: {len(InferenceConfig.label2ind)} BC25 species")

# Save file_id → row index
file_to_row = {fid: i for i, fid in enumerate(file_ids)}
with open(OUT_DIR / "file_index.json", "w") as f:
    json.dump(file_to_row, f, indent=2)


# ─── Upload to Kaggle Dataset ───
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

DATASET_SLUG = "maekeso/birdclef2026-exp069a-babych-pseudo"

meta = {
    "id": DATASET_SLUG,
    "title": "BC2026 exp069a Babych ensemble pseudo (206 BC25 class)",
    "licenses": [{"name": "CC0-1.0"}],
}
with open(OUT_DIR / "dataset-metadata.json", "w") as f:
    json.dump(meta, f)

try:
    api.dataset_status(DATASET_SLUG)
    print(f"Dataset {DATASET_SLUG} exists → creating new version")
    api.dataset_create_version(folder=str(OUT_DIR), version_notes="exp069a Babych pseudo", quiet=False)
except Exception as e:
    print(f"Dataset {DATASET_SLUG} new → creating: {str(e)[:60]}")
    api.dataset_create_new(folder=str(OUT_DIR), public=False, quiet=False)

print(f"✓ exp069a pseudo gen DONE: {DATASET_SLUG}")
""")


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nPatched: {NB_PATH}")
print(f"Cells: {len(nb['cells'])}")
