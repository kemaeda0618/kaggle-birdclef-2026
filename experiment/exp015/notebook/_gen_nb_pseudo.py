"""Generate exp015 NB: R1 student → pseudo CSV (R2 input).

Pipeline:
- Load exp015 R1 trained convnext_pico SED student (ckpt_best_ns22.pth) from
  kernel_sources (`maekeso/birdclef2026-exp015-train`).
- Run inference on all 10,658 train_soundscapes (12 windows × 5s each).
- Blend: 0.5*sigmoid(clip_logits) + 0.5*sigmoid(frame_max).
- Gaussian smooth across windows per file (sigma=0.65).
- Power Transform γ=1.2 (Babych H1 sharpening, training-time pseudo target).
- NO threshold filter (Natsume LB evidence: filter hurts).
- Save pseudo_labels.csv (filename, start_sec, end_sec, label_0...label_233).
- Upload as Kaggle Dataset `maekeso/exp015-r1-pseudo`.

Run: python _gen_nb_pseudo.py  ->  writes nb_pseudo.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}


cells = []

# =============================================================================
# Cell 0: Header
# =============================================================================
cells.append(md_cell(r"""# exp015 — R1 → Pseudo (R2 input)

**Pipeline:** R1 student (exp015 convnext_pico.d1_in1k SED, ConvNeXt-Pico LN+GELU+DW) → generate pseudo on
all 10,658 train_soundscapes → upload as Kaggle Dataset for R2 training.

## Multi-Iterative Noisy Student (Babych BC2025 1st-place recipe)
- **R1** student (current model) acts as teacher for R2
- R2 student trains on focal + labeled_sc + **pseudo_sc** with stronger aug
- Repeat for R3, R4, ... (each round adds noise / inflates pseudo share)

## NB outputs
- `pseudo_labels.csv`: 10,658 × 12 = ~127,896 rows × (3 meta + 234 labels)
- Uploaded as `maekeso/exp015-r1-pseudo`

## Required inputs (attach when pushing)
- `birdclef-2026` (competition data)
- `tuckerarrants/perch-v2-no-dft-onnx` (NOT used here — bundled for env parity)
- kernel_sources: `maekeso/birdclef2026-exp015-train` (R1 ckpts)

## Constraints
- GPU T4x2, ~75 min total
- Internet ON for Kaggle Dataset upload
- NO threshold filter (Natsume LB evidence shows filter hurts)
- Power Transform γ=1.2 (Babych H1, training-time pseudo target sharpening)
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — install, imports
# ============================================================
import subprocess, sys, os, time

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "onnxruntime", "onnxruntime-gpu"], check=False)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "timm>=1.0.0", "onnxruntime-gpu", "librosa", "soundfile"])

# Re-import onnxruntime after install
for _mod_name in list(sys.modules):
    if _mod_name.startswith("onnxruntime"):
        del sys.modules[_mod_name]

import json, glob, math, gc
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchaudio
import timm
from scipy.ndimage import gaussian_filter1d

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

START = time.time()
""", "setup"))

# =============================================================================
# Cell 2: Paths
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths — locate competition data, R1 ckpt
# ============================================================
# BC2026 competition data
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"
print(f"BASE: {BASE}")

TS_DIR = BASE / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
assert TS_DIR.is_dir(), f"train_soundscapes missing: {TS_DIR}"

# R1 ckpt — kernel_sources (train NB output)
R1_DIR = None
CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp015-train"),
    Path("/kaggle/input/birdclef2026-exp015-train"),
    Path("/kaggle/input/datasets/maekeso/exp015-state"),
    Path("/kaggle/input/exp015-state"),
]
for p in CANDIDATES:
    if p.exists():
        if any(p.rglob("ckpt_best_*.pth")) or any(p.rglob("ckpt_latest*.pth")):
            R1_DIR = p; break
if R1_DIR is None:
    for hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        R1_DIR = hit.parent; break

assert R1_DIR is not None, (
    "R1 ckpt not found. Attach maekeso/birdclef2026-exp015-train as kernel_sources, "
    "or maekeso/exp015-state as dataset_sources"
)
print(f"R1 dir: {R1_DIR}")
for f in sorted(R1_DIR.rglob("ckpt_*.pth"))[:8]:
    print(f"  {f.relative_to(R1_DIR)!s}  {f.stat().st_size/1e6:.2f} MB")
""", "paths"))

# =============================================================================
# Cell 3: Config — must match training
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Config — must match R1 training exactly
# ============================================================
NUM_CLASSES = 234
SR = 32000

TRAIN_DURATION = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 256
FMIN           = 20
FMAX           = 16000

BACKBONE = "convnext_pico.d1_in1k"

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

N_WINDOWS = 12       # 60s / 5s
CHUNK_N   = TRAIN_SAMPLES

# Post-processing
GAUSS_SIGMA  = 0.65
POWER_GAMMA  = 1.2

# Label order (must match training; PRIMARY_LABELS from sample_submission)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"Backbone: {BACKBONE}")
print(f"Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
print(f"Post: gauss sigma={GAUSS_SIGMA}, power gamma={POWER_GAMMA}")
print(f"NO threshold filter (Natsume LB evidence)")
""", "config"))

# =============================================================================
# Cell 4: Model architecture (identical to training)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Model — rebuild convnext_pico SED architecture (same as R1 training)
# ============================================================
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits

print("OK model defs ready")
""", "model"))

# =============================================================================
# Cell 5: Load ckpt
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Load R1 ckpt — prefer ckpt_best_ns22.pth, fallback to latest
# ============================================================
CKPT_PRIORITY = ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth"]
ckpt_path = None
for name in CKPT_PRIORITY:
    hits = list(R1_DIR.rglob(name))
    if hits:
        ckpt_path = hits[0]; break
assert ckpt_path is not None, f"No ckpt found under {R1_DIR}"
print(f"Loading ckpt: {ckpt_path}")

try:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(ckpt_path), map_location="cpu")

print(f"  epoch={state.get('epoch')}, "
      f"best_ns22={state.get('best_ns22', float('nan')):.4f}, "
      f"best_macro={state.get('best_macro', float('nan')):.4f}")

model = BirdSEDModel().to(device)
model.load_state_dict(state["model_state"], strict=False)
model.eval()
model = model.to(memory_format=torch.channels_last)
print(f"OK model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
""", "load_ckpt"))

# =============================================================================
# Cell 6: Inference loop on train_soundscapes
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Inference — iterate over all train_soundscapes (10,658 files)
# ============================================================
import soundfile as sf
import librosa

def load_audio_32k_mono(path, target_samples=60 * SR):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < target_samples:
        wav = np.pad(wav, (0, target_samples - len(wav)))
    elif len(wav) > target_samples:
        wav = wav[:target_samples]
    return wav.astype(np.float32)


def file_to_chunks(path):
    wav = load_audio_32k_mono(path, target_samples=N_WINDOWS * CHUNK_N)
    chunks = wav.reshape(N_WINDOWS, CHUNK_N)
    return chunks.astype(np.float32)


def sigmoid_np(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                    np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50)))
                    ).astype(np.float32)


mel_tf = MelSpecTransform().to(device)

sc_files = sorted(glob.glob(str(TS_DIR / "*.ogg")))
print(f"train_soundscapes: {len(sc_files)} files")
assert len(sc_files) > 0

all_filenames = []
all_start_secs = []
all_end_secs = []
all_probs = []   # list of (N_WINDOWS, NUM_CLASSES) per file

t0 = time.time()
N_FILES = len(sc_files)

with torch.no_grad():
    for fi, fpath in enumerate(sc_files):
        stem = Path(fpath).stem
        try:
            chunks = file_to_chunks(fpath)
        except Exception as e:
            print(f"WARN: failed to read {stem}: {e}")
            chunks = np.zeros((N_WINDOWS, CHUNK_N), dtype=np.float32)

        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)   # (12, 1, 160000)
        mel = mel_tf(wav_t)
        # per-instance standardize
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        mel = mel.to(memory_format=torch.channels_last)

        with autocast():
            clip_logits, framewise = model(mel, return_framewise=True)
            frame_max = framewise.max(dim=1).values
            p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()
            p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
        probs_file = 0.5 * p_clip + 0.5 * p_fmax     # (12, 234)

        # Per-file Gaussian smooth across windows axis
        probs_file = gaussian_filter1d(probs_file, sigma=GAUSS_SIGMA, axis=0,
                                        mode="nearest").astype(np.float32)

        all_probs.append(probs_file)
        for wi in range(N_WINDOWS):
            all_filenames.append(stem)
            all_start_secs.append(wi * TRAIN_DURATION)
            all_end_secs.append((wi + 1) * TRAIN_DURATION)

        if (fi + 1) % 200 == 0 or fi == N_FILES - 1 or fi == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 1e-6)
            eta = (N_FILES - fi - 1) / max(rate, 1e-6)
            print(f"  [{fi+1:5d}/{N_FILES}] {elapsed:6.1f}s  {rate:5.2f} files/s  ETA {eta/60:5.1f} min")

prob_mat = np.concatenate(all_probs, axis=0).astype(np.float32)   # (N*12, 234)
filenames_arr = np.array(all_filenames)
start_secs_arr = np.array(all_start_secs, dtype=np.float32)
end_secs_arr   = np.array(all_end_secs,   dtype=np.float32)
print(f"\nInference: {prob_mat.shape}, mean={prob_mat.mean():.4f}, max={prob_mat.max():.4f} "
      f"in {time.time()-t0:.0f}s")
""", "inference"))

# =============================================================================
# Cell 7: Post-processing — Power Transform γ=1.2 (no filter)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Post-processing — Power Transform γ=1.2 only (no row_max filter)
# ============================================================
# Backup raw before transform
print(f"Pre-PT: mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"95%ile={np.percentile(prob_mat, 95):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")

prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)

print(f"\nPost-PT (γ={POWER_GAMMA}): mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"95%ile={np.percentile(prob_mat, 95):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")

# row_max stats (for diagnostics only, NOT filtering — Natsume LB evidence)
row_max = prob_mat.max(axis=1)
print(f"\nrow_max stats (for diagnostics, NO filter applied):")
print(f"  25%ile: {np.percentile(row_max, 25):.4f}")
print(f"  50%ile: {np.percentile(row_max, 50):.4f}")
print(f"  75%ile: {np.percentile(row_max, 75):.4f}")
print(f"  99%ile: {np.percentile(row_max, 99):.4f}")
print(f"  >=0.05 rows: {(row_max >= 0.05).sum()}/{len(row_max)} "
      f"({(row_max >= 0.05).mean()*100:.1f}%)")
""", "postproc"))

# =============================================================================
# Cell 8: Save pseudo_labels.csv
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: Save pseudo_labels.csv — full output (no filter)
# ============================================================
df = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)
df.insert(0, "filename",  filenames_arr)
df.insert(1, "start_sec", start_secs_arr)
df.insert(2, "end_sec",   end_secs_arr)

out_path = Path("/kaggle/working/pseudo_labels.csv")
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"  shape: {df.shape}")
print(f"  size:  {out_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  files: {df['filename'].nunique()}")

print("\n=== Preview ===")
print(df.head(3))
print(f"\nTotal time: {(time.time()-START)/60:.1f} min")
""", "save_csv"))

# =============================================================================
# Cell 9: Upload as Kaggle Dataset
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 9: Upload to maekeso/exp015-r1-pseudo
# ============================================================
import shutil, tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

DATASET_USER = "maekeso"
DATASET_SLUG = "birdclef2026-exp015-pseudo-r1"   # match exp014 convention
DATASET_TITLE = "exp015 R1 pseudo labels"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # Copy only the pseudo_labels.csv (and optionally metadata)
    src = Path("/kaggle/working/pseudo_labels.csv")
    shutil.copy(str(src), str(td / src.name))
    print(f"Staged: {src.name}  {src.stat().st_size/1e6:.2f} MB")

    meta = {
        "title": DATASET_TITLE,
        "id": f"{DATASET_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = f"R1 pseudo (epoch={state.get('epoch')}, ns22={state.get('best_ns22', float('nan')):.4f})"
    uploaded = False
    # Try update existing
    try:
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"OK Uploaded to {DATASET_USER}/{DATASET_SLUG} (new version, {version_notes})")
        uploaded = True
    except Exception as e:
        msg = str(e)
        print(f"  dataset_create_version error: {msg[:300]}")
        if "not found" in msg.lower() or "404" in msg or "Could not find dataset" in msg:
            try:
                api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
                print(f"OK Created {DATASET_USER}/{DATASET_SLUG} (first time)")
                uploaded = True
            except Exception as e2:
                print(f"  dataset_create_new error: {str(e2)[:300]}")

    if not uploaded:
        print("\nUpload failed — pseudo_labels.csv remains in /kaggle/working/ for manual upload")
""", "upload"))

# =============================================================================
# Assemble notebook
# =============================================================================
nb_out = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb_pseudo.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
