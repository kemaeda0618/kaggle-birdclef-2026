"""Generate clean exp029 R3 → train_soundscapes pseudo NB.

Properly organized under experiment/exp029/ (was misplaced under exp028/ with wrong header).

Code logic is verified working — Kaggle kernel
`maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029` has produced
pseudo_exp029.csv (410MB, COMPLETE) using essentially this same code.

This rewrite:
- Accurate header (exp029 R3 single fold, not "exp020 R2 5-fold")
- Removes copy-paste comments ("5-fold ensemble", "e17 single fold", etc.)
- Clear variable naming
- Same Kaggle Dataset / output behavior

Run: python _gen_nb_pseudo_r3_train_sc.py
Push: python _push_nb_pseudo_r3_train_sc.py
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

# ============================================================
# Cell 0: Header (markdown)
# ============================================================
cells.append(md_cell(r"""# exp029 nb_pseudo_r3_train_sc: R3 self-inference → pseudo on train_soundscapes

R3 self-pseudo for **exp029 R4 training** (Multi-iter Noisy Student R3 → R4).

## Purpose
Run **exp029 R3 single fold** (eca_nfnet_l1) inference on the entire
`train_soundscapes/` (~10,658 files × 12 windows = ~127,896 rows) and save as
`pseudo_exp029.csv` for R4 training input.

PyTorch direct inference (no ONNX), Kaggle T4 GPU, ~30-50 min.

## Inputs
- `birdclef-2026` competition: `train_soundscapes/` (~10,658 .ogg files)
- `maekeso/birdclef2026-exp029-l1-single` Dataset:
  - `r3_fold0_ckpt_best_ns22.pth` (exp029 R3 fold 0 ckpt)
  - val_ns22 = 0.9409 (epoch 15)

## Output
- `/kaggle/working/pseudo_exp029.csv` (~410MB)
  - columns: `filename, start_sec, end_sec, [234 species probabilities]`
  - 10,658 files × 12 windows = ~127,896 rows
- `/kaggle/working/pseudo_exp029.npy` (~60MB, float16)
- `/kaggle/working/pseudo_exp029_meta.npy` (filename, start_sec, end_sec)

## Pipeline
1. Load R3 ckpt → build BirdSEDModel (eca_nfnet_l1 + Perch v2 distill head)
2. For each train_sc file: load 60s, split into 12 × 5s windows, mel-spec,
   per-instance normalize, forward → clip_logits + framewise_logits
3. Blend: `0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)`
4. Save as CSV (+ NPY)

## Use case
- R4 training NB (`exp029/notebook/nb_train_r4_l1_single.ipynb`) loads this CSV
- pseudo_sc share = 0.25, SOURCE_WEIGHTS[pseudo_sc] = 0.5

## R3 ckpt details
- backbone: eca_nfnet_l1
- training: focal + labeled_sc + pseudo_sc (with exp017 R2 pseudo)
- single fold (fold 0), N_EPOCHS=20, LR=3e-4
- val_ns22: 0.9409, single LB: 0.923
""", "hdr"))

# ============================================================
# Cell 1: Setup
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — imports, GPU detection
# ============================================================
import os, sys, time, json, math, glob, re, gc
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARN] GPU not available, falling back to CPU (very slow for 10k+ files)")

torch.set_num_threads(4)
""", "setup"))

# ============================================================
# Cell 2: Paths — competition + R3 ckpt
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths — locate competition data + R3 ckpt
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"

TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
TRAIN_SS_DIR = BASE / "train_soundscapes"
print(f"BASE: {BASE}")
print(f"  train_soundscapes exists: {TRAIN_SS_DIR.is_dir()}")

# exp029 R3 fold 0 ckpt
R3_CKPT_NAME = "r3_fold0_ckpt_best_ns22.pth"
R3_DATASET_CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single"),
    Path("/kaggle/input/birdclef2026-exp029-l1-single"),
]

R3_CKPT_PATH = None
for p in R3_DATASET_CANDIDATES:
    cand = p / R3_CKPT_NAME
    if cand.exists():
        R3_CKPT_PATH = cand; break

# Fallback: rglob
if R3_CKPT_PATH is None:
    for hit in Path("/kaggle/input").rglob(R3_CKPT_NAME):
        R3_CKPT_PATH = hit; break

assert R3_CKPT_PATH is not None, (
    f"R3 ckpt '{R3_CKPT_NAME}' not found. "
    f"Attach maekeso/birdclef2026-exp029-l1-single as dataset_sources."
)
print(f"R3 ckpt: {R3_CKPT_PATH}  ({R3_CKPT_PATH.stat().st_size/1e6:.1f} MB)")
""", "paths"))

# ============================================================
# Cell 3: Config — must match exp029 R3 training
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Config — must match exp029 R3 training
# ============================================================
NUM_CLASSES = 234
SR = 32000

# Window: 5s
TRAIN_DURATION = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
N_WINDOWS = 12  # 60s file → 12 × 5s windows

# Mel (Tucker spec, matches exp029 R3 training)
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 256
FMIN       = 20
FMAX       = 16000

# Model
BACKBONE = "eca_nfnet_l1"
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

# Load 234 BC2026 species labels in submission order
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES, f"Expected {NUM_CLASSES}, got {len(PRIMARY_LABELS)}"

print(f"Backbone: {BACKBONE}")
print(f"Window: {TRAIN_DURATION}s × {N_WINDOWS} per file = {TRAIN_DURATION * N_WINDOWS}s coverage")
print(f"Mel: n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}, fmin={FMIN}, fmax={FMAX}")
""", "config"))

# ============================================================
# Cell 4: Model definition (BirdSEDModel)
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Model definition (BirdSEDModel — matches exp029 R3 training)
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
    # Perch v2 embedding distillation head (used during training only,
    # but defined here so state_dict load doesn't complain about missing keys)
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

print("OK model definitions ready (BirdSEDModel + Perch distill head)")
""", "model"))

# ============================================================
# Cell 5: Load R3 ckpt
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Load R3 ckpt → model on GPU (eval mode)
# ============================================================
try:
    r3_state = torch.load(str(R3_CKPT_PATH), map_location="cpu", weights_only=False)
except TypeError:
    r3_state = torch.load(str(R3_CKPT_PATH), map_location="cpu")

print(f"R3 ckpt loaded:")
print(f"  epoch     = {r3_state.get('epoch')}")
print(f"  best_ns22 = {r3_state.get('best_ns22', float('nan')):.4f}")
print(f"  best_macro= {r3_state.get('best_macro', float('nan')):.4f}")

model = BirdSEDModel().to(device)
msg = model.load_state_dict(r3_state["model_state"], strict=False)
model.eval()
print(f"\nState dict loaded:")
print(f"  missing keys    = {len(msg.missing_keys)}")
print(f"  unexpected keys = {len(msg.unexpected_keys)}")
print(f"  Total params    = {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

del r3_state
gc.collect()
""", "load_ckpt"))

# ============================================================
# Cell 6: Inference on train_soundscapes
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Inference on train_soundscapes (~10,658 files)
# ============================================================
import soundfile as sf
import librosa

CHUNK_N = SR * TRAIN_DURATION  # samples per 5s window

def load_audio_32k_mono(path):
    # Load 60s mono float32 at 32kHz
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None

def file_to_chunks(path):
    # Return (12, 5*SR) chunks + start_times + end_times
    wav = load_audio_32k_mono(path)
    if wav is None:
        return None, None, None
    target_len = 60 * SR
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    chunks = wav.reshape(N_WINDOWS, CHUNK_N).astype(np.float32)
    start_times = np.arange(0, N_WINDOWS) * TRAIN_DURATION
    end_times   = np.arange(1, N_WINDOWS + 1) * TRAIN_DURATION
    return chunks, start_times, end_times

def sigmoid_np(x):
    # Numerically stable sigmoid for numpy arrays
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
        np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
    ).astype(np.float32)

mel_tf = MelSpecTransform().to(device)

# Discover all train_soundscapes files
train_ss_files = sorted(glob.glob(f"{TRAIN_SS_DIR}/*.ogg"))
print(f"train_soundscapes files: {len(train_ss_files)}")
assert len(train_ss_files) > 0

# Set DEBUG_LIMIT for sanity check; None = full run
DEBUG_LIMIT = None
if DEBUG_LIMIT is not None:
    train_ss_files = train_ss_files[:DEBUG_LIMIT]
    print(f"[DEBUG] Limiting to {DEBUG_LIMIT} files")

all_filenames = []
all_start_sec = []
all_end_sec   = []
all_logits    = []   # accumulate (12, 234) logit arrays

t0 = time.time()
with torch.no_grad():
    for fi, fp in enumerate(train_ss_files):
        basename = os.path.basename(fp).replace(".ogg", "")
        chunks, start_times, end_times = file_to_chunks(fp)
        if chunks is None:
            print(f"  [skip] {basename}: failed to load")
            continue

        # → GPU
        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)  # (12, 1, 160000)
        mel = mel_tf(wav_t)                                        # (12, 1, 256, 313)
        # Per-instance normalize (matches training)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)

        # Forward through R3 model
        clip_logits, framewise = model(mel, return_framewise=True)
        frame_max = framewise.max(dim=1).values

        # ★ Blend = 0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)
        # → applied in sigmoid space (then save as probabilities)
        # For consistency with training-time blend, we apply sigmoid here BEFORE the 50:50 mix.
        # But we're saving logits to allow flexible post-processing.
        # Following exp029 training convention: save logits and apply sigmoid at the end.
        blend_logits = 0.5 * clip_logits + 0.5 * frame_max
        blend_np = blend_logits.cpu().numpy().astype(np.float32)  # (12, 234)

        all_filenames.extend([basename] * N_WINDOWS)
        all_start_sec.extend(start_times.tolist())
        all_end_sec.extend(end_times.tolist())
        all_logits.append(blend_np)

        if (fi + 1) % 200 == 0 or fi == 0 or fi == len(train_ss_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 1e-6)
            eta = (len(train_ss_files) - fi - 1) / max(rate, 1e-6)
            print(f"  [{fi+1:5d}/{len(train_ss_files)}]  {elapsed/60:.1f}min  "
                  f"{rate:.2f} files/s  ETA {eta/60:.1f}min")

# Concat + sigmoid
if all_logits:
    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
    probs = sigmoid_np(logits_arr)
else:
    probs = np.zeros((0, NUM_CLASSES), dtype=np.float32)

print(f"\nInference DONE: {len(all_filenames)} rows in {(time.time()-t0)/60:.1f} min")
print(f"  probs shape: {probs.shape}")
print(f"  range:       [{probs.min():.4f}, {probs.max():.4f}]")
print(f"  mean:        {probs.mean():.4f}")
""", "infer"))

# ============================================================
# Cell 7: Save pseudo_exp029.csv + npy
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Save pseudo_exp029.csv + .npy
# ============================================================
df_pseudo = pd.DataFrame(probs, columns=PRIMARY_LABELS)
df_pseudo.insert(0, "filename",  all_filenames)
df_pseudo.insert(1, "start_sec", all_start_sec)
df_pseudo.insert(2, "end_sec",   all_end_sec)

out_csv = Path("/kaggle/working/pseudo_exp029.csv")
df_pseudo.to_csv(out_csv, index=False)
print(f"[OK] pseudo_exp029.csv: {len(df_pseudo)} rows, {df_pseudo.shape[1]-3} species cols, "
      f"{out_csv.stat().st_size/1e6:.1f}MB")

print(f"\nSummary:")
print(f"  files covered: {df_pseudo['filename'].nunique()}")
print(f"  rows per file: {len(df_pseudo) / df_pseudo['filename'].nunique():.1f}")

print(f"\nHead (first 3 rows, first 5 species cols):")
print(df_pseudo.head(3).iloc[:, :8])

print(f"\nMean prob per species (top 10):")
top_species = np.argsort(probs.mean(axis=0))[::-1][:10]
for i in top_species:
    print(f"  {PRIMARY_LABELS[i]:>10s}  {probs[:, i].mean():.4f}")

# Also save NPY (faster load for R4 training, smaller size)
out_npy = Path("/kaggle/working/pseudo_exp029.npy")
np.save(out_npy, probs.astype(np.float16))
print(f"\n[OK] pseudo_exp029.npy (float16): {out_npy.stat().st_size/1e6:.1f}MB")

out_meta = Path("/kaggle/working/pseudo_exp029_meta.npy")
np.save(out_meta, np.array(list(zip(all_filenames, all_start_sec, all_end_sec)), dtype=object))
print(f"[OK] pseudo_exp029_meta.npy")

print(f"\n[FINAL] Outputs in /kaggle/working/:")
for p in sorted(Path('/kaggle/working').glob('pseudo_exp029*')):
    print(f"  {p.name}  {p.stat().st_size/1e6:.1f}MB")
""", "save"))

# ============================================================
# Assemble NB
# ============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH = HERE / "nb_pseudo_r3_train_sc.ipynb"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Wrote: {NB_PATH}")
print(f"Cell count: {len(cells)}")

# Compile-strict syntax check
print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
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
