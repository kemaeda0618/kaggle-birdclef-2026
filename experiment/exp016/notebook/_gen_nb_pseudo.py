"""Generate exp016 NB: R1 → Pseudo labels on Colab Pro.

Pipeline:
- Load exp016 R1 regnety_008 SED student ckpt from Drive
  (`/content/drive/MyDrive/kaggle/birdclef2026/output/exp016/r1/ckpt_best_ns22.pth`)
- Run inference on all train_soundscapes (12 windows × 5s each, 10,658 files)
- Blend: 0.5*sigmoid(clip_logits) + 0.5*sigmoid(frame_max)
- Gaussian smooth across windows per file (sigma=0.65)
- Power Transform γ=1.2 (Babych H1 sharpening, training-time pseudo target)
- NO threshold filter (Natsume LB evidence: filter hurts)
- Save pseudo_labels.csv to Drive `output/exp016/r1-pseudo/pseudo_labels.csv`

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

cells.append(md_cell(r"""# exp016 — R1 → Pseudo (Colab Pro)

**Pipeline:** R1 student (regnety_008 SED) → pseudo on 10,658 train_soundscapes → Drive 保存

## 構成
- Backbone: regnety_008 (~6M、RegNet 系)
- Postproc: Gaussian (sigma=0.65) + Power Transform (γ=1.2)、フィルタなし

## 入出力
- 入力 ckpt: `/content/drive/MyDrive/kaggle/birdclef2026/output/exp016/r1/ckpt_best_ns22.pth`
- 出力 CSV: `/content/drive/MyDrive/kaggle/birdclef2026/output/exp016/r1-pseudo/pseudo_labels.csv`
- データ: `/content/drive/MyDrive/kaggle/birdclef2026/train_soundscapes/`
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — Drive mount, pip install
# ============================================================
!pip install -q timm onnxruntime-gpu librosa soundfile scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

DRIVE_INPUT_DIR     = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_CKPT_DIR      = DRIVE_INPUT_DIR / "output" / "exp016" / "r1"
DRIVE_PSEUDO_DIR    = DRIVE_INPUT_DIR / "output" / "exp016" / "r1-pseudo"
DRIVE_PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
assert DRIVE_INPUT_DIR.exists(), f"Drive input folder missing: {DRIVE_INPUT_DIR}"
assert DRIVE_CKPT_DIR.exists(), f"Drive ckpt dir missing (run train first): {DRIVE_CKPT_DIR}"
print(f"Drive input:  {DRIVE_INPUT_DIR}")
print(f"Drive ckpt:   {DRIVE_CKPT_DIR}")
print(f"Drive pseudo: {DRIVE_PSEUDO_DIR}")

LOCAL_DATA = Path("/content/data")
LOCAL_OUT  = Path("/content/output")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
""", "setup"))

# =============================================================================
# Cell 2: Data prep — Kaggle API direct DL (single zip, fast)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Data prep — Kaggle API direct DL for soundscapes + ckpt copy
# ============================================================
import time, zipfile, subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm.auto import tqdm

api = KaggleApi(); api.authenticate()
print("kaggle authenticated")

T0_total = time.time()

TS_DIR = LOCAL_DATA / "train_soundscapes"

need_dl = not TS_DIR.exists() or sum(1 for _ in TS_DIR.glob("*.ogg")) < 10000

if need_dl:
    print(f"\n[1/2] Downloading birdclef-2026 competition data via Kaggle API...")
    print(f"      (only soundscapes required; full zip is single-file fastest)")
    t0 = time.time()
    api.competition_download_files(
        "birdclef-2026",
        path=str(LOCAL_DATA),
        force=False, quiet=False,
    )
    print(f"  DL done in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    zips = list(LOCAL_DATA.glob("birdclef-2026*.zip"))
    assert zips, "No birdclef-2026 zip found after download"
    zip_path = zips[0]
    print(f"\n[2/2] Extracting (selective, soundscapes + CSVs)...")
    t_extract = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        wanted = [i for i in zf.infolist()
                  if i.filename.startswith("train_soundscapes/") or
                     i.filename.endswith(".csv") or
                     i.filename == "recording_location.txt"]
        total_bytes = sum(i.file_size for i in wanted)
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="extract", smoothing=0.05, mininterval=1.0)
        for info in wanted:
            zf.extract(info, LOCAL_DATA)
            pbar.update(info.file_size)
        pbar.close()
    print(f"  extracted {len(wanted)} entries in {time.time()-t_extract:.0f}s")
    zip_path.unlink()
    print(f"  removed zip")
else:
    print("train_soundscapes already present locally")

n_ts = sum(1 for _ in TS_DIR.glob("*.ogg")) if TS_DIR.exists() else 0
print(f"\n  train_soundscapes: {n_ts} .ogg files")

# Place CSVs under /content/data/competition
comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["sample_submission.csv", "taxonomy.csv"]:
    src_in_root = LOCAL_DATA / fn
    dst = comp / fn
    if src_in_root.exists() and not dst.exists():
        shutil.copy2(str(src_in_root), str(dst))
        print(f"  competition/{fn} OK")
    elif not src_in_root.exists():
        drive_src = DRIVE_INPUT_DIR / fn
        if drive_src.exists() and not dst.exists():
            shutil.copy2(str(drive_src), str(dst))
            print(f"  competition/{fn} OK (Drive fallback)")

# Copy R1 ckpt from Drive (small, 1-file)
LOCAL_CKPT_DIR = LOCAL_DATA / "r1-ckpt"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PRIORITY = ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth"]
ckpt_src = None
for name in CKPT_PRIORITY:
    p = DRIVE_CKPT_DIR / name
    if p.exists():
        ckpt_src = p; break
assert ckpt_src is not None, f"No ckpt found in {DRIVE_CKPT_DIR}"
LOCAL_CKPT = LOCAL_CKPT_DIR / ckpt_src.name
if not LOCAL_CKPT.exists():
    sz = ckpt_src.stat().st_size
    print(f"\nCopying ckpt from Drive: {ckpt_src.name} ({sz/1e6:.1f}MB)")
    t0 = time.time()
    with open(ckpt_src, "rb") as fin, open(LOCAL_CKPT, "wb") as fout:
        pbar = tqdm(total=sz, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="ckpt copy", mininterval=1.0)
        while True:
            buf = fin.read(4 * 1024 * 1024)
            if not buf: break
            fout.write(buf); pbar.update(len(buf))
        pbar.close()
    print(f"  done in {time.time()-t0:.1f}s")
else:
    print(f"\nckpt already local: {LOCAL_CKPT.name}")

print(f"\n=== Total prep time: {(time.time()-T0_total)/60:.1f} min ===")
""", "dl_data"))

# =============================================================================
# Cell 3: Imports + Config
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Imports + Config + Paths
# ============================================================
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

START = time.time()

# ===== Paths =====
TS_DIR = LOCAL_DATA / "train_soundscapes"
SAMPLE_SUB_PATH = LOCAL_DATA / "competition" / "sample_submission.csv"
CKPT_PATH = LOCAL_CKPT
assert TS_DIR.is_dir() and CKPT_PATH.exists()

# ===== Config (must match training) =====
NUM_CLASSES = 234
SR = 32000
TRAIN_DURATION = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 256
FMIN           = 20
FMAX           = 16000
BACKBONE = "regnety_008"
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

N_WINDOWS = 12
CHUNK_N   = TRAIN_SAMPLES

GAUSS_SIGMA  = 0.65
POWER_GAMMA  = 1.2

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"Backbone: {BACKBONE}")
print(f"Post: gauss sigma={GAUSS_SIGMA}, power gamma={POWER_GAMMA}")
print(f"NO threshold filter (Natsume LB evidence)")
""", "config"))

# =============================================================================
# Cell 4: Model defs (identical to training)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Model — rebuild regnety_008 SED architecture
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
# Cell 5: Load R1 ckpt
# ============================================================
print(f"Loading ckpt: {CKPT_PATH}")
try:
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(CKPT_PATH), map_location="cpu")

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
# Cell 6: Inference on all train_soundscapes
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Inference — iterate over all train_soundscapes
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


mel_tf = MelSpecTransform().to(device)

sc_files = sorted(glob.glob(str(TS_DIR / "*.ogg")))
print(f"train_soundscapes: {len(sc_files)} files")
assert len(sc_files) > 0

all_filenames = []
all_start_secs = []
all_end_secs = []
all_probs = []

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

        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)
        mel = mel_tf(wav_t)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        mel = mel.to(memory_format=torch.channels_last)

        with autocast():
            clip_logits, framewise = model(mel, return_framewise=True)
            frame_max = framewise.max(dim=1).values
            p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()
            p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
        probs_file = 0.5 * p_clip + 0.5 * p_fmax

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

prob_mat = np.concatenate(all_probs, axis=0).astype(np.float32)
filenames_arr = np.array(all_filenames)
start_secs_arr = np.array(all_start_secs, dtype=np.float32)
end_secs_arr   = np.array(all_end_secs,   dtype=np.float32)
print(f"\nInference: {prob_mat.shape}, mean={prob_mat.mean():.4f}, max={prob_mat.max():.4f} "
      f"in {time.time()-t0:.0f}s")
""", "inference"))

# =============================================================================
# Cell 7: Post-processing
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Post-processing — Power Transform γ=1.2
# ============================================================
print(f"Pre-PT: mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"95%ile={np.percentile(prob_mat, 95):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")

prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)

print(f"\nPost-PT (γ={POWER_GAMMA}): mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"95%ile={np.percentile(prob_mat, 95):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")

row_max = prob_mat.max(axis=1)
print(f"\nrow_max stats (diagnostic, NO filter applied):")
print(f"  50%ile: {np.percentile(row_max, 50):.4f}")
print(f"  99%ile: {np.percentile(row_max, 99):.4f}")
print(f"  >=0.05 rows: {(row_max >= 0.05).sum()}/{len(row_max)} "
      f"({(row_max >= 0.05).mean()*100:.1f}%)")
""", "postproc"))

# =============================================================================
# Cell 8: Save CSV and mirror to Drive
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: Save pseudo_labels.csv (local + Drive)
# ============================================================
df = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)
df.insert(0, "filename",  filenames_arr)
df.insert(1, "start_sec", start_secs_arr)
df.insert(2, "end_sec",   end_secs_arr)

local_csv = LOCAL_OUT / "pseudo_labels.csv"
df.to_csv(local_csv, index=False)
print(f"Local saved: {local_csv}")
print(f"  shape: {df.shape}")
print(f"  size:  {local_csv.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  files: {df['filename'].nunique()}")

print("\n=== Preview ===")
print(df.head(3))

# Mirror to Drive
drive_csv = DRIVE_PSEUDO_DIR / "pseudo_labels.csv"
print(f"\nMirroring to Drive: {drive_csv}")
t0 = time.time()
shutil.copy2(str(local_csv), str(drive_csv))
print(f"  done in {time.time()-t0:.1f}s")

# Save a meta JSON alongside
meta = {
    "backbone": BACKBONE,
    "ckpt": str(CKPT_PATH.name),
    "epoch": state.get("epoch"),
    "best_ns22": state.get("best_ns22"),
    "best_macro": state.get("best_macro"),
    "n_files": int(df["filename"].nunique()),
    "n_rows": int(len(df)),
    "gauss_sigma": GAUSS_SIGMA,
    "power_gamma": POWER_GAMMA,
}
(DRIVE_PSEUDO_DIR / "pseudo_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(f"  meta written: {DRIVE_PSEUDO_DIR}/pseudo_meta.json")

print(f"\nTotal time: {(time.time()-START)/60:.1f} min")
""", "save_csv"))

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
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb_pseudo.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
