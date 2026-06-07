"""Generate exp032 R1 pseudo regeneration NB (Colab Pro Blackwell / G4).

Purpose:
Drive 上の exp032 R1 pseudo CSV が R1 v2 mel-fix (bad、LB 0.786 ckpt) で
上書きされてる可能性があるため、Kaggle Dataset v1 (R1 v1、LB 0.876 ckpt)
から download して再 inference、Drive 上書き。

Output:
  /content/drive/MyDrive/kaggle/birdclef2026/output/exp032/r1-pseudo/pseudo_labels.csv

Run time 想定: 15-25 min on Colab G4 (10592 files × 12 windows × b3 forward)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr", r"""# exp032 R1 Pseudo 再生成 NB (Colab Pro)

## 目的
Drive 上の R1 pseudo CSV が **R1 v2 mel-fix ckpt (LB 0.786 bad)** で上書きされてる可能性。
Kaggle Dataset v1 (R1 v1、LB 0.876 good) を強制 download して **R1 v1 ckpt から fresh 再 inference**、
Drive R1 pseudo CSV を上書き。

## Run after this:
exp032/notebook/nb_train_r2.ipynb (R2 訓練) — 上書きされた R1 v1 pseudo を使う

## 構成
1. Drive mount + kaggle.json auth
2. Competition data DL (cached if 既存)
3. **Kaggle Dataset v1 から R1 ckpt DL (version_number=1 pin)**
4. BirdSEDModel (b3 backbone) 定義 + v1 ckpt load
5. Inference on train_soundscapes (10,592 files × 12 windows)
6. POWER_GAMMA=1.2 適用
7. pseudo_labels.csv を Drive へ上書き保存

Run time: 15-25 min (G4)
"""))

# Cell 1: Setup
cells.append(code_cell("setup", r"""# ============================================================
# Setup: Drive + kaggle.json
# ============================================================
get_ipython().system('pip install -q timm librosa soundfile scipy')

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess, gc
from pathlib import Path

DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_R1_PSEUDO_DIR = DRIVE_INPUT_DIR / "output" / "exp032" / "r1-pseudo"
DRIVE_R1_PSEUDO = DRIVE_R1_PSEUDO_DIR / "pseudo_labels.csv"
DRIVE_R1_PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
assert DRIVE_INPUT_DIR.exists(), f"Drive not mounted: {DRIVE_INPUT_DIR}"
print(f"Drive: {DRIVE_INPUT_DIR}")
print(f"R1 pseudo target: {DRIVE_R1_PSEUDO}")
if DRIVE_R1_PSEUDO.exists():
    _mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(DRIVE_R1_PSEUDO.stat().st_mtime))
    _size = DRIVE_R1_PSEUDO.stat().st_size / 1e6
    print(f"  existing R1 pseudo: {_size:.2f} MB, mtime={_mtime} (will be OVERWRITTEN)")
else:
    print(f"  no existing R1 pseudo CSV (will be CREATED)")

# kaggle.json
KJ_CANDIDATES = [
    DRIVE_INPUT_DIR / "kaggle.json",
    Path("/content/drive/MyDrive/kaggle.json"),
    Path("/content/kaggle.json"),
]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
assert KJ is not None, "kaggle.json not found"
KAGGLE_CFG = Path.home() / ".kaggle"
KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
_creds = json.loads(KJ.read_text())
if _creds.get("key", "").startswith("KGAT_"):
    os.environ["KAGGLE_API_TOKEN"] = _creds["key"]
print(f"kaggle.json: {KJ}")

LOCAL_DATA = Path("/content/data")
LOCAL_OUT = Path("/content/output")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
"""))

# Cell 2: Download competition data
cells.append(code_cell("dl_data", r"""# ============================================================
# Download competition data (cached if 既存)
# ============================================================
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm.auto import tqdm

api = KaggleApi(); api.authenticate()
TS_DIR = LOCAL_DATA / "train_soundscapes"
TAXO = LOCAL_DATA / "taxonomy.csv"
SAMPLE_SUB = LOCAL_DATA / "sample_submission.csv"

need_dl = not TAXO.exists() or not TS_DIR.exists() or sum(1 for _ in TS_DIR.glob("*.ogg")) < 10000

if need_dl:
    print("Downloading birdclef-2026 (~25 GB)...")
    t0 = time.time()
    api.competition_download_files("birdclef-2026", path=str(LOCAL_DATA), force=False, quiet=False)
    zips = list(LOCAL_DATA.glob("birdclef-2026*.zip"))
    assert zips
    zip_path = zips[0]
    print(f"  DL done in {(time.time()-t0)/60:.1f} min; extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        pbar = tqdm(total=total, unit="B", unit_scale=True)
        for info in infos:
            zf.extract(info, LOCAL_DATA)
            pbar.update(info.file_size)
        pbar.close()
    zip_path.unlink()
    print(f"  extracted in {(time.time()-t0)/60:.1f} min total")
else:
    print("Competition data already present")

n_ts = sum(1 for _ in TS_DIR.glob("*.ogg")) if TS_DIR.exists() else 0
print(f"  train_soundscapes: {n_ts} files")
assert n_ts > 0, "train_soundscapes empty"
"""))

# Cell 3: Download R1 v1 ckpt
cells.append(code_cell("dl_v1_ckpt", r"""# ============================================================
# Download R1 v1 ckpt from Kaggle Dataset (version_number=1 pinned)
# ============================================================
R1_CKPT_LOCAL_DIR = LOCAL_DATA / "exp032_r1_v1"
R1_CKPT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
R1_CKPT_LOCAL = R1_CKPT_LOCAL_DIR / "ckpt_best_ns22.pth"

if R1_CKPT_LOCAL.exists():
    print(f"R1 v1 ckpt already at: {R1_CKPT_LOCAL} ({R1_CKPT_LOCAL.stat().st_size/1e6:.1f} MB)")
else:
    print("Downloading maekeso/birdclef2026-exp032-weights version 1...")
    # Try SDK with version_number param
    try:
        api.dataset_download_files(
            "maekeso/birdclef2026-exp032-weights",
            path=str(R1_CKPT_LOCAL_DIR),
            unzip=True, quiet=False,
            version_number=1,
        )
        print("  SDK download with version_number=1 OK")
    except TypeError:
        # version_number not supported in this SDK version
        print("  SDK version_number unsupported, using direct URL download...")
        import requests
        from urllib.parse import quote
        _key = _creds["key"]
        _headers = {"Authorization": f"Bearer {_key}"} if _key.startswith("KGAT_") else {}
        _auth = (_creds["username"], _key) if not _key.startswith("KGAT_") else None
        url = "https://www.kaggle.com/api/v1/datasets/download/maekeso/birdclef2026-exp032-weights?datasetVersionNumber=1"
        _zip_path = R1_CKPT_LOCAL_DIR / "v1.zip"
        with requests.get(url, headers=_headers, auth=_auth, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        print(f"  downloaded {_zip_path.stat().st_size/1e6:.1f} MB")
        with zipfile.ZipFile(_zip_path) as zf:
            zf.extractall(R1_CKPT_LOCAL_DIR)
        _zip_path.unlink()

assert R1_CKPT_LOCAL.exists(), f"ckpt missing: {R1_CKPT_LOCAL}"

# Verify ckpt content
import torch
state = torch.load(str(R1_CKPT_LOCAL), map_location="cpu", weights_only=False)
print(f"\nckpt content:")
print(f"  epoch: {state.get('epoch')}")
print(f"  best_ns22: {state.get('best_ns22', float('nan')):.4f}")
print(f"  best_macro: {state.get('best_macro', float('nan')):.4f}")

_val_ns22 = state.get("best_ns22", float("nan"))
# v1 expected: ~0.9156 (LB 0.876)、v2 expected: ~0.9095 (LB 0.786)
if 0.910 < _val_ns22 < 0.920:
    print(f"\nOK: val_ns22={_val_ns22:.4f} matches R1 v1 (good)")
else:
    print(f"\nWARN: val_ns22={_val_ns22:.4f} unexpected, may not be v1")
    print(f"  v1 expected: ~0.9156 (LB 0.876)")
    print(f"  v2 expected: ~0.9095 (LB 0.786)")
"""))

# Cell 4: Config
cells.append(code_cell("config", r"""# ============================================================
# Config — match R1 v1 (BC2026 standard mel)
# ============================================================
NUM_CLASSES = 234
SR = 32000
TRAIN_DURATION = 5
TRAIN_SAMPLES = SR * TRAIN_DURATION
N_WINDOWS = 12

# Mel-spec — BC2026 standard (R1 v1 と同じ、LB 0.876 由来)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000

BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"
USE_PERCH_DISTILL = False   # R1 と同じ、Antoine 流

POWER_GAMMA = 1.2   # R1 と同じ post-PT
GAUSS_SIGMA = 0.65  # window-wise smoothing

import pandas as pd
sample_sub = pd.read_csv(LOCAL_DATA / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"BACKBONE: {BACKBONE}")
print(f"mel: N_FFT={N_FFT}, HOP={HOP_LENGTH}, N_MELS={N_MELS}, F=[{FMIN},{FMAX}]")
print(f"POWER_GAMMA={POWER_GAMMA}, GAUSS_SIGMA={GAUSS_SIGMA}")
"""))

# Cell 5: Model
cells.append(code_cell("model", r"""# ============================================================
# Model — BirdSEDModel (exp017 template、b3 backbone、no Perch distill)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


# Load v1 ckpt
model = BirdSEDModel().to(device)
try:
    state = torch.load(str(R1_CKPT_LOCAL), map_location=device, weights_only=False)
except TypeError:
    state = torch.load(str(R1_CKPT_LOCAL), map_location=device)
msg = model.load_state_dict(state["model_state"], strict=False)
print(f"loaded R1 v1 ckpt: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
print(f"  source val_ns22={state.get('best_ns22', float('nan')):.4f}")
model.eval()
model = model.to(memory_format=torch.channels_last)
mel_tf = MelSpecTransform().to(device)
print(f"OK model ready, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
"""))

# Cell 6: Inference loop
cells.append(code_cell("infer", r"""# ============================================================
# Inference on train_soundscapes — generate fresh R1 pseudo
# ============================================================
import soundfile as sf
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
import glob
from torch.amp import autocast

def load_audio_32k_mono(path, target_samples=60 * SR):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < target_samples:
        wav = np.pad(wav, (0, target_samples - len(wav)))
    elif len(wav) > target_samples:
        wav = wav[:target_samples]
    return wav.astype(np.float32)


def file_to_chunks(path):
    wav = load_audio_32k_mono(path, target_samples=N_WINDOWS * TRAIN_SAMPLES)
    return wav.reshape(N_WINDOWS, TRAIN_SAMPLES).astype(np.float32)


sc_files = sorted(glob.glob(str(TS_DIR / "*.ogg")))
print(f"train_soundscapes: {len(sc_files)} files")
assert len(sc_files) > 0

all_filenames, all_start_secs, all_end_secs, all_probs = [], [], [], []
t0 = time.time()

with torch.no_grad():
    for fi, fpath in enumerate(tqdm(sc_files, desc="infer", mininterval=2.0)):
        stem = Path(fpath).stem
        try:
            chunks = file_to_chunks(fpath)
        except Exception as e:
            print(f"WARN: {stem}: {e}")
            chunks = np.zeros((N_WINDOWS, TRAIN_SAMPLES), dtype=np.float32)

        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)
        mel = mel_tf(wav_t)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        mel = mel.to(memory_format=torch.channels_last)
        with autocast("cuda"):
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

prob_mat = np.concatenate(all_probs, axis=0).astype(np.float32)
filenames_arr = np.array(all_filenames)
start_secs_arr = np.array(all_start_secs, dtype=np.float32)
end_secs_arr = np.array(all_end_secs, dtype=np.float32)
print(f"\nInference: {prob_mat.shape}, mean={prob_mat.mean():.4f}, max={prob_mat.max():.4f}")
print(f"  Pre-PT: mean={prob_mat.mean():.6f}, 99%ile={np.percentile(prob_mat, 99):.4f}")

# Apply POWER_GAMMA (R1 と同じ post-PT)
prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)
print(f"  Post-PT (γ={POWER_GAMMA}): mean={prob_mat.mean():.6f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")
print(f"  Inference total: {(time.time()-t0)/60:.1f} min")
"""))

# Cell 7: Save pseudo CSV
cells.append(code_cell("save", r"""# ============================================================
# Save pseudo CSV — overwrite Drive R1 pseudo
# ============================================================
df = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)
df.insert(0, "filename", filenames_arr)
df.insert(1, "start_sec", start_secs_arr)
df.insert(2, "end_sec", end_secs_arr)

# Local copy
local_csv = LOCAL_OUT / "pseudo_labels.csv"
df.to_csv(local_csv, index=False)
print(f"local saved: {local_csv} ({local_csv.stat().st_size/1e6:.1f} MB)")

# Backup existing Drive R1 pseudo (if any)
if DRIVE_R1_PSEUDO.exists():
    _bak = DRIVE_R1_PSEUDO.with_suffix(".backup_before_v1_regen.csv")
    shutil.copy(DRIVE_R1_PSEUDO, _bak)
    print(f"backed up existing R1 pseudo to: {_bak.name}")

# Overwrite Drive R1 pseudo
shutil.copy(local_csv, DRIVE_R1_PSEUDO)
print(f"OVERWROTE Drive R1 pseudo: {DRIVE_R1_PSEUDO} ({DRIVE_R1_PSEUDO.stat().st_size/1e6:.1f} MB)")

print(f"\nDONE. Now you can run exp032 R2 train NB.")
print(f"  R1 pseudo (R1 v1 由来) ready at: {DRIVE_R1_PSEUDO}")
"""))

# Cell 8: Disconnect
cells.append(code_cell("disconnect", r"""# ============================================================
# Auto-disconnect Colab runtime (save credits)
# ============================================================
print("Done. Disconnecting runtime in 30 sec...")
time.sleep(30)
from google.colab import runtime
runtime.unassign()
"""))

# Assemble NB
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"name": "exp032_regen_r1_pseudo.ipynb"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_regen_r1_pseudo.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path.name}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
