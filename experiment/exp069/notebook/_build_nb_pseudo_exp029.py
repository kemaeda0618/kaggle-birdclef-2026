"""Build nb_pseudo_exp029.ipynb - Streaming exp029 R3 (eca_nfnet_l1) pseudo gen.

Streaming: load file → mel → exp029 forward → save → discard. CPU mode.
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_pseudo_exp029.ipynb")

cells = []


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


# ─────────── Cell 0: Title ───────────
cells.append(md_cell("""# exp069b-exp029: exp029 R3 (eca_nfnet_l1) pseudo gen (streaming, CPU)

**Pattern B-CPU parallel stream NB**: exp029 R3 model のみ streaming で full 10,658 files カバー。

**Output**: `/kaggle/working/exp029_raw_234.npz` (10658, 12, 234) float16

**並行 NB**:
  - nb_pseudo_nb4 (NB4 v11)
  - nb_pseudo_tucker (Tucker SED 5-fold)

**後段**: exp069c で 3 NPZ blend
"""))


# ─────────── Cell 1: Setup ───────────
cells.append(code_cell("""# Setup
import sys, os, time, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import timm
import tqdm.auto as tqdm
from scipy.ndimage import gaussian_filter1d

DEVICE = torch.device("cpu")
print(f"DEVICE: {DEVICE}")
print(f"torch: {torch.__version__}")
print(f"timm: {timm.__version__}")
START = time.time()
"""))


# ─────────── Cell 2: CFG ───────────
cells.append(code_cell("""# CFG
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12
N_CLASSES = 234
WINDOW_SAMPLES = SR * WINDOW_SEC

# exp029 R3 mel params
E17_N_MELS = 256
E17_N_FFT = 2048
E17_HOP = 512
E17_FMIN = 20
E17_FMAX = 16000
E17_BACKBONE = "eca_nfnet_l1"
E17_USE_DISTILL = True
E17_PERCH_DIM = 1536
E17_TRAIN_SAMPLES = SR * WINDOW_SEC
GAUSS_SIGMA = 0.65

# Paths
_data_path_candidates = [
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
]
DATA_PATH = None
for _p in _data_path_candidates:
    if Path(_p).exists():
        DATA_PATH = _p; break
assert DATA_PATH is not None
TRAIN_SC_DIR = Path(DATA_PATH) / "train_soundscapes"

# exp029 ckpt
E17_STATE_DIR = None
for _p in ["/kaggle/input/birdclef2026-exp029-l1-single", "/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single"]:
    if Path(_p).exists():
        E17_STATE_DIR = Path(_p); break
assert E17_STATE_DIR is not None, "exp029 ckpt not found"

OUT_DIR = Path("/kaggle/working")
test_files = sorted(TRAIN_SC_DIR.glob("*.ogg"))
print(f"TRAIN_SC: {TRAIN_SC_DIR}, files: {len(test_files)}")
print(f"E17_STATE_DIR: {E17_STATE_DIR}")
"""))


# ─────────── Cell 3: Model architecture ───────────
cells.append(code_cell("""# exp029 R3 architecture (copy of _E17SED from exp048 NB)
class _E17MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=E17_N_FFT, hop_length=E17_HOP,
            n_mels=E17_N_MELS, f_min=E17_FMIN, f_max=E17_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _E17GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E17DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _E17SED(nn.Module):
    def __init__(self, backbone_name=E17_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E17_TRAIN_SAMPLES // E17_HOP + 1
            dummy = torch.randn(1, 1, E17_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E17GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E17_USE_DISTILL:
            self.distill_head = _E17DistillHead(self.backbone_dim, E17_PERCH_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if E17_USE_DISTILL else h
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
"""))


# ─────────── Cell 4: Load weights ───────────
cells.append(code_cell("""# Load exp029 R3 ckpt
E17_CKPT = None
for _name in [
    "r3_fold0_ckpt_best_ns22.pth",
    "r3_fold0_ckpt_best_macro.pth",
    "r3_fold0_ckpt_latest.pth",
    "ckpt_best_ns22.pth",
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]:
    _hits = list(E17_STATE_DIR.rglob(_name))
    if _hits:
        E17_CKPT = _hits[0]; break
assert E17_CKPT is not None, f"No ckpt under {E17_STATE_DIR}"
print(f"ckpt: {E17_CKPT.name} ({E17_CKPT.stat().st_size/1e6:.1f}MB)")

try:
    _state = torch.load(str(E17_CKPT), map_location="cpu", weights_only=False)
except TypeError:
    _state = torch.load(str(E17_CKPT), map_location="cpu")
print(f"epoch={_state.get('epoch')}, best_ns22={_state.get('best_ns22', float('nan')):.4f}")

e17_model = _E17SED().to(DEVICE)
e17_model.load_state_dict(_state["model_state"], strict=False)
e17_model.eval()
e17_mel_tf = _E17MelTF().to(DEVICE)
print(f"e17 loaded: {sum(p.numel() for p in e17_model.parameters())/1e6:.1f}M params ({time.time()-START:.0f}s)")
"""))


# ─────────── Cell 5: Streaming inference ───────────
cells.append(code_cell("""# Streaming inference loop
import librosa

def load_one_60s(fp, sr=SR):
    y, _ = librosa.load(str(fp), sr=sr, mono=True)
    target_len = sr * 60
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


N_FILES = len(test_files)
probs_e17 = np.zeros((N_FILES, N_WINDOWS, N_CLASSES), dtype=np.float16)
file_ids = []

t0 = time.time()
with torch.no_grad():
    for fi, fp in enumerate(tqdm.tqdm(test_files, desc="exp029 stream")):
        y = load_one_60s(fp)
        chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(DEVICE)   # (12, 1, 160000)
        mel = e17_mel_tf(wav_t)
        # per-instance standardize
        for _i in range(mel.size(0)):
            mel[_i] = (mel[_i] - mel[_i].mean()) / (mel[_i].std() + 1e-6)
        clip, frame = e17_model(mel, return_framewise=True)
        frame_max = frame.max(dim=1).values
        p_clip = torch.sigmoid(clip).cpu().numpy().astype(np.float32)
        p_frame = torch.sigmoid(frame_max).cpu().numpy().astype(np.float32)
        p_mean = 0.5 * p_clip + 0.5 * p_frame
        p_smooth = gaussian_filter1d(p_mean, sigma=GAUSS_SIGMA, axis=0, mode="nearest").astype(np.float32)
        probs_e17[fi] = p_smooth.astype(np.float16)
        file_ids.append(fp.stem)

        del y, chunks, wav_t, mel, clip, frame, frame_max, p_clip, p_frame, p_mean, p_smooth
        if (fi + 1) % 200 == 0 or fi == N_FILES - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (N_FILES - fi - 1) / rate / 60
            print(f"  [{fi+1}/{N_FILES}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"exp029 inference DONE in {(time.time()-t0)/60:.1f} min")
"""))


# ─────────── Cell 6: Load labels ───────────
cells.append(code_cell("""sample_sub = pd.read_csv(Path(DATA_PATH) / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES
print(f"PRIMARY_LABELS: {len(PRIMARY_LABELS)} species")
"""))


# ─────────── Cell 7: Save NPZ ───────────
cells.append(code_cell("""np.savez_compressed(
    OUT_DIR / "exp029_raw_234.npz",
    probs=probs_e17,
    file_ids=np.array(file_ids),
)
print(f"Saved exp029_raw_234.npz: {(OUT_DIR / 'exp029_raw_234.npz').stat().st_size/1e6:.1f} MB")

with open(OUT_DIR / "primary_labels.json", "w") as f:
    json.dump(list(PRIMARY_LABELS), f, indent=2)
with open(OUT_DIR / "file_index.json", "w") as f:
    json.dump({fid: i for i, fid in enumerate(file_ids)}, f)
with open(OUT_DIR / "stream_meta.json", "w") as f:
    json.dump({"stream_id": "exp029", "n_files": len(file_ids), "shape": list(probs_e17.shape)}, f)

print(f"OK exp069b-exp029 DONE: total {(time.time()-START)/60:.1f} min")
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Built: {OUT_PATH} ({len(cells)} cells)")
