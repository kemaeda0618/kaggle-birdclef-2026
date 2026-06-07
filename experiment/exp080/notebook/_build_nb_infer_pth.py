"""Build exp080 inference NB (PyTorch .pth on CPU).

ONNX export 失敗のため .pth 直 load で推論。
CPU 90 min limit 内に完走させるため eff_b0 batch 12 (per file) で順次処理。
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_infer_pth.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md_cell("""# exp080 inference (PyTorch CPU, .pth direct)

ONNX export 失敗のため PyTorch CPU で直接推論。
- Load `m_single_ckpt_best.pth` (Colab training output)
- 各 test_soundscape file: 60s wav → 12 × 5s chunks → forward → sigmoid
- 90 min limit 内に完走 (推定 ~60-70 min)

Note: val_focal_macro 0.955 で training 完了 (clean focal holdout)
"""))

cells.append(code_cell("""!pip install -q timm
import sys, os, time, json, gc
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

DEVICE = torch.device("cpu")  # Kaggle submission = CPU only
torch.set_num_threads(4)
print(f"Python: {sys.version[:50]}, torch: {torch.__version__}, timm: {timm.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
START = time.time()
"""))

cells.append(code_cell("""# CFG (must match training)
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12
N_CLASSES = 234
WINDOW_SAMPLES = SR * WINDOW_SEC
N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000
TOP_DB = 80

def find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None

DATA_PATH = find_dir([
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
])
assert DATA_PATH is not None
TEST_SC_DIR = Path(DATA_PATH) / "test_soundscapes"
SAMPLE_SUB = Path(DATA_PATH) / "sample_submission.csv"

# Load .pth from kernel_source or dataset
PTH_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp080-train-output",
    "/kaggle/input/datasets/maekeso/birdclef2026-exp080-train-output",
])
assert PTH_DIR is not None, "exp080 train output not mounted"
PTH_PATH = next(PTH_DIR.rglob("m_single_ckpt_best.pth"))
print(f"Loading: {PTH_PATH} ({PTH_PATH.stat().st_size/1e6:.1f} MB)")

OUT_DIR = Path("/kaggle/working")
"""))

cells.append(code_cell("""# Model architecture (must match training NB)
def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class AttHead(nn.Module):
    def __init__(self, in_chans, p=0.5, num_class=234, hidden_dim=512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2), nn.Linear(in_chans, hidden_dim), nn.ReLU(), nn.Dropout(p),
        )
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)
    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        return {"framewise_logit": self.fix_scale(feat)}


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
        norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
        return (Xstd - norm_min) / (norm_max - norm_min + self.eps)


class SpecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(sample_rate=SR, normalized=True, n_fft=N_FFT,
                             hop_length=HOP_LENGTH, win_length=N_FFT,
                             f_max=F_MAX, n_mels=N_MELS, f_min=F_MIN),
            T.AmplitudeToDB(top_db=TOP_DB),
        )
        self.norm = NormalizeMelSpec()
    def forward(self, x):
        return self.norm(self.feature_extractor(x))


class CLEFClassifierSED(nn.Module):
    def __init__(self, num_classes=N_CLASSES, drop_path_rate=0.0):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)

    def forward(self, wav):
        spec = self.mel_spectr_generator(wav)
        spec3 = torch.stack([spec, spec, spec], 1)
        feat = self.backbone(spec3)[-1]
        framewise_logit = self.head(feat)["framewise_logit"]
        return framewise_logit.max(dim=-1).values  # clip_logit
"""))

cells.append(code_cell("""# Load model + ckpt
ckpt = torch.load(str(PTH_PATH), weights_only=False, map_location="cpu")
print(f"  ckpt keys: {list(ckpt.keys())}")
print(f"  trained at epoch: {ckpt.get('epoch', '?')}")
print(f"  val_focal_macro: {ckpt.get('val_focal_macro', '?'):.4f}")

model = CLEFClassifierSED(num_classes=N_CLASSES, drop_path_rate=0.0).to(DEVICE)
msg = model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()
print(f"  Loaded model state (strict=True)")
print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Load primary_labels from sample_submission
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES

def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)
"""))

cells.append(code_cell("""# Inference loop
test_files = sorted(TEST_SC_DIR.glob("*.ogg"))
print(f"Test files: {len(test_files)}")

rows = []
t0 = time.time()
with torch.no_grad():
    for fi, fp in enumerate(tqdm.tqdm(test_files, desc="Infer")):
        try:
            y, _ = librosa.load(str(fp), sr=SR, mono=True)
        except Exception as e:
            print(f"  load fail {fp.name}: {e}")
            y = np.zeros(SR * 60, dtype=np.float32)

        target = SR * 60
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]

        chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        for ci in range(N_WINDOWS):
            m = np.abs(chunks[ci]).max()
            if m > 0:
                chunks[ci] = chunks[ci] / m

        wav_tensor = torch.from_numpy(chunks)  # (12, 160000)
        clip_logit = model(wav_tensor)
        probs = sigmoid_np(clip_logit.cpu().numpy())

        file_stem = fp.stem
        for ci in range(N_WINDOWS):
            end_sec = (ci + 1) * 5
            row_id = f"{file_stem}_{end_sec}"
            rows.append([row_id] + probs[ci].tolist())

        if (fi + 1) % 50 == 0 or fi == len(test_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 0.001)
            eta = (len(test_files) - fi - 1) / max(rate, 0.001) / 60
            print(f"  [{fi+1}/{len(test_files)}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"\\nInference DONE: {(time.time()-t0)/60:.1f} min, {len(rows)} rows")
"""))

cells.append(code_cell("""# Build submission
sub_df = pd.DataFrame(rows, columns=["row_id"] + PRIMARY_LABELS)
print(f"sub_df: {sub_df.shape}")

assert sub_df["row_id"].nunique() == len(sub_df), "Duplicate row_id"
print(f"  mean prob: {sub_df[PRIMARY_LABELS].mean().mean():.5f}")
print(f"  max prob:  {sub_df[PRIMARY_LABELS].max().max():.4f}")
print(f"  NaN: {sub_df[PRIMARY_LABELS].isna().sum().sum()}")

if len(test_files) > 0 and len(sample_sub) > 0:
    sub_df = sub_df.set_index("row_id").reindex(sample_sub["row_id"]).reset_index()
    sub_df[PRIMARY_LABELS] = sub_df[PRIMARY_LABELS].fillna(0.0)

sub_df.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"\\nSaved: {OUT_DIR / 'submission.csv'} ({(OUT_DIR / 'submission.csv').stat().st_size/1e6:.1f} MB)")
print(f"Total time: {(time.time()-START)/60:.1f} min")
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
print(f"Built: {OUT_PATH}")
print(f"  cells: {len(cells)}")
