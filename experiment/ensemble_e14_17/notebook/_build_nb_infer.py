"""Build exp014-017 R2 ensemble inference NB.

4 model ensemble (Perch distill SED):
- exp014: HGNetV2-B0 (LB 0.907)
- exp015: convnext_pico (LB 0.910)
- exp016: regnety_008 (LB 0.903)
- exp017: eca_nfnet_l0 (LB 0.921)

All use Tucker mel spec (256/2048/512) + 5s window + same AttHead architecture.

Inference:
- Sliding 5s window step 5s = 12 chunks/file (non-overlapping for 60s)
- Compute mel ONCE per file (shared across 4 models)
- Equal weight average + smoothing
- PyTorch CPU (Kaggle 90 min limit、~60-80 min 想定)
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "nb_infer_ensemble.ipynb"


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md_cell("""# exp014-017 R2 ensemble inference

**4 Perch distill SED models (5s window、Tucker mel)**:

| Exp | Backbone | LB single |
|---|---|---|
| exp014 | HGNetV2-B0 | 0.907 |
| exp015 | convnext_pico | 0.910 |
| exp016 | regnety_008 | 0.903 |
| exp017 | eca_nfnet_l0 | **0.921** ★ |

**Ensemble**: equal weight average + Gaussian smoothing
**Expected LB**: 0.92-0.94 (single LB の average + diversity bonus)

**Sub source**:
- exp014: kernel `maekeso/birdclef2026-exp014-train-r2`
- exp015: dataset `maekeso/birdclef2026-exp015-weights`
- exp016: dataset `maekeso/birdclef2026-exp016-weights`
- exp017: dataset `maekeso/birdclef2026-exp017-weights`
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
from scipy.ndimage import gaussian_filter1d

DEVICE = torch.device("cpu")
torch.set_num_threads(4)
print(f"Python: {sys.version[:50]}, torch: {torch.__version__}, timm: {timm.__version__}")
START = time.time()
"""))

cells.append(code_cell("""# CFG (Tucker spec、5s)
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS_OUT = 12
N_CLASSES = 234
WINDOW_SAMPLES = SR * WINDOW_SEC

N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000

GAUSSIAN_SIGMA = 0.65

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

# === Locate 4 ckpts ===
def find_ckpt(roots, candidates):
    for root in roots:
        if root is None or not root.exists(): continue
        for cand in candidates:
            for fp in root.rglob(cand):
                return fp
    return None

# exp014: kernel output (HGNet R2)
EXP014_ROOT = find_dir([
    "/kaggle/input/birdclef2026-exp014-train-r2",
    "/kaggle/input/notebooks/maekeso/birdclef2026-exp014-train-r2",
])
EXP014_PTH = find_ckpt([EXP014_ROOT], [
    "ckpt_best_ns22.pth", "ckpt_best_macro.pth",
    "ckpt_ep04.pth",   # R2 真 peak (memory)
    "ckpt_ep05.pth", "ckpt_ep03.pth", "ckpt_ep06.pth",
])

# exp015: dataset (convnext_pico R2)
EXP015_ROOT = find_dir([
    "/kaggle/input/birdclef2026-exp015-weights",
    "/kaggle/input/datasets/maekeso/birdclef2026-exp015-weights",
])
EXP015_PTH = find_ckpt([EXP015_ROOT], [
    "r2_ckpt_best_ns22.pth", "r2_ckpt_best_macro.pth",
    "ckpt_best_ns22.pth", "ckpt_best_macro.pth",
])

# exp016: dataset (regnety_008 R2)
EXP016_ROOT = find_dir([
    "/kaggle/input/birdclef2026-exp016-weights",
    "/kaggle/input/datasets/maekeso/birdclef2026-exp016-weights",
])
EXP016_PTH = find_ckpt([EXP016_ROOT], [
    "r2_ckpt_best_ns22.pth", "r2_ckpt_best_macro.pth",
])

# exp017: dataset (eca_nfnet_l0 R2)
EXP017_ROOT = find_dir([
    "/kaggle/input/birdclef2026-exp017-weights",
    "/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights",
])
EXP017_PTH = find_ckpt([EXP017_ROOT], [
    "r2_ckpt_best_ns22.pth", "r2_ckpt_best_macro.pth",
])

print(f"exp014: {EXP014_PTH}")
print(f"exp015: {EXP015_PTH}")
print(f"exp016: {EXP016_PTH}")
print(f"exp017: {EXP017_PTH}")
assert EXP014_PTH and EXP015_PTH and EXP016_PTH and EXP017_PTH, "Missing ckpt"

OUT_DIR_WORK = Path("/kaggle/working")
"""))

cells.append(code_cell("""# Model architecture (shared across 4 backbones)
class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                             (x.size(-2), 1)).pow(1.0 / self.p)


class AttHead(nn.Module):
    def __init__(self, in_chans, num_class=234, hidden_dim=512, p=0.5):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        time_att_logit = torch.tanh(self.attention(feat))
        framewise_logit = self.fix_scale(feat)
        framewise_prob = torch.sigmoid(framewise_logit)
        clipwise_prob = torch.sum(
            framewise_prob * torch.softmax(time_att_logit, dim=-1),
            dim=-1,
        )
        clipwise_logit = torch.log(clipwise_prob / (1 - clipwise_prob).clamp(min=1e-6))
        return {
            "framewise_logit": framewise_logit,
            "clipwise_logit": clipwise_logit,
            "clipwise_prob": clipwise_prob,
        }


class CLEFClassifierSED(nn.Module):
    def __init__(self, backbone_name, num_classes=N_CLASSES, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.distill_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, 1536),
        )
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)

    def forward(self, mel, return_framewise=False):
        spec3 = torch.cat([mel, mel, mel], dim=1) if mel.size(1) == 1 else mel
        feat = self.backbone(spec3)[-1]
        head_out = self.head(feat)
        clip_logit = head_out["clipwise_logit"]
        framewise = head_out["framewise_logit"]
        if return_framewise:
            return clip_logit, framewise
        return clip_logit


def load_model(ckpt_path, backbone_name):
    model = CLEFClassifierSED(backbone_name=backbone_name).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  {backbone_name}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} val_ns22={ckpt.get('best_ns22', ckpt.get('val_ns22', '?'))}")
    return model
"""))

cells.append(code_cell("""# Load 4 models
print("Loading 4 models...")
m_exp014 = load_model(EXP014_PTH, "hgnetv2_b0.ssld_stage2_ft_in1k")
m_exp015 = load_model(EXP015_PTH, "convnext_pico.d1_in1k")
m_exp016 = load_model(EXP016_PTH, "regnety_008")
m_exp017 = load_model(EXP017_PTH, "eca_nfnet_l0")

MODELS = [m_exp014, m_exp015, m_exp016, m_exp017]
MODEL_NAMES = ["exp014_HGNet", "exp015_convnext_pico", "exp016_regnety_008", "exp017_eca_nfnet_l0"]

# Equal weights (can tune later)
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
print(f"\\nEnsemble: {len(MODELS)} models, weights={WEIGHTS}")

# Mel transform (Tucker spec)
mel_transform = T.MelSpectrogram(
    sample_rate=SR, normalized=True, n_fft=N_FFT,
    hop_length=HOP_LENGTH, win_length=N_FFT,
    f_max=F_MAX, n_mels=N_MELS, f_min=F_MIN,
).to(DEVICE)
amp_to_db = T.AmplitudeToDB(top_db=80).to(DEVICE)

def compute_mel(wav_tensor):
    mel = mel_transform(wav_tensor)
    mel = amp_to_db(mel)
    B = mel.size(0)
    for i in range(B):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    return mel

sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES
"""))

cells.append(code_cell("""# Inference function: ensemble 4 models
def infer_file_ensemble(audio_60s):
    \"\"\"60s audio → 12 chunks × 234 sp probability via 4-model ensemble.\"\"\"
    # Reshape 60s into 12 × 5s chunks
    chunks = audio_60s[:SR * 60].reshape(N_WINDOWS_OUT, WINDOW_SAMPLES).astype(np.float32)
    # Normalize per chunk
    for ci in range(N_WINDOWS_OUT):
        m = np.abs(chunks[ci]).max()
        if m > 0: chunks[ci] = chunks[ci] / m

    wav_t = torch.from_numpy(chunks).unsqueeze(1).to(DEVICE)  # (12, 1, samples)
    mel = compute_mel(wav_t)  # (12, 1, n_mels, n_frames)

    # Forward through 4 models
    probs_sum = np.zeros((N_WINDOWS_OUT, N_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for w, m in zip(WEIGHTS, MODELS):
            clip_logit, framewise = m(mel, return_framewise=True)
            frame_max = framewise.max(dim=-1).values
            p_clip = torch.sigmoid(clip_logit).float().cpu().numpy()
            p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
            probs = 0.5 * p_clip + 0.5 * p_fmax
            probs_sum += w * probs

    # Smoothing across 12 windows
    probs_smoothed = gaussian_filter1d(probs_sum, sigma=GAUSSIAN_SIGMA, axis=0,
                                       mode="nearest").astype(np.float32)
    return probs_smoothed
"""))

cells.append(code_cell("""# Run inference on test_soundscapes
test_files = sorted(TEST_SC_DIR.glob("*.ogg"))
print(f"Test files: {len(test_files)}")

rows = []
t0 = time.time()
for fi, fp in enumerate(tqdm.tqdm(test_files, desc="Infer ensemble")):
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

    probs = infer_file_ensemble(y)

    file_stem = fp.stem
    for k in range(N_WINDOWS_OUT):
        end_sec = (k + 1) * 5
        row_id = f"{file_stem}_{end_sec}"
        rows.append([row_id] + probs[k].tolist())

    if (fi + 1) % 50 == 0 or fi == len(test_files) - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / max(elapsed, 0.001)
        eta = (len(test_files) - fi - 1) / max(rate, 0.001) / 60
        print(f"  [{fi+1}/{len(test_files)}] {elapsed:.0f}s rate={rate:.2f}f/s eta={eta:.1f}min")

print(f"\\nInference DONE: {(time.time()-t0)/60:.1f} min, {len(rows)} rows")

# Build submission
sub_df = pd.DataFrame(rows, columns=["row_id"] + PRIMARY_LABELS)
assert sub_df["row_id"].nunique() == len(sub_df), "Duplicate row_id"
print(f"sub_df: {sub_df.shape}, mean={sub_df[PRIMARY_LABELS].mean().mean():.5f}, max={sub_df[PRIMARY_LABELS].max().max():.4f}")

if len(test_files) > 0 and len(sample_sub) > 0:
    sub_df = sub_df.set_index("row_id").reindex(sample_sub["row_id"]).reset_index()
    sub_df[PRIMARY_LABELS] = sub_df[PRIMARY_LABELS].fillna(0.0)

sub_df.to_csv(OUT_DIR_WORK / "submission.csv", index=False)
print(f"\\nSaved: {OUT_DIR_WORK / 'submission.csv'} ({(OUT_DIR_WORK / 'submission.csv').stat().st_size/1e6:.1f} MB)")
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
