"""Build exp081 inference NB (10s sliding window, Tucker mel, PyTorch CPU).

Design (Babych BC25 style):
- Sliding 10s window with step 5s (12 windows per 60s file)
- Pad audio: 2.5s leading + 60s + 2.5s trailing = 65s
- Full 65s mel computed ONCE, sliced into 12 chunks
- Smoothing kernel [0.1, 0.2, 0.4, 0.2, 0.1] across 12 windows
- Output: 12 rows per file (row_id _5, _10, ..., _60)

Model: eca_nfnet_l0 + AttHead (mel input)
Compute: CPU only (Kaggle 90min limit)
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_infer.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md_cell("""# exp081 inference: 10s sliding window (Tucker mel)

**Setup**:
- Window: 10s
- Mel: Tucker spec (n_mels=256, n_fft=2048, hop=512)
- Sliding step: 5s → 12 windows/file = 12 row outputs
- Pad audio: 2.5s + 60s + 2.5s = 65s
- Smoothing: [0.1, 0.2, 0.4, 0.2, 0.1] across 12 windows
- Inference: PyTorch CPU (eca_nfnet_l0、Kaggle 90min limit)

**Input**:
- Kaggle Dataset: `maekeso/birdclef2026-exp081-weights` (.pth from exp081 R2 training)
- Competition: `birdclef-2026/test_soundscapes/`

**Output**: `/kaggle/working/submission.csv`
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

DEVICE = torch.device("cpu")  # Kaggle submission = CPU only
torch.set_num_threads(4)
print(f"Python: {sys.version[:50]}, torch: {torch.__version__}, timm: {timm.__version__}")
START = time.time()
"""))

cells.append(code_cell("""# CFG (must match training)
SR = 32_000
WINDOW_SEC = 10
N_WINDOWS_OUT = 12   # output rows per file
STEP_SEC = 5         # sliding step
N_CLASSES = 234

# Mel (Tucker spec)
N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000

# Padded total length for 12 windows: (12-1)*5 + 10 = 65s
PADDED_SEC = (N_WINDOWS_OUT - 1) * STEP_SEC + WINDOW_SEC   # 65
PAD_LEAD_SEC = (PADDED_SEC - 60) / 2                       # 2.5s
PAD_TRAIL_SEC = PADDED_SEC - 60 - PAD_LEAD_SEC             # 2.5s
PADDED_SAMPLES = SR * PADDED_SEC
PAD_LEAD_SAMPLES = int(SR * PAD_LEAD_SEC)
PAD_TRAIL_SAMPLES = int(SR * PAD_TRAIL_SEC)
WINDOW_SAMPLES = SR * WINDOW_SEC
STEP_SAMPLES = SR * STEP_SEC

# Smoothing (Babych spec)
SMOOTHING_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
GAUSSIAN_SIGMA = 0.65

print(f"WINDOW: {WINDOW_SEC}s, STEP: {STEP_SEC}s")
print(f"PADDED: {PADDED_SEC}s ({PAD_LEAD_SEC}s lead + 60s + {PAD_TRAIL_SEC}s trail)")
print(f"N_WINDOWS_OUT: {N_WINDOWS_OUT}")

# Paths
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

PTH_DIR = find_dir([
    "/kaggle/input/birdclef2026-exp081-weights",
    "/kaggle/input/datasets/maekeso/birdclef2026-exp081-weights",
])
assert PTH_DIR is not None, "exp081 weights not mounted"

# Look for best ckpt (priority: ckpt_best_ns22.pth, then ckpt_best_macro, then ckpt_latest)
PTH_PATH = None
for cand_name in ["ckpt_best_ns22.pth", "r2_ckpt_best_ns22.pth", "ckpt_best_macro.pth", "r2_ckpt_best_macro.pth", "ckpt_latest.pth"]:
    for cand in PTH_DIR.rglob(cand_name):
        PTH_PATH = cand
        break
    if PTH_PATH is not None:
        break
assert PTH_PATH is not None, f"No ckpt found in {PTH_DIR}"
print(f"Loading: {PTH_PATH} ({PTH_PATH.stat().st_size/1e6:.1f} MB)")

OUT_DIR = Path("/kaggle/working")
"""))

cells.append(code_cell("""# Model architecture (must match training NB cell 5)
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
        # Babych style: pure framewise (attention 捨て、clip_max を後段で取る)
        # Standard SED: clipwise = sum(framewise * softmax(time_att))
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
    def __init__(self, backbone_name="eca_nfnet_l0", num_classes=N_CLASSES, drop_path_rate=0.0):
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

    def forward(self, mel, return_framewise=False, return_distill=False):
        spec3 = torch.cat([mel, mel, mel], dim=1) if mel.size(1) == 1 else mel
        feat = self.backbone(spec3)[-1]
        head_out = self.head(feat)
        clip_logit = head_out["clipwise_logit"]
        framewise = head_out["framewise_logit"]
        if return_framewise and return_distill:
            distill = self.distill_head(feat)
            return clip_logit, framewise, distill
        if return_framewise:
            return clip_logit, framewise
        if return_distill:
            distill = self.distill_head(feat)
            return clip_logit, distill
        return clip_logit


def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)
"""))

cells.append(code_cell("""# Load ckpt
ckpt = torch.load(str(PTH_PATH), weights_only=False, map_location="cpu")
print(f"  ckpt keys: {list(ckpt.keys())[:10]}")
print(f"  epoch: {ckpt.get('epoch', '?')}")
print(f"  val_ns22: {ckpt.get('val_ns22', ckpt.get('best_ns22', '?'))}")
print(f"  val_macro: {ckpt.get('val_macro', ckpt.get('best_macro', '?'))}")

model = CLEFClassifierSED().to(DEVICE)

state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
msg = model.load_state_dict(state, strict=False)
model.eval()
print(f"  Load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
"""))

cells.append(code_cell("""# Mel transform (must match training)
mel_transform = T.MelSpectrogram(
    sample_rate=SR, normalized=True, n_fft=N_FFT,
    hop_length=HOP_LENGTH, win_length=N_FFT,
    f_max=F_MAX, n_mels=N_MELS, f_min=F_MIN,
).to(DEVICE)
amp_to_db = T.AmplitudeToDB(top_db=80).to(DEVICE)

def compute_mel(wav_tensor):
    \"\"\"wav_tensor: (B, 1, n_samples). Returns mel (B, 1, n_mels, n_frames).\"\"\"
    mel = mel_transform(wav_tensor)  # (B, 1, n_mels, n_frames)
    mel = amp_to_db(mel)
    # Per-sample normalize
    B = mel.size(0)
    for i in range(B):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    return mel


# Sample submission for column order
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES
print(f"Primary labels: {len(PRIMARY_LABELS)}")
"""))

cells.append(code_cell("""# Sliding inference function
def infer_file(audio_60s):
    \"\"\"audio_60s: np.ndarray of shape (60s × SR,) = (1920000,).
    Returns: predictions (N_WINDOWS_OUT=12, N_CLASSES=234) as probability.
    \"\"\"
    # 1. Pad: 2.5s lead + 60s + 2.5s trail = 65s
    audio_padded = np.zeros(PADDED_SAMPLES, dtype=np.float32)
    audio_padded[PAD_LEAD_SAMPLES:PAD_LEAD_SAMPLES + len(audio_60s)] = audio_60s[:60 * SR]

    # 2. Slice into 12 windows (10s each, step 5s, starting at sec 0, 5, ..., 55 in padded coords)
    chunks = np.zeros((N_WINDOWS_OUT, WINDOW_SAMPLES), dtype=np.float32)
    for k in range(N_WINDOWS_OUT):
        start = k * STEP_SAMPLES
        end = start + WINDOW_SAMPLES
        chunks[k] = audio_padded[start:end]

    # 3. To tensor (B=12, 1, n_samples)
    wav_t = torch.from_numpy(chunks).unsqueeze(1).to(DEVICE)

    # 4. Compute mel (B, 1, 256, n_frames)
    mel = compute_mel(wav_t)

    # 5. Model forward
    with torch.no_grad():
        clip_logit, framewise = model(mel, return_framewise=True)
        frame_max = framewise.max(dim=-1).values
        p_clip = torch.sigmoid(clip_logit).float().cpu().numpy()
        p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
    probs = 0.5 * p_clip + 0.5 * p_fmax  # (12, 234)

    # 6. Smoothing across 12 windows
    probs_smoothed = gaussian_filter1d(probs, sigma=GAUSSIAN_SIGMA, axis=0,
                                       mode="nearest").astype(np.float32)
    return probs_smoothed
"""))

cells.append(code_cell("""# Inference loop on test_soundscapes
test_files = sorted(TEST_SC_DIR.glob("*.ogg"))
print(f"Test files: {len(test_files)}")

rows = []
t0 = time.time()
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

    probs = infer_file(y)  # (12, 234)

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
"""))

cells.append(code_cell("""# Build submission CSV
sub_df = pd.DataFrame(rows, columns=["row_id"] + PRIMARY_LABELS)
print(f"sub_df: {sub_df.shape}")

# Sanity
assert sub_df["row_id"].nunique() == len(sub_df), "Duplicate row_id"
print(f"  mean prob: {sub_df[PRIMARY_LABELS].mean().mean():.5f}")
print(f"  max prob:  {sub_df[PRIMARY_LABELS].max().max():.4f}")
print(f"  NaN: {sub_df[PRIMARY_LABELS].isna().sum().sum()}")

# Reindex to match sample_submission order
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
