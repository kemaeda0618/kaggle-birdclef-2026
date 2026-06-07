"""Build standalone 3-fold ensemble inference NB:
  exp029 R3 fold 0 + exp106 fold 1 + exp106 fold 2

All 3 ckpts share exp029 R3 architecture (eca_nfnet_l1 + Perch distill + GeM + dense + att/cla).
PyTorch direct inference (no OV conversion needed for one-off sub).
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path("experiment/exp106/notebook/nb_exp106_standalone_3fold.ipynb")
NB_PATH.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1 = """# === exp106 + exp029 R3 standalone 3-fold ensemble inference (PyTorch) ===
# Stack: exp029 R3 fold 0 + exp106 R3 fold 1 + exp106 R3 fold 2
# All ckpts use exp029 R3 architecture (eca_nfnet_l1 + Perch distill).
import os, sys, time, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import timm
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
"""

CELL2 = """# === Constants (matches exp029 R3 training config) ===
BASE = Path("/kaggle/input/birdclef-2026")
TEST_DIR = BASE / "test_soundscapes"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC          # 160000
N_WINDOWS = 12
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000

E17_BACKBONE = "eca_nfnet_l1"
E17_TRAIN_SAMPLES = SR * 5
E17_USE_DISTILL = True
E17_PERCH_DIM = 1536
N_TF_DIM = E17_TRAIN_SAMPLES // HOP + 1   # 313

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
print(f"N_CLASSES={N_CLASSES}, sample_sub rows={len(sample_sub)}")
print(f"TEST_DIR exists: {TEST_DIR.exists()}")
"""

CELL3 = """# === Locate 3 ckpts ===
CKPT_SPECS = [
    # (label, search_paths, ckpt_filename_candidates)
    ("exp029_fold0", [
        Path("/kaggle/input/birdclef2026-exp029-l1-single"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single"),
    ], ["r3_fold0_ckpt_best_ns22.pth"]),
    ("exp106_fold1", [
        Path("/kaggle/input/birdclef2026-exp106-fold1"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp106-fold1"),
    ], ["r3_fold1_ckpt_best_ns22.pth"]),
    ("exp106_fold2", [
        Path("/kaggle/input/birdclef2026-exp106-fold2"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp106-fold2"),
    ], ["r3_fold2_ckpt_best_ns22.pth"]),
]

CKPTS = {}
for label, search_paths, names in CKPT_SPECS:
    found = None
    for p in search_paths:
        if not p.exists():
            continue
        for name in names:
            hits = list(p.rglob(name))
            if hits:
                found = hits[0]; break
        if found:
            break
    assert found is not None, f"ckpt not found for {label}"
    CKPTS[label] = found
    print(f"  {label}: {found} ({found.stat().st_size/1e6:.1f}MB)")
"""

CELL4 = """# === Architecture (exp029 R3 BirdSEDModel) ===
class _GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _BirdSED(nn.Module):
    def __init__(self, backbone_name=E17_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, N_MELS, N_TF_DIM)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E17_USE_DISTILL:
            self.distill_head = _DistillHead(self.backbone_dim, E17_PERCH_DIM)
    def forward(self, x):
        h = self.backbone(x)
        h_cls = h.detach() if E17_USE_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        return clip_logits, framewise_logits.permute(0, 2, 1)
"""

CELL5 = """# === Load 3 ckpts ===
models = {}
for label, ckpt_path in CKPTS.items():
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    val = state.get('best_ns22', float('nan'))
    print(f"  {label}: epoch={state.get('epoch')}, best_ns22={val:.4f}")
    m = _BirdSED().cpu()
    missing, unexpected = m.load_state_dict(state["model_state"], strict=False)
    print(f"    missing={len(missing)}, unexpected={len(unexpected)}")
    m.eval()
    models[label] = m
print(f"Loaded {len(models)} models")
"""

CELL6 = """# === Mel transform ===
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
)
db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)


def load_60s(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    target = N_WINDOWS * WINDOW_SAMPLES
    if len(wav) < target:
        wav = np.concatenate([wav, np.zeros(target - len(wav), dtype=np.float32)])
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


# Locate test audio (fallback to train_soundscapes for dry-run)
if TEST_DIR.exists():
    test_files = sorted(TEST_DIR.glob("*.ogg"))
else:
    test_files = []
if not test_files:
    test_files = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:20]
    print(f"[DRY-RUN] {len(test_files)} train_soundscape files")
else:
    print(f"hidden test_soundscapes: {len(test_files)} files")
"""

CELL7 = """# === Inference loop (3-fold ensemble) ===
t0 = time.time()
all_probs = []
file_ids = []

with torch.no_grad():
    for fi, fpath in enumerate(test_files):
        raw_60s = load_60s(fpath)
        chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
        wav_t = torch.from_numpy(chunks).unsqueeze(1)  # (12, 1, 160000)
        mel = db_tf(mel_tf(wav_t))
        # per-instance standardize (matches exp029 R3)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)

        # 3-fold ensemble: average clip + frame
        clip_sum = None; frame_sum = None
        for label, m in models.items():
            clip_logits, frame = m(mel)
            if clip_sum is None:
                clip_sum = clip_logits.numpy().copy()
                frame_sum = frame.numpy().copy()
            else:
                clip_sum += clip_logits.numpy()
                frame_sum += frame.numpy()
        n = len(models)
        clip_avg = clip_sum / n
        frame_avg = frame_sum / n
        frame_max = frame_avg.max(axis=1)

        # sigmoid blend + gaussian smooth (matches exp090 e29-infer)
        p_clip = 1.0 / (1.0 + np.exp(-np.clip(clip_avg, -50, 50)))
        p_frame = 1.0 / (1.0 + np.exp(-np.clip(frame_max, -50, 50)))
        p_mean = (0.5 * p_clip + 0.5 * p_frame).astype(np.float32)
        p_smooth = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        all_probs.append(p_smooth)

        fname_no_ext = fpath.stem
        for w in range(N_WINDOWS):
            end_sec = (w + 1) * WINDOW_SEC
            file_ids.append(f"{fname_no_ext}_{end_sec}")

        if (fi + 1) % 50 == 0 or fi == len(test_files) - 1:
            print(f"  [{fi+1}/{len(test_files)}] {time.time()-t0:.0f}s")

all_probs = np.concatenate(all_probs, axis=0)
print(f"all_probs shape: {all_probs.shape}, time: {time.time()-t0:.0f}s")
"""

CELL8 = """# === Build submission.csv ===
preds_df = pd.DataFrame(all_probs, columns=PRIMARY_LABELS)
preds_df.insert(0, "row_id", file_ids)

sub = sample_sub[["row_id"]].merge(preds_df, on="row_id", how="left")
sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(0.0)
n_missing = (sub[PRIMARY_LABELS].sum(axis=1) == 0).sum()
print(f"missing row_ids (filled 0): {n_missing}")

sub.to_csv("submission.csv", index=False)
print(f"submission.csv: shape {sub.shape}")
print(f"  mean pred: {sub[PRIMARY_LABELS].values.mean():.6f}")
print(f"  max pred: {sub[PRIMARY_LABELS].values.max():.6f}")
print(sub.head(3))
"""

CELLS = [
    (CELL1, "imports"),
    (CELL2, "const"),
    (CELL3, "locate-ckpts"),
    (CELL4, "arch"),
    (CELL5, "load"),
    (CELL6, "mel-files"),
    (CELL7, "infer-loop"),
    (CELL8, "submission"),
]

nb = {
    "cells": [cell(s, cid) for s, cid in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes, {len(CELLS)} cells)")

for src, cid in CELLS:
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<{cid}>", "exec")
        print(f"  [OK] {cid}")
    except SyntaxError as e:
        print(f"  [FAIL] {cid}: {e}")
