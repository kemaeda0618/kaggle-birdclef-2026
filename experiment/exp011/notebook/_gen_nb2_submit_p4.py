"""Generate exp011 Phase 4 SUBMISSION notebook (CPU 90min inference).

Phase 4 submit = Phase 2 submit と同じ推論パイプライン (ogg 直読 + GPU mel は既に Phase 2 submit で使用済み)。
変更点:
- kernel_sources: birdclef2026-exp011-train-phase2 -> birdclef2026-exp011-train-phase4-sed
- WEIGHT_FNAME: best.pth (Phase 4 は Val-B primary = best.pth に保存)
- Header / コメント更新

CPU only. machine_shape NOT specified.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

# Header
cells.append(md_cell("hdr", """# exp011 Phase 4: Submission (CPU 90min)

Phase 4 (raw waveform mixup) の weight を使った提出 NB。推論パイプラインは Phase 2 submit と同じ。

- Weight: kernel_sources `maekeso/birdclef2026-exp011-train-phase4-sed` の `best.pth` (= Val-B primary)
- 12 windows per file: end_time in {5, 10, ..., 60}, 20-sec context window (start = et - 20)
- PP: file_confidence_scale (top_K=2, power=0.4) + adaptive delta-shift smoothing (alpha base=0.20)
- Output: `/kaggle/working/submission.csv`
"""))

# Imports
cells.append(code_cell("imports", r"""import os, gc, time, json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision
import timm

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
torch.set_num_threads(os.cpu_count() or 4)
WALL_START = time.time()"""))

# Config
cells.append(code_cell("config", r"""# ==============================================================
# CONFIG (must match training: _gen_nb1_train_p4.py)
# ==============================================================
CFG = dict(
    sr=32_000, n_mels=256, n_fft=2048, hop_length=512,
    fmin=20, fmax=16_000, top_db=80,
    db_min=-80.0, db_max=20.0,
    chunk_duration=20.0, target_size=(256, 256),
    backbone="tf_efficientnetv2_b0", num_classes=234,
    in_channels=3, dropout=0.1, gem_p_init=3.0,
)
CFG["chunk_frames"] = int(CFG["chunk_duration"] * CFG["sr"] / CFG["hop_length"]) + 1  # 1251
print(f"chunk_frames: {CFG['chunk_frames']}")"""))

# Paths
cells.append(code_cell("paths", r"""# ==============================================================
# PATHS (autodetect)
# ==============================================================
DATA_ROOT = None
for cand in [Path("/kaggle/input/competitions/birdclef-2026"),
             Path("/kaggle/input/birdclef-2026")]:
    if cand.exists():
        DATA_ROOT = cand; break
assert DATA_ROOT is not None, "birdclef-2026 not mounted"
print(f"DATA_ROOT: {DATA_ROOT}")

TEST_DIR = DATA_ROOT / "test_soundscapes"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"

# Phase 4: Val-B primary = best.pth
WEIGHT_FNAME = "best.pth"
WEIGHT_PATH = None
for cand in [
    Path(f"/kaggle/input/notebooks/maekeso/birdclef2026-exp011-train-phase4-sed/weights/{WEIGHT_FNAME}"),
    Path(f"/kaggle/input/notebooks/maekeso/birdclef2026-exp011-train-phase4-sed/{WEIGHT_FNAME}"),
    Path(f"/kaggle/input/birdclef2026-exp011-train-phase4-sed/weights/{WEIGHT_FNAME}"),
]:
    if cand.exists():
        WEIGHT_PATH = cand; break
assert WEIGHT_PATH is not None, f"{WEIGHT_FNAME} not found in kernel_sources"
print(f"WEIGHT: {WEIGHT_PATH}")

sub_df_head = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
SPECIES = list(sub_df_head.columns[1:])
print(f"Species: {len(SPECIES)}")

test_files = sorted(TEST_DIR.rglob("*.ogg")) if TEST_DIR.exists() else []
print(f"test_soundscapes: {len(test_files)} files")"""))

# Mel transform
cells.append(code_cell("mel-transform", r"""# ==============================================================
# MEL TRANSFORM (identical to training pipeline)
# ==============================================================
mel_transform_audio = T.MelSpectrogram(
    sample_rate=CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
    n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"], power=2.0,
).to(DEVICE)

db_transform = T.AmplitudeToDB(stype="power", top_db=CFG["top_db"]).to(DEVICE)


def compute_full_mel(wav_path):
    audio, sr = torchaudio.load(str(wav_path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != CFG["sr"]:
        audio = torchaudio.functional.resample(audio, sr, CFG["sr"])
    if audio.shape[1] < CFG["n_fft"]:
        audio = F.pad(audio, (0, CFG["n_fft"] - audio.shape[1]))
    audio = audio.to(DEVICE)
    with torch.no_grad():
        mel = mel_transform_audio(audio)
        mel_db = db_transform(mel)
    mel_db = mel_db.squeeze(0).clamp(min=CFG["db_min"], max=CFG["db_max"])
    return mel_db


if test_files:
    _t0 = time.time()
    _m = compute_full_mel(test_files[0])
    print(f"compute_full_mel test: {test_files[0].name} -> {tuple(_m.shape)} in {time.time()-_t0:.2f}s")"""))

# Post transform
cells.append(code_cell("post-transform", r"""# ==============================================================
# POST TRANSFORM (resize 256x256 + per-sample min-max + 3ch)
# ==============================================================
class MelPostTransform(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.resize = torchvision.transforms.Resize(target_size, antialias=True)

    @torch.no_grad()
    def forward(self, mel_db):
        x = mel_db.float()
        x = self.resize(x.unsqueeze(1)).squeeze(1)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        mn = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        x = (x - mn) / (mx - mn + 1e-7)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return x


post_transform = MelPostTransform(CFG["target_size"]).to(DEVICE)


def crop_window(full_mel, start_frame, n_frames):
    Tf = full_mel.shape[1]
    s = max(0, start_frame)
    e = min(Tf, start_frame + n_frames)
    out = torch.zeros(full_mel.shape[0], n_frames, device=full_mel.device, dtype=full_mel.dtype)
    if s < e:
        out_offset = s - start_frame
        out[:, out_offset:out_offset + (e - s)] = full_mel[:, s:e]
    return out


print("MelPostTransform + crop_window ready")"""))

# Model
cells.append(code_cell("model", r"""# ==============================================================
# MODEL (identical to training)
# ==============================================================
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init)); self.eps = eps

    def forward(self, x):
        x = x.float()
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        clipwise_logit = (att * cls).sum(dim=-1)
        return {"clipwise_logit": clipwise_logit,
                "clipwise_prob": torch.sigmoid(clipwise_logit)}


class SEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["backbone"], pretrained=False, in_chans=cfg["in_channels"],
            features_only=False, global_pool="", num_classes=0,
        )
        feat_dim = self.backbone.num_features
        self.gem_pool = GEMFreqPool(p_init=cfg["gem_p_init"])
        self.head = AttentionSEDHead(feat_dim, cfg["num_classes"], cfg["dropout"])

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.gem_pool(feat)
        return self.head(pooled)


model = SEDModel(CFG)
ckpt = torch.load(str(WEIGHT_PATH), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()
ma = ckpt.get("metrics_val_a", {})
mb = ckpt.get("metrics_val_b", {})
print(f"Loaded {WEIGHT_FNAME}: epoch={ckpt['epoch']+1}")
print(f"  Val-A AUC={ma.get('macro_auc', 'NA')} ({ma.get('num_classes_evaluated', 'NA')} cls)")
print(f"  Val-B AUC={mb.get('macro_auc', 'NA')} ({mb.get('num_classes_evaluated', 'NA')} cls)")
print(f"Backbone features: {model.backbone.num_features}, classes: {CFG['num_classes']}")"""))

# Inference
cells.append(code_cell("inference", r"""# ==============================================================
# INFERENCE (12 windows per file, 20-sec context each)
# ==============================================================
END_TIMES = list(range(5, 65, 5))
N_WINDOWS = len(END_TIMES)
FRAMES_PER_SEC = CFG["sr"] / CFG["hop_length"]
CHUNK_FR = CFG["chunk_frames"]


@torch.no_grad()
def predict_file(wav_path):
    full_mel = compute_full_mel(wav_path)
    chunks = []
    for et in END_TIMES:
        start_sec = et - CFG["chunk_duration"]
        start_frame = int(start_sec * FRAMES_PER_SEC)
        chunks.append(crop_window(full_mel, start_frame, CHUNK_FR))
    batch = torch.stack(chunks, dim=0)
    mel = post_transform(batch)
    out = model(mel)
    return out["clipwise_prob"].cpu().numpy()


if test_files:
    _t0 = time.time()
    _p = predict_file(test_files[0])
    print(f"predict_file smoke: {test_files[0].name} -> {_p.shape} in {time.time()-_t0:.2f}s")
    print(f"Estimated total: {len(test_files) * (time.time()-_t0):.0f}s for {len(test_files)} files")
else:
    print("WARNING: no test files (likely local run); skip smoke test")"""))

# Submission
cells.append(code_cell("submission", r"""# ==============================================================
# RUN ALL FILES -> post-processing -> submission.csv
# ==============================================================
SUB_PATH = Path("/kaggle/working/submission.csv")

PP_FILE_CONF_TOPK = 2
PP_FILE_CONF_POWER = 0.4
PP_DSS_BASE_ALPHA = 0.20

if not test_files:
    sub_template = pd.read_csv(SAMPLE_SUB_CSV)
    sub_template.to_csv(SUB_PATH, index=False)
    print(f"No test files; wrote sample_submission template to {SUB_PATH}")
else:
    all_probs = []
    all_stems = []
    t0 = time.time()
    for i, fp in enumerate(test_files):
        probs = predict_file(fp)
        all_probs.append(probs)
        all_stems.append(fp.stem)
        if (i + 1) % 20 == 0 or i == len(test_files) - 1:
            elapsed = time.time() - t0
            est_total = elapsed / (i + 1) * len(test_files)
            print(f"  [{i+1}/{len(test_files)}] elapsed {elapsed:.0f}s, est total {est_total:.0f}s")

    view = np.stack(all_probs, axis=0).astype(np.float32)
    print(f"Predictions shape: {view.shape}")

    # PP-1: file_confidence_scale
    top_mean = np.sort(view, axis=1)[:, -PP_FILE_CONF_TOPK:, :].mean(axis=1, keepdims=True)
    scale = np.power(np.clip(top_mean, 1e-7, 1.0), PP_FILE_CONF_POWER)
    view = view * scale
    print(f"  PP-1 file_confidence_scale (top_K={PP_FILE_CONF_TOPK}, power={PP_FILE_CONF_POWER}) applied")

    # PP-2: adaptive delta-shift smoothing
    smoothed = view.copy()
    n_t = view.shape[1]
    alpha_means = []
    for t in range(1, n_t - 1):
        conf = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = PP_DSS_BASE_ALPHA * (1.0 - conf)
        neighbor_avg = (view[:, t - 1, :] + view[:, t + 1, :]) / 2.0
        smoothed[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
        alpha_means.append(float(alpha.mean()))
    view = np.clip(smoothed, 0.0, 1.0)
    print(f"  PP-2 adaptive delta-shift smoothing (alpha base={PP_DSS_BASE_ALPHA}) applied; mean alpha={np.mean(alpha_means):.3f}")

    flat = view.reshape(-1, view.shape[-1])
    row_ids = [f"{stem}_{et}" for stem in all_stems for et in END_TIMES]
    sub_df = pd.DataFrame(flat, columns=SPECIES)
    sub_df.insert(0, "row_id", row_ids)
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"\nWrote {len(sub_df)} rows to {SUB_PATH}")
    print(sub_df.head(2))

print(f"\nWall time: {(time.time()-WALL_START)/60:.1f} min")"""))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb2_submit_p4.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
