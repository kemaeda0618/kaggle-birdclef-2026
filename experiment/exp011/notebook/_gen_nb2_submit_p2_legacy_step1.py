"""Generate exp011 Phase 2 Step 1 SUBMISSION notebook.

Differences from Phase 1 submit:
- 20-sec context window (was 10s)
- target_size (256, 384) (was 256, 256)
- Per-chunk mel computation (training match)
- weight from kernel_sources: maekeso/birdclef2026-exp011-train-phase2-step1

CPU only. Uses same PP as Phase 1 v2 (file_conf_scale + delta-shift smoothing).
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

# ── Header ──
cells.append(md_cell("hdr", """# exp011 Phase 2 Step 1: Submission (CPU 90min)

- Weight: kernel_sources `maekeso/birdclef2026-exp011-train-phase2-step1` (best.pth)
- Context window: **20 sec ending at end_time** (matches training anchor convention)
- target_size: (256, 384)
- PP: file_confidence_scale (top_K=2, power=0.4) + adaptive delta-shift smoothing (alpha=0.20)
"""))

# ── Imports ──
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

# ── Config ──
cells.append(code_cell("config", r"""# ==============================================================
# CONFIG (must match training: _gen_nb1_train_p2.py)
# ==============================================================
CFG = dict(
    sr=32_000, n_mels=256, n_fft=2048, hop_length=512,
    fmin=20, fmax=16_000, top_db=80,
    db_min=-80.0, db_max=20.0,
    chunk_duration=20.0, target_size=(256, 384),
    backbone="tf_efficientnetv2_b0", num_classes=234,
    in_channels=3, dropout=0.1, gem_p_init=3.0,
)
CFG["chunk_samples"] = int(CFG["chunk_duration"] * CFG["sr"])  # 640000
CFG["chunk_frames"] = int(CFG["chunk_duration"] * CFG["sr"] / CFG["hop_length"]) + 1  # 1251
print(f"chunk_samples: {CFG['chunk_samples']}, chunk_frames: {CFG['chunk_frames']}")"""))

# ── Paths ──
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

# Weight from kernel_sources (Phase 2 Step 1 training NB output)
WEIGHT_PATH = None
for cand in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp011-train-phase2-step1/weights/best.pth"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp011-train-phase2-step1/best.pth"),
    Path("/kaggle/input/birdclef2026-exp011-train-phase2-step1/weights/best.pth"),
]:
    if cand.exists():
        WEIGHT_PATH = cand; break
assert WEIGHT_PATH is not None, "best.pth not found in kernel_sources"
print(f"WEIGHT: {WEIGHT_PATH}")

sub_df_head = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
SPECIES = list(sub_df_head.columns[1:])
print(f"Species: {len(SPECIES)}")

test_files = sorted(TEST_DIR.rglob("*.ogg")) if TEST_DIR.exists() else []
print(f"test_soundscapes: {len(test_files)} files")"""))

# ── Audio loading ──
cells.append(code_cell("audio", r"""# ==============================================================
# AUDIO LOADING + chunk anchoring (20s ending at end_time)
# ==============================================================
def load_audio_full(path):
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != CFG["sr"]:
        audio = torchaudio.functional.resample(audio, sr, CFG["sr"])
    return audio.squeeze(0).numpy().astype(np.float32)


def take_chunk_at(audio, start_sample, n_samples):
    out = np.zeros(n_samples, dtype=np.float32)
    s = max(0, start_sample)
    e = min(len(audio), start_sample + n_samples)
    if s < e:
        out_off = s - start_sample
        out[out_off:out_off + (e - s)] = audio[s:e]
    return out


# Sanity check
if test_files:
    _t0 = time.time()
    _a = load_audio_full(test_files[0])
    print(f"load_audio_full smoke: {test_files[0].name} -> {len(_a)} samples in {time.time()-_t0:.2f}s")"""))

# ── Mel transform (matches training) ──
cells.append(code_cell("mel-transform", r"""# ==============================================================
# MEL TRANSFORM (raw audio -> mel_db -> resize/norm/3ch)
# ==============================================================
class MelTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg["sr"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"], f_min=cfg["fmin"], f_max=cfg["fmax"], power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=cfg["top_db"])
        self.resize = torchvision.transforms.Resize(cfg["target_size"], antialias=True)
        self.db_min = cfg["db_min"]
        self.db_max = cfg["db_max"]

    @torch.no_grad()
    def forward(self, audio):
        # audio: (B, n_samples)
        audio = audio.float()
        mel = self.mel_spec(audio)
        mel_db = self.amp_to_db(mel)
        mel_db = mel_db.clamp(min=self.db_min, max=self.db_max)
        x = self.resize(mel_db.unsqueeze(1)).squeeze(1)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        mn = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        x = (x - mn) / (mx - mn + 1e-7)
        return x.unsqueeze(1).repeat(1, 3, 1, 1)


mel_transform = MelTransform(CFG).to(DEVICE)
print("MelTransform ready")"""))

# ── Model ──
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
print(f"Loaded best.pth: epoch={ckpt['epoch']+1}")
print(f"  val_a_auc: {ckpt['metrics_val_a']['macro_auc']:.4f} ({ckpt['metrics_val_a']['num_classes_evaluated']} cls)")
print(f"  val_b_auc: {ckpt['metrics_val_b']['macro_auc']:.4f} ({ckpt['metrics_val_b']['num_classes_evaluated']} cls)")"""))

# ── Inference (12 windows per file, batched) ──
cells.append(code_cell("inference", r"""# ==============================================================
# INFERENCE (12 windows per file, batched mel + model)
# ==============================================================
END_TIMES = list(range(5, 65, 5))  # [5, 10, ..., 60]
N_WINDOWS = len(END_TIMES)


@torch.no_grad()
def predict_file(wav_path):
    audio = load_audio_full(wav_path)  # 1D float32
    n_s = CFG["chunk_samples"]
    chunks = []
    for et in END_TIMES:
        end_sample = int(et * CFG["sr"])
        start_sample = end_sample - n_s
        chunks.append(take_chunk_at(audio, start_sample, n_s))
    batch = torch.from_numpy(np.stack(chunks, axis=0)).to(DEVICE)  # (12, n_s)
    mel = mel_transform(batch)
    out = model(mel)
    return out["clipwise_prob"].cpu().numpy()  # (12, 234)


# Smoke test
if test_files:
    _t0 = time.time()
    _p = predict_file(test_files[0])
    _dt = time.time() - _t0
    print(f"predict_file smoke: {test_files[0].name} -> {_p.shape} in {_dt:.2f}s")
    print(f"Estimated total: {len(test_files) * _dt:.0f}s for {len(test_files)} files")
else:
    print("WARNING: no test files (likely local run); skip smoke test")"""))

# ── Submission ──
cells.append(code_cell("submission", r"""# ==============================================================
# RUN ALL FILES + PP -> submission.csv
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

# ── Assemble ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb2_submit_p2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
