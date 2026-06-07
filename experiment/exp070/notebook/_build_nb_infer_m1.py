"""Build M1 (exp070) inference NB for Kaggle CPU sub.

M1 5-fold ensemble inference on test_soundscapes:
  - Load 5 M1 ckpts from `maekeso/birdclef2026-exp070-m1-nfnet-l1-perch`
  - For each test chunk (5s) compute prediction
  - 5-fold rank-avg ensemble
  - Generate submission.csv

Kaggle CPU 90min limit. nfnet_l1 + 5s mel × 234 class.
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_infer_m1.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# exp070 M1 inference (5-fold ensemble, CPU)

**M1 5-fold ensemble submission**:
  - Backbone: eca_nfnet_l1.ra2_in1k
  - 5s × 256 × 2048 × 512 mel
  - DistillHead (Perch 1536-d) — inference 時 unused
  - 5 fold rank-avg ensemble
  - Output: submission.csv (BC2026 format)

**Kaggle Settings**:
  - CPU only (no GPU)
  - internet=False
  - competition_sources=["birdclef-2026"]
  - dataset_sources=["maekeso/birdclef2026-exp070-m1-nfnet-l1-perch"]
""")

# =============================================================
code("""# ============================================================
# Cell 1: Imports + Setup
# ============================================================
import os, sys, time, gc, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import timm

DEVICE = torch.device("cpu")  # ★ Kaggle CPU NB
torch.set_num_threads(4)
print(f"Device: {DEVICE}, torch {torch.__version__}, timm {timm.__version__}")
START = time.time()
""")

# =============================================================
code("""# ============================================================
# Cell 2: Config (M1 spec)
# ============================================================
NUM_CLASSES = 234
SR = 32000
CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC      # 160_000 samples per 5s chunk
N_WINDOWS = 12                      # 60s / 5s
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000

BACKBONE = "eca_nfnet_l1"
PERCH_EMBED_DIM = 1536
DROP_PATH = 0.0                     # inference: drop_path=0
HIDDEN_DIM = 512

BATCH_SIZE = 32                     # CPU batch size

# Paths
DATA_PATHS = ["/kaggle/input/competitions/birdclef-2026",
              "/kaggle/input/birdclef-2026"]
DATA_PATH = None
for _p in DATA_PATHS:
    if Path(_p).exists():
        DATA_PATH = Path(_p); break
assert DATA_PATH is not None, f"BC2026 data not found"

TEST_DIR = DATA_PATH / "test_soundscapes"
SAMPLE_SUB_PATH = DATA_PATH / "sample_submission.csv"
TAXONOMY_CSV = DATA_PATH / "taxonomy.csv"

M1_CKPT_DIR = None
for _p in ["/kaggle/input/birdclef2026-exp070-m1-nfnet-l1-perch",
           "/kaggle/input/datasets/maekeso/birdclef2026-exp070-m1-nfnet-l1-perch"]:
    if Path(_p).exists():
        M1_CKPT_DIR = Path(_p); break
assert M1_CKPT_DIR is not None, "M1 ckpts not attached"
print(f"M1 ckpt dir: {M1_CKPT_DIR}")

# Locate fold ckpts
fold_ckpts = sorted(M1_CKPT_DIR.glob("m1_fold*_ckpt_best_ns22.pth"))
if len(fold_ckpts) == 0:
    fold_ckpts = sorted(M1_CKPT_DIR.glob("*ckpt_best_ns22.pth"))
if len(fold_ckpts) == 0:
    fold_ckpts = sorted(M1_CKPT_DIR.rglob("*ckpt_best*.pth"))
print(f"Found {len(fold_ckpts)} fold ckpts:")
for f in fold_ckpts:
    print(f"  {f}")
assert len(fold_ckpts) > 0, "No M1 ckpts found"
""")

# =============================================================
code("""# ============================================================
# Cell 3: Load BC26 label list
# ============================================================
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES, f"Expected 234 labels, got {len(PRIMARY_LABELS)}"
print(f"BC26 labels: {len(PRIMARY_LABELS)}")
""")

# =============================================================
code("""# ============================================================
# Cell 4: Model architecture (M1 spec, identical to train NB)
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


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


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
                 drop_path_rate=DROP_PATH, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = CHUNK_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.0), nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.0),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        # DistillHead present for state_dict compat, unused at inference
        self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        framewise_logits = self.cla(h_cls)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


print("OK model defs")
""")

# =============================================================
code("""# ============================================================
# Cell 5: Test soundscape file enumeration
# ============================================================
test_files = sorted(TEST_DIR.glob("*.ogg"))
print(f"Test soundscape files: {len(test_files)}")
if len(test_files) == 0:
    # Sanity check: try local files for dev
    print("[WARN] No test files in TEST_DIR — using sample_submission row_ids")
""")

# =============================================================
code("""# ============================================================
# Cell 6: Inference function (load audio, chunk, predict)
# ============================================================
def load_audio_60s(path):
    \"\"\"Load 60s audio, return waveform [SR * 60].\"\"\"
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return np.zeros(SR * 60, dtype=np.float32)


def split_into_chunks(wav, chunk_samples=CHUNK_SAMPLES, n_chunks=N_WINDOWS):
    \"\"\"Split 60s waveform into 12 chunks of 5s each.\"\"\"
    target = chunk_samples * n_chunks
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    chunks = []
    for i in range(n_chunks):
        chunks.append(wav[i*chunk_samples:(i+1)*chunk_samples])
    return np.stack(chunks)   # (12, 160000)


def predict_one_fold(model, mel_transform, all_chunks, batch_size=BATCH_SIZE):
    \"\"\"Predict on all chunks (N_files * 12, 160000) with one fold model.\"\"\"
    model.eval()
    preds_blend = []
    n = len(all_chunks)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            batch = torch.from_numpy(all_chunks[s:s+batch_size]).unsqueeze(1).to(DEVICE)  # (B, 1, 160000)
            mel = mel_transform(batch)
            # Per-instance z-score normalization
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            # Forward
            clip_logits, framewise = model(mel, return_framewise=True)
            frame_max = framewise.max(dim=1).values
            p_clip = torch.sigmoid(clip_logits).float().numpy()
            p_fmax = torch.sigmoid(frame_max).float().numpy()
            p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_blend.append(p_blend)
    return np.concatenate(preds_blend)   # (N, 234)


print("OK inference funcs")
""")

# =============================================================
code("""# ============================================================
# Cell 7: Pre-load all test waveforms + extract chunks
# ============================================================
t0 = time.time()
print(f"Loading {len(test_files)} test files...")

# Build all_chunks: shape (n_files * 12, 160000)
all_chunks = []
file_ids = []
for i, f in enumerate(test_files):
    wav = load_audio_60s(f)
    chunks = split_into_chunks(wav)
    all_chunks.append(chunks)
    fname = f.stem   # e.g. BC2026_Test_0001_S05_20250227_010002
    for c in range(N_WINDOWS):
        # row_id = {filename}_{end_time}
        end_sec = (c + 1) * 5
        file_ids.append(f"{fname}_{end_sec}")
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_files)} loaded ({(time.time()-t0)/60:.1f}min)")
all_chunks = np.concatenate(all_chunks, axis=0)
print(f"All chunks shape: {all_chunks.shape}, files: {len(test_files)}")
print(f"  Load time: {(time.time()-t0)/60:.1f}min")
""")

# =============================================================
code("""# ============================================================
# Cell 8: 5-fold ensemble inference
# ============================================================
mel_transform = MelSpecTransform().to(DEVICE)

n_total_chunks = len(all_chunks)
all_fold_preds = np.zeros((len(fold_ckpts), n_total_chunks, NUM_CLASSES), dtype=np.float32)

for fi, ckpt_path in enumerate(fold_ckpts):
    t_fold = time.time()
    print(f"\\n=== Fold {fi}: {ckpt_path.name} ===")

    model = BirdSEDModel()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"  Load state: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    val_ns22 = ckpt.get("best_ns22", ckpt.get("val_ns22", -1))
    val_macro = ckpt.get("best_macro", ckpt.get("val_macro", -1))
    print(f"  ckpt val_ns22={val_ns22:.4f} val_macro={val_macro:.4f}")
    model = model.to(DEVICE)
    model.eval()

    preds = predict_one_fold(model, mel_transform, all_chunks)
    all_fold_preds[fi] = preds
    print(f"  Fold {fi} done in {(time.time()-t_fold)/60:.1f}min, preds shape={preds.shape}")

    del model; gc.collect()

print(f"\\nAll {len(fold_ckpts)} folds done in {(time.time()-START)/60:.1f}min")
""")

# =============================================================
code("""# ============================================================
# Cell 9: 5-fold rank-avg ensemble + submission
# ============================================================
print("=== 5-fold rank-avg ensemble ===")

# Rank-avg per fold (per-class percentile rank)
fold_ranks = np.zeros_like(all_fold_preds)
for fi in range(len(fold_ckpts)):
    flat = all_fold_preds[fi]    # (N, 234)
    rank = pd.DataFrame(flat).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    fold_ranks[fi] = rank

ensemble_pred = fold_ranks.mean(axis=0)   # (N, 234)
print(f"Ensemble pred shape: {ensemble_pred.shape}")
print(f"Stats: mean={ensemble_pred.mean():.4f}, max={ensemble_pred.max():.4f}, min={ensemble_pred.min():.4f}")

# Build submission DataFrame
sub_df = pd.DataFrame(ensemble_pred, columns=PRIMARY_LABELS)
sub_df.insert(0, "row_id", file_ids)

# Verify row count matches sample_submission
expected_rows = len(sample_sub)
actual_rows = len(sub_df)
print(f"Submission rows: {actual_rows} (expected {expected_rows})")
if actual_rows != expected_rows:
    print(f"[WARN] Row count mismatch")
    # Align to sample_sub row order if possible
    sub_map = sub_df.set_index("row_id")
    aligned = sample_sub[["row_id"]].copy()
    for lbl in PRIMARY_LABELS:
        aligned[lbl] = aligned["row_id"].map(sub_map[lbl] if lbl in sub_map.columns else None).fillna(0.0)
    sub_df = aligned
    print(f"  Aligned to sample_sub: {len(sub_df)} rows")

# Save
out_path = Path("/kaggle/working") / "submission.csv"
sub_df.to_csv(out_path, index=False)
print(f"\\nOK Submission saved: {out_path} ({out_path.stat().st_size/1e6:.1f}MB)")
print(f"\\nTotal time: {(time.time()-START)/60:.1f}min")
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_OUT}")
print(f"Cells: {len(CELLS)}")
for i, c in enumerate(CELLS):
    src = "".join(c.get("source", []))
    head = src.split('\n')[0][:70] if src else "(empty)"
    print(f"  Cell {i:2d} ({c['cell_type']:8s}) {len(src):6d} chars | {head}")
