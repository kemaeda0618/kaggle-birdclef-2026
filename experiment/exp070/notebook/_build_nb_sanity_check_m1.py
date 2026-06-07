"""Build M1 PyTorch vs ONNX sanity check NB (Colab).

Verifies if ONNX export is correct by comparing PyTorch and ONNX outputs
on identical inputs. If diff is large, ONNX is broken.

Loads M1 fold 0 from:
  - .pth: Drive (output/exp070/fold0/r3/ckpt_best_ns22.pth)
  - .onnx: Drive (if cached) or Kaggle Dataset download
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_sanity_check_m1.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# M1 PyTorch vs ONNX Sanity Check (Colab)

**Goal**: 0 sub cost で M1 ONNX が壊れてるかを直接検証

**Method**:
1. M1 fold 0 .pth → PyTorch model
2. M1 fold 0 .onnx → ONNX runtime session
3. Same dummy mel input → both
4. Compare outputs (max abs diff, mean diff)

**Conclusion**:
- max diff < 1e-3: ONNX OK (recall LB 0.81 ≠ ONNX issue)
- max diff > 1e-1: ONNX broken (need re-export)
""")

# =============================================================
code("""# Cell 1: Setup
!pip install -q timm onnx onnxruntime kaggle 2>&1 | tail -2

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil
from pathlib import Path

DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_EXP_DIR   = DRIVE_INPUT_DIR / "output" / "exp070"
assert DRIVE_EXP_DIR.exists()

# Kaggle auth (for ONNX download fallback)
KJ_CANDIDATES = [DRIVE_INPUT_DIR / "kaggle.json", Path("/content/drive/MyDrive/kaggle.json")]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
if KJ is not None:
    KAGGLE_CFG = Path.home() / ".kaggle"
    KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
    os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
    creds = json.loads(KJ.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]
""")

# =============================================================
code("""# Cell 2: Imports + model arch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
import onnxruntime as ort

print(f"torch: {torch.__version__}, ort: {ort.__version__}, timm: {timm.__version__}")

# M1 model spec
NUM_CLASSES = 234
SR = 32000
CHUNK_SAMPLES = SR * 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
BACKBONE = "eca_nfnet_l1"
PERCH_EMBED_DIM = 1536
HIDDEN_DIM = 512


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
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES, drop_path_rate=0.0, hidden_dim=HIDDEN_DIM):
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
        self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x):
        # Same as inference path in train NB
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        framewise_logits = self.cla(h_cls)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        return clip_logits, framewise_logits.permute(0, 2, 1)


print("OK model defs")
""")

# =============================================================
code("""# Cell 3: Locate .pth and .onnx
import time

FOLD = 0   # ★ M1 fold 0 (submitted LB 0.814)

# .pth path (Drive)
pth_path = DRIVE_EXP_DIR / f"fold{FOLD}" / "r3" / "ckpt_best_ns22.pth"
assert pth_path.exists(), f".pth not found: {pth_path}"
print(f".pth: {pth_path} ({pth_path.stat().st_size/1e6:.1f}MB)")

# .onnx path (Drive cache or download from Kaggle Dataset)
onnx_cache_dir = Path("/content/m1_onnx_cache")
onnx_cache_dir.mkdir(exist_ok=True)
onnx_path = onnx_cache_dir / f"m1_fold{FOLD}.onnx"

if not onnx_path.exists():
    print(f"\\nDownloading M1 ONNX from Kaggle Dataset...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files("maekeso/birdclef2026-exp070-m1-onnx",
                                path=str(onnx_cache_dir), unzip=True, quiet=False)
    print(f"  Files in cache:")
    for f in sorted(onnx_cache_dir.iterdir()):
        print(f"    {f.name}: {f.stat().st_size/1e6:.1f}MB")

assert onnx_path.exists(), f".onnx not found: {onnx_path}"
print(f"\\n.onnx: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f}MB)")
""")

# =============================================================
code("""# Cell 4: Load PyTorch model
DEVICE = torch.device("cpu")

print(f"Loading PyTorch model from {pth_path.name}...")
model_pt = BirdSEDModel()
ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
state = ckpt.get("model_state", ckpt)
msg = model_pt.load_state_dict(state, strict=False)
print(f"  load_state_dict: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
if msg.missing_keys:
    print(f"    missing samples: {msg.missing_keys[:5]}")
if msg.unexpected_keys:
    print(f"    unexpected samples: {msg.unexpected_keys[:5]}")
val_ns22 = ckpt.get("best_ns22", ckpt.get("val_ns22", -1))
val_macro = ckpt.get("best_macro", ckpt.get("val_macro", -1))
print(f"  ckpt val_ns22={val_ns22:.4f} val_macro={val_macro:.4f}")
model_pt.eval()
print(f"  Params: {sum(p.numel() for p in model_pt.parameters())/1e6:.1f}M")

print(f"\\nLoading ONNX session...")
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
print(f"  Inputs: {[i.name for i in sess.get_inputs()]}")
print(f"  Outputs: {[o.name for o in sess.get_outputs()]}")
""")

# =============================================================
code("""# Cell 5: Diff check on dummy input
print("="*60)
print("=== Sanity check: PyTorch vs ONNX output diff ===")
print("="*60)

# Dummy mel input (same as ONNX export)
n_tf = CHUNK_SAMPLES // HOP_LENGTH + 1
torch.manual_seed(42)
np.random.seed(42)
dummy_mel_np = np.random.randn(4, 1, N_MELS, n_tf).astype(np.float32)
dummy_mel_t = torch.from_numpy(dummy_mel_np)

# PyTorch inference
with torch.no_grad():
    clip_pt, frame_pt = model_pt(dummy_mel_t)
clip_pt_np = clip_pt.float().numpy()
frame_pt_np = frame_pt.float().numpy()
print(f"\\nPyTorch outputs:")
print(f"  clip_logit shape: {clip_pt_np.shape}, mean={clip_pt_np.mean():.4f}, std={clip_pt_np.std():.4f}")
print(f"  framewise shape: {frame_pt_np.shape}, mean={frame_pt_np.mean():.4f}, std={frame_pt_np.std():.4f}")

# ONNX inference
ort_out = sess.run(["clip_logit", "framewise"], {"mel": dummy_mel_np})
clip_onnx = ort_out[0]
frame_onnx = ort_out[1]
print(f"\\nONNX outputs:")
print(f"  clip_logit shape: {clip_onnx.shape}, mean={clip_onnx.mean():.4f}, std={clip_onnx.std():.4f}")
print(f"  framewise shape: {frame_onnx.shape}, mean={frame_onnx.mean():.4f}, std={frame_onnx.std():.4f}")

# Diff metrics
print(f"\\n=== DIFF (PyTorch vs ONNX) ===")
clip_diff = np.abs(clip_pt_np - clip_onnx)
frame_diff = np.abs(frame_pt_np - frame_onnx)
print(f"clip_logit:")
print(f"  max abs diff: {clip_diff.max():.6f}")
print(f"  mean abs diff: {clip_diff.mean():.6f}")
print(f"  rel err (mean): {(clip_diff.mean() / (np.abs(clip_pt_np).mean() + 1e-8)):.4%}")
print(f"framewise:")
print(f"  max abs diff: {frame_diff.max():.6f}")
print(f"  mean abs diff: {frame_diff.mean():.6f}")

# Correlation
corr_clip = np.corrcoef(clip_pt_np.flatten(), clip_onnx.flatten())[0, 1]
print(f"\\nclip_logit correlation: {corr_clip:.6f}")

# Verdict
print(f"\\n{'='*60}")
if clip_diff.max() < 1e-3:
    print(f"VERDICT: ★ ONNX is CORRECT (max diff {clip_diff.max():.2e} < 1e-3)")
    print(f"         M1 LB 0.81 is NOT due to ONNX bug")
    print(f"         Likely cause: fold variance, paradigm, or submission format")
elif clip_diff.max() < 0.1:
    print(f"VERDICT: ⚠ ONNX has SMALL DRIFT (max diff {clip_diff.max():.2e})")
    print(f"         Probably OK for inference, but check")
else:
    print(f"VERDICT: ★ ONNX is LIKELY BROKEN (max diff {clip_diff.max():.2e} > 0.1)")
    print(f"         M1 LB 0.81 is LIKELY due to ONNX export bug")
    print(f"         Need to re-export with different settings")
print(f"{'='*60}")
""")

# =============================================================
code("""# Cell 6: Speed bench (bonus)
import time

print("=== Speed bench (10 iters, batch 16) ===")
batch_np = np.random.randn(16, 1, N_MELS, n_tf).astype(np.float32)
batch_t = torch.from_numpy(batch_np)

# Warmup
with torch.no_grad():
    _ = model_pt(batch_t)
_ = sess.run(["clip_logit", "framewise"], {"mel": batch_np})

# Bench
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model_pt(batch_t)
pt_time = (time.time() - t0) / 10 * 1000

t0 = time.time()
for _ in range(10):
    _ = sess.run(["clip_logit", "framewise"], {"mel": batch_np})
onnx_time = (time.time() - t0) / 10 * 1000

print(f"  PyTorch: {pt_time:.1f} ms/iter")
print(f"  ONNX:    {onnx_time:.1f} ms/iter (speedup {pt_time/onnx_time:.2f}x)")
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
for i, c in enumerate(CELLS):
    src = "".join(c.get("source", []))
    head = src.split('\n')[0][:70] if src else "(empty)"
    print(f"  Cell {i:2d} ({c['cell_type']:8s}) {len(src):6d} chars | {head}")
