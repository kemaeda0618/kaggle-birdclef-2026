"""Build multi-format export NB for M1 (Colab).

Exports M1 5-fold ckpts to ALL 3 formats:
  - .pth (re-training/SWA 用)
  - .onnx (ONNX runtime fallback)
  - .xml + .bin (OpenVINO IR、★ inference 最速)

Upload all 3 formats to single Kaggle Dataset:
  `maekeso/birdclef2026-exp070-m1-multi-fmt`
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_export_multi_format_m1.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# exp070 M1 Multi-Format Export (Colab)

**Goal**: M1 5-fold を **3 形式すべて** に export → 統合 Kaggle Dataset

- `.pth` (re-training / SWA 用)
- `.onnx` (ONNX runtime fallback)
- **`.xml + .bin` (OpenVINO IR、Kaggle CPU 最速 inference)** ★

**Target Dataset**: `maekeso/birdclef2026-exp070-m1-multi-fmt`

[[feedback_train_nb_export_pth_onnx]] 3 形式 export 規約準拠
""")

# =============================================================
code("""# Cell 1: Setup
!pip install -q timm onnx onnxruntime openvino kaggle 2>&1 | tail -3

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time
from pathlib import Path

DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_EXP_DIR   = DRIVE_INPUT_DIR / "output" / "exp070"
assert DRIVE_EXP_DIR.exists(), f"Drive exp dir missing: {DRIVE_EXP_DIR}"

# Kaggle auth
KJ_CANDIDATES = [DRIVE_INPUT_DIR / "kaggle.json",
                  Path("/content/drive/MyDrive/kaggle.json")]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
if KJ is not None:
    KAGGLE_CFG = Path.home() / ".kaggle"
    KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
    os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
    creds = json.loads(KJ.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]
    print(f"kaggle.json: {KJ}")
""")

# =============================================================
code("""# Cell 2: Imports + model arch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

import onnx
import onnxruntime as ort
import openvino as ov

print(f"torch: {torch.__version__}, timm: {timm.__version__}")
print(f"onnx: {onnx.__version__}, ort: {ort.__version__}, openvino: {ov.__version__}")

DEVICE = torch.device("cpu")  # export needs CPU consistency

# Model spec (M1)
NUM_CLASSES = 234
SR = 32000
CHUNK_SAMPLES = SR * 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
BACKBONE = "eca_nfnet_l1"
PERCH_EMBED_DIM = 1536
HIDDEN_DIM = 512
DROP_PATH = 0.0


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
        self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x):
        # x: (B, 1, N_MELS, T) - mel spec input
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
code("""# Cell 3: Locate M1 ckpts
ckpt_paths = []
for fold_k in range(5):
    fold_dir = DRIVE_EXP_DIR / f"fold{fold_k}" / "r3"
    ckpt = fold_dir / "ckpt_best_ns22.pth"
    if ckpt.exists():
        ckpt_paths.append(ckpt)
        print(f"  fold {fold_k}: {ckpt} ({ckpt.stat().st_size/1e6:.1f}MB)")
    else:
        print(f"  fold {fold_k}: MISSING {ckpt}")

assert len(ckpt_paths) > 0, "No M1 ckpts found"
print(f"\\nFound {len(ckpt_paths)} ckpts")
""")

# =============================================================
code("""# Cell 4: Export per fold to 3 formats
import time

OUT_DIR = Path("/content/m1_multi_fmt")
OUT_DIR.mkdir(exist_ok=True)

n_tf = CHUNK_SAMPLES // HOP_LENGTH + 1
print(f"Input shape: (B, 1, {N_MELS}, {n_tf})")

for fi, ckpt_path in enumerate(ckpt_paths):
    t0 = time.time()
    print(f"\\n{'='*60}\\n=== Fold {fi}: {ckpt_path.name} ===\\n{'='*60}")

    # 1. Copy .pth
    pth_path = OUT_DIR / f"m1_fold{fi}_ckpt_best.pth"
    shutil.copy2(str(ckpt_path), str(pth_path))
    print(f"  [1/3] .pth copied: {pth_path.name} ({pth_path.stat().st_size/1e6:.1f}MB)")

    # 2. Build model and export to ONNX
    model = BirdSEDModel()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"  Load state: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    model.eval()

    dummy = torch.randn(1, 1, N_MELS, n_tf)
    onnx_path = OUT_DIR / f"m1_fold{fi}.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["mel"], output_names=["clip_logit", "framewise"],
        dynamic_axes={
            "mel": {0: "batch"},
            "clip_logit": {0: "batch"},
            "framewise": {0: "batch"},
        },
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  [2/3] .onnx exported: {onnx_path.name} ({onnx_path.stat().st_size/1e6:.1f}MB)")

    # 3. Convert ONNX → OpenVINO IR
    ov_model = ov.convert_model(str(onnx_path))
    ov_xml_path = OUT_DIR / f"m1_fold{fi}.xml"
    ov.save_model(ov_model, str(ov_xml_path))
    # .xml + .bin generated
    ov_bin_path = OUT_DIR / f"m1_fold{fi}.bin"
    assert ov_xml_path.exists() and ov_bin_path.exists()
    print(f"  [3/3] OpenVINO IR: {ov_xml_path.name} + {ov_bin_path.name} ({ov_bin_path.stat().st_size/1e6:.1f}MB)")

    print(f"  Total fold {fi} time: {time.time()-t0:.1f}s")
    del model; del ov_model

print(f"\\n{'='*60}\\nAll {len(ckpt_paths)} folds exported to 3 formats\\n{'='*60}")
print(f"\\nFiles in {OUT_DIR}:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f}MB")
""")

# =============================================================
code("""# Cell 5: Sanity check - load all 3 formats and verify
import numpy as np

print("=== Sanity check: load all 3 formats from fold 0 ===")
dummy_mel_np = np.random.randn(2, 1, N_MELS, n_tf).astype(np.float32)
dummy_mel_t = torch.from_numpy(dummy_mel_np)

# 1. PyTorch .pth
print("\\n[1] PyTorch .pth")
model = BirdSEDModel()
state = torch.load(str(OUT_DIR / "m1_fold0_ckpt_best.pth"), map_location="cpu", weights_only=False)
model.load_state_dict(state.get("model_state", state), strict=False)
model.eval()
with torch.no_grad():
    clip_pt, frame_pt = model(dummy_mel_t)
print(f"  clip_logit shape: {clip_pt.shape}, mean={clip_pt.float().mean():.4f}")

# 2. ONNX runtime
print("\\n[2] ONNX runtime")
sess = ort.InferenceSession(str(OUT_DIR / "m1_fold0.onnx"), providers=["CPUExecutionProvider"])
out = sess.run(["clip_logit", "framewise"], {"mel": dummy_mel_np})
clip_onnx = out[0]
print(f"  clip_logit shape: {clip_onnx.shape}, mean={clip_onnx.mean():.4f}")

# 3. OpenVINO
print("\\n[3] OpenVINO runtime")
core = ov.Core()
compiled = core.compile_model(model=str(OUT_DIR / "m1_fold0.xml"), device_name="CPU")
ov_output = compiled([dummy_mel_np])
ov_keys = list(ov_output.keys())
clip_ov = ov_output[ov_keys[0]]
print(f"  outputs: {[str(k) for k in ov_keys]}")
print(f"  clip_logit shape: {clip_ov.shape}, mean={clip_ov.mean():.4f}")

# Compare outputs
print("\\n[Diff check] (should be very small)")
pt_np = clip_pt.float().numpy()
print(f"  PyTorch vs ONNX:    max abs diff = {np.abs(pt_np - clip_onnx).max():.6f}")
print(f"  PyTorch vs OpenVINO: max abs diff = {np.abs(pt_np - clip_ov).max():.6f}")
print(f"  ONNX vs OpenVINO:   max abs diff = {np.abs(clip_onnx - clip_ov).max():.6f}")

# Quick speed bench
import time
print("\\n[Speed bench] (10 inferences of batch 16)")
batch_np = np.random.randn(16, 1, N_MELS, n_tf).astype(np.float32)
batch_t = torch.from_numpy(batch_np)

# PyTorch
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(batch_t)
print(f"  PyTorch: {(time.time()-t0)/10*1000:.1f} ms/iter")

# ONNX
t0 = time.time()
for _ in range(10):
    _ = sess.run(["clip_logit", "framewise"], {"mel": batch_np})
print(f"  ONNX:    {(time.time()-t0)/10*1000:.1f} ms/iter")

# OpenVINO
t0 = time.time()
for _ in range(10):
    _ = compiled([batch_np])
print(f"  OpenVINO: {(time.time()-t0)/10*1000:.1f} ms/iter")

print("\\nOK all 3 formats verified")
""")

# =============================================================
code("""# Cell 6: Upload to Kaggle Dataset
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp070-m1-multi-fmt"
TITLE = "birdclef2026 exp070 m1 multi fmt"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_staged = 0
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            shutil.copy2(str(f), str(td / f.name))
            n_staged += 1
    print(f"Staged {n_staged} files")
    total_size = sum(f.stat().st_size for f in td.iterdir()) / 1e9
    print(f"Total size: {total_size:.2f} GB")

    meta = {"title": TITLE, "id": f"{USER}/{SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        api.dataset_list_files(f"{USER}/{SLUG}")
        exists = True
    except Exception:
        exists = False

    try:
        if exists:
            api.dataset_create_version(folder=str(td), version_notes="M1 5-fold pth+onnx+ov IR",
                                        dir_mode="zip", quiet=False)
            print("OK new version uploaded")
        else:
            api.dataset_create_new(folder=str(td), public=False,
                                    dir_mode="zip", quiet=False)
            print("OK new dataset created")
    except Exception as e:
        print(f"[UPLOAD ERROR] {type(e).__name__}: {str(e)[:400]}")

print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
""")

# =============================================================
code("""# Cell 7: auto-disconnect
print("Multi-format export done. Terminating Colab in 5s...")
import time as _t; _t.sleep(5)
from google.colab import runtime
runtime.unassign()
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
