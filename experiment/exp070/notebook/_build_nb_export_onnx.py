"""Build ONNX export NB for M1 (Colab).

Workflow:
  1. Mount Drive
  2. Auth Kaggle
  3. Load 5 M1 ckpts from Drive
  4. Build model architecture (BirdSEDModel without DistillHead)
  5. Export each fold to ONNX (mel input → clip_logit + framewise)
  6. Upload to Kaggle Dataset `maekeso/birdclef2026-exp070-m1-onnx`
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_export_onnx_m1.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# exp070 M1 ONNX Export (Colab)

**Goal**: M1 5-fold ckpts (.pth) → ONNX (.onnx) → Kaggle Dataset
- Mel input → clip_logit + framewise outputs (DistillHead 除外)
- ONNX runtime CPU で 2-3x speedup at inference
- Upload to `maekeso/birdclef2026-exp070-m1-onnx`
""")

# =============================================================
code("""# Cell 1: Setup
!pip install -q timm onnx onnxruntime kaggle 2>&1 | tail -2

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, torch {torch.__version__}, timm {timm.__version__}")

# Model architecture (must match M1 train NB)
NUM_CLASSES = 234
SR = 32000
CHUNK_SAMPLES = SR * 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000
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
        # Return clip + framewise (permuted to (B, T, num_classes))
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
code("""# Cell 4: Export each fold to ONNX
import time

ONNX_OUT_DIR = Path("/content/m1_onnx")
ONNX_OUT_DIR.mkdir(exist_ok=True)

# Dummy mel input for ONNX trace
# Shape: (B=1, 1, N_MELS=256, T=314)
n_tf = CHUNK_SAMPLES // HOP_LENGTH + 1
print(f"ONNX input shape: (B, 1, {N_MELS}, {n_tf})")

for fi, ckpt_path in enumerate(ckpt_paths):
    t0 = time.time()
    print(f"\\n=== Fold {fi}: {ckpt_path.name} ===")

    model = BirdSEDModel()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"  Load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    model.eval()

    # ONNX export (dynamo=False per [[feedback_kaggle_onnx_offline_install]])
    dummy = torch.randn(1, 1, N_MELS, n_tf)
    onnx_path = ONNX_OUT_DIR / f"m1_fold{fi}.onnx"
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
    print(f"  ONNX exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f}MB) in {time.time()-t0:.1f}s")

print(f"\\nAll {len(ckpt_paths)} ONNX files in {ONNX_OUT_DIR}")
for f in sorted(ONNX_OUT_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f}MB")
""")

# =============================================================
code("""# Cell 5: Sanity check - load ONNX and predict on dummy
import onnxruntime as ort

print("=== ONNX sanity check ===")
sample_onnx = ONNX_OUT_DIR / "m1_fold0.onnx"
sess = ort.InferenceSession(str(sample_onnx), providers=["CPUExecutionProvider"])

print(f"Inputs: {[i.name for i in sess.get_inputs()]}")
print(f"Outputs: {[o.name for o in sess.get_outputs()]}")

dummy_mel = np.random.randn(2, 1, N_MELS, n_tf).astype(np.float32)
outputs = sess.run(["clip_logit", "framewise"], {"mel": dummy_mel})
print(f"clip_logit shape: {outputs[0].shape}  (expect [2, 234])")
print(f"framewise shape: {outputs[1].shape}  (expect [2, T, 234])")

# Sigmoid check
clip_sigmoid = 1.0 / (1.0 + np.exp(-outputs[0]))
print(f"clip_sigmoid stats: mean={clip_sigmoid.mean():.4f}, max={clip_sigmoid.max():.4f}")
print(f"OK ONNX inference works")
""")

# =============================================================
code("""# Cell 6: Upload to Kaggle Dataset
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp070-m1-onnx"
TITLE = "birdclef2026 exp070 m1 onnx"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_staged = 0
    for f in sorted(ONNX_OUT_DIR.glob("*.onnx")):
        shutil.copy2(str(f), str(td / f.name))
        n_staged += 1
    print(f"Staged {n_staged} ONNX files")

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
            api.dataset_create_version(folder=str(td), version_notes="M1 5-fold ONNX",
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
print("ONNX export done. Terminating Colab...")
import time as _t; _t.sleep(3)
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
