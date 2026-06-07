"""Build exp083 R1 inference NB from exp017 nb_infer.ipynb (5s canonical paradigm).

Differences vs exp017:
- Backbone: eca_nfnet_l0 → convnext_pico.d1_in1k
- Mel: N_FFT 2048 → 4096 (Hi-freq), WIN_LENGTH=N_FFT explicit
- State source: birdclef2026-exp017-weights → exp083-state (Kaggle Dataset)
- ckpt search: R1 only (no r2_ prefix)
- internet=off (drop !pip install)
"""
import json
from pathlib import Path
import shutil

# Read exp017 nb_infer.ipynb as base
src_nb = Path("experiment/exp017/notebook/nb_infer.ipynb")
dst_nb = Path("experiment/exp083/notebook/nb_infer_r1.ipynb")
shutil.copy(src_nb, dst_nb)

nb = json.loads(dst_nb.read_text(encoding="utf-8"))

# Modify cells
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    cid = c.get("id", "")

    # Header: exp017 → exp083
    if cid == "hdr":
        new_src = '''# exp083 R1 — Inference (Kaggle CPU 90min, internet=off)

★ exp083 R1 = convnext_pico × 5s × Hi-freq mel (N_FFT=4096)
- val_ns22 0.9115 / val_macro 0.9115 (epoch 22 best)
- Per-taxon: Insecta 0.944 ★ / Aves 0.832 (Hi-freq mel の trade-off)

## Diagnostic purpose
R2 train 前に R1 単独 LB を確認、明らかな bug がないか様子見。
- Hi-freq mel が test 側で expected な per-class profile を出すか?
- Aves drag が val どおりか?

## Constraints
- Kaggle Code-Only competition: CPU only, no internet, 90 min total
- Submit format: `row_id` + 234 species columns

## Inputs
- `birdclef-2026` (competition)
- `maekeso/exp083-state` (R1 ckpts、ckpt_best_ns22.pth = val 0.9115)

## Pipeline (exp017 canonical 5s paradigm)
1. Locate ckpt under /kaggle/input
2. Rebuild convnext_pico BirdSEDModel (must match training)
3. Per test file: decode → 12 chunks (5s each) → mel → model → sigmoid
4. Blend in **LOGIT space**: 0.5*clip + 0.5*frame_max, gauss smooth on logits, sigmoid LAST
5. Write submission.csv

## ★ exp083 specific
- Mel: Hi-freq (N_FFT=4096, WIN_LENGTH=4096, HOP=512, N_MELS=256, FMIN=20, FMAX=16000)
- Backbone: convnext_pico.d1_in1k (~9M params, fast)
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell 1 (setup): drop !pip install (internet=off)
    if cid == "setup":
        new_src = '''# ============================================================
# Cell 1: Setup — internet=off, timm pre-installed in Kaggle
# ============================================================
import os, sys, time, json, math, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, torch: {torch.__version__}, timm: {timm.__version__}")
torch.set_num_threads(4)
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell 2 (paths): exp017-weights → exp083-state
    if cid == "paths":
        new_src = '''# ============================================================
# Cell 2: Paths — locate competition data + R1 ckpt
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"

TEST_DIR = BASE / "test_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
print(f"BASE: {BASE}")
print(f"  test_soundscapes exists: {TEST_DIR.exists()}")

# Locate exp083 R1 state Dataset
STATE_DIR = None
CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/exp083-state"),
    Path("/kaggle/input/exp083-state"),
]
for p in CANDIDATES:
    if p.exists():
        if any(p.rglob("ckpt_best_*.pth")) or any(p.rglob("ckpt_latest*.pth")):
            STATE_DIR = p; break

if STATE_DIR is None:
    for hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        STATE_DIR = hit.parent; break

assert STATE_DIR is not None, (
    "ckpt not found. Attach maekeso/exp083-state as dataset_sources"
)
print(f"State dir: {STATE_DIR}")
for f in sorted(STATE_DIR.rglob("ckpt_*.pth"))[:6]:
    print(f"  {f.relative_to(STATE_DIR)!s}  {f.stat().st_size/1e6:.2f} MB")
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell 3 (config): mel + backbone for exp083
    if cid == "config":
        new_src = '''# ============================================================
# Cell 3: Config — must match exp083 R1 training
# ============================================================
NUM_CLASSES = 234
SR = 32000

TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION

# ★ exp083 Hi-freq mel
N_FFT      = 4096      # ★ exp083: 2048→4096 (freq res 2x)
HOP_LENGTH = 512       # 同 (output time frame 同じ)
N_MELS     = 256
FMIN       = 20
FMAX       = 16000
WIN_LENGTH = N_FFT     # ★ codebase convention

BACKBONE = "convnext_pico.d1_in1k"

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES

print(f"Backbone: {BACKBONE}")
print(f"Mel: n_fft={N_FFT}, hop={HOP_LENGTH}, win_length={WIN_LENGTH}, n_mels={N_MELS}")
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell 4 (model): Add WIN_LENGTH to MelSpecTransform + R1 ckpt search
    if cid == "model":
        new_src = '''# ============================================================
# Cell 4: Model — rebuild convnext_pico SED architecture (Hi-freq mel)
# ============================================================
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,    # ★ exp083: explicit (= N_FFT 規約)
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


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


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


# Load best R1 ckpt (★ exp083 R1: no r2_ prefix needed)
CKPT_PRIORITY = [
    "ckpt_best_ns22.pth",     # R1 best ns22 (val 0.9115)
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]
ckpt_path = None
for name in CKPT_PRIORITY:
    hits = list(STATE_DIR.rglob(name))
    if hits:
        ckpt_path = hits[0]; break
assert ckpt_path is not None, f"No ckpt found under {STATE_DIR}"
print(f"Loading ckpt: {ckpt_path.name}")

try:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(ckpt_path), map_location="cpu")
print(f"  epoch={state.get('epoch')}, "
      f"best_ns22={state.get('best_ns22', float('nan')):.4f}, "
      f"best_macro={state.get('best_macro', float('nan')):.4f}")

model = BirdSEDModel().to(device)
msg = model.load_state_dict(state["model_state"], strict=False)
model.eval()
print(f"  Load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
print(f"OK model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

# Save
dst_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {dst_nb}")

# Build push script
push_script = '''"""Push exp083 R1 inference NB (CPU sub)."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_infer_r1.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp083-infer-r1"
TITLE = "birdclef2026 exp083 infer r1"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/exp083-state",
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "")
    if err: print("Error:", err)
'''

push_path = Path("experiment/exp083/notebook/_push_nb_infer_r1.py")
push_path.write_text(push_script, encoding="utf-8")
print(f"OK Built {push_path}")

# Syntax check
import ast
nb = json.loads(dst_nb.read_text(encoding="utf-8"))
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
