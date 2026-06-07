"""Build exp083 R2 Colab NB based on exp082 R2 (Colab pattern).

Changes vs exp082:
- 20s Babych → 5s Hi-freq mel (N_FFT=4096, HOP=512, N_MELS=256, FMIN=20)
- BACKBONE: eca_nfnet_l0 → convnext_pico.d1_in1k
- TRAIN_DURATION 20 → 5
- LabeledSC 4x aggregation removed (5s native, no aggregation needed)
- Perch distill 4x averaging removed (5s native, 1 forward = 1 embedding)
- Pseudo gen N_WINDOWS: 3 → 12 (60s/5s=12)
- Output paths: exp082 → exp083
- Kaggle Dataset slug: exp082 → exp083
"""
import json
import shutil
from pathlib import Path

# Copy exp082 R2 NB as base
src_nb = Path("experiment/exp082/notebook/nb_train_r2.ipynb")
dst_nb = Path("experiment/exp083/notebook/nb_train_r2_colab.ipynb")
shutil.copy(src_nb, dst_nb)

nb = json.loads(dst_nb.read_text(encoding="utf-8"))

# Replace cell-level changes
for c in nb["cells"]:
    src_text = "".join(c.get("source", []))
    cid = c.get("id", "")

    # Header
    if cid == "hdr":
        new_src = '''# exp083 R2 — convnext_pico × 5s Hi-freq mel × Multi-Iter Noisy Student (Colab Pro L4)

**exp083 R2 = R1 (val 0.9115、LB 0.886) + R1 pseudo distill**

## Changes from exp082 R2 (base)
- `TRAIN_DURATION` / `VAL_DURATION`: 20 → **5**
- Mel spec: Babych (224/4096/1252/0) → **Hi-freq (256/4096/512/20)**
  - 5s × 32000 / 512 = 313 frames
- `N_WINDOWS` in pseudo_infer: 3 → **12** (60s / 5s = 12 windows/file)
- LabeledSC aggregation: 4× 5s → 1× 5s = **no aggregation** (1:1)
- Perch distill: 4× 5s averaged → **1× 5s** (native 5s)
- BACKBONE: eca_nfnet_l0 → **convnext_pico.d1_in1k**
- Output: `output/exp083/r2/`, Kaggle slug `birdclef2026-exp083-weights`

## 不変項目
- Init: R1 ckpt warm start
- LR 3e-4 (ConvNeXt 適用)
- Batch: 96 (Colab L4 24GB、convnext_pico 軽量で余裕)
- N_TOTAL_EPOCHS 20 (R2 standard)
- ALPHA_DISTILL 1.0、Perch v2 distill
- SHARES focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20

## 想定時間 (Colab L4)
- DL data: 5-15 min (初回)
- R1 ckpt + pseudo CSV: Drive から copy ~1 min
- R2 train: 20 epoch × ~10-15 min/ep = ~3-5h
- Pseudo gen: ~10 min
- Drive mirror + Kaggle Dataset upload: 5-10 min
- **合計 ~4-6h、Colab L4 1 session 完走**

## 期待
- val_ns22: 0.93-0.94 (R1 0.9115 → R2 +0.02 lift)
- LB R2 single: 0.91-0.92
- ensemble bonus: Hi-freq mel diversity で +0.003-0.008
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell setup — paths exp082 → exp083
    if cid == "setup":
        new_src = src_text.replace("exp082", "exp083")
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell config — major rewrite (mel + duration + backbone)
    if cid == "config":
        new_src = '''# ============================================================
# Cell 3: Imports + Config + Paths (★ exp083: 5s Hi-freq mel + convnext_pico)
# ============================================================
import os, time, json, gc, random, math
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

# ===== Paths =====
BASE = LOCAL_DATA / "competition"
TA_DIR = LOCAL_DATA / "train_audio"
TS_DIR = LOCAL_DATA / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
LABELS_PATH = BASE / "train_soundscapes_labels.csv"
PERCH_PATH = LOCAL_DATA / "perch-onnx" / "perch_v2_no_dft.onnx"
PSEUDO_CSV_PATH = LOCAL_DATA / "r1-pseudo" / "pseudo_labels.csv"
OUT_DIR = LOCAL_OUT
print(f"BASE: {BASE}")
print(f"TA_DIR: {TA_DIR.exists()}, TS_DIR: {TS_DIR.exists()}")
print(f"PERCH: {PERCH_PATH.exists()}, PSEUDO: {PSEUDO_CSV_PATH.exists()}")

# ===== Reproducibility =====
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ===== Config =====
NUM_CLASSES = 234
SR = 32000
TRAIN_DURATION = 5    # ★ exp083: 20→5
VAL_DURATION   = 5    # ★ exp083: 20→5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT          = 4096   # ★ exp083: Hi-freq mel (R1 と同)
HOP_LENGTH     = 512    # ★ exp083: 1252→512 (5s で 313 frames)
N_MELS         = 256    # ★ exp083: 224→256
FMIN           = 20     # ★ exp083: 0→20
FMAX           = 16000
WIN_LENGTH     = N_FFT  # codebase convention

BACKBONE = "convnext_pico.d1_in1k"   # ★ exp083: convnext_pico

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

N_FOLDS = 5
FOLDS = [0]
FORCE_FRESH = True       # ★ R2 = fresh init (warm start from R1 ckpt in cell 11)
# ★ exp083 R1 was BATCH=64 LR=3e-4 (Kaggle T4)。Colab L4 24GB なら BATCH=96 で sqrt scale LR
N_TOTAL_EPOCHS = 20      # R2 standard
BATCH = 96               # ★ convnext_pico 軽量、Colab L4 で余裕
LR = 3.7e-4              # ★ sqrt scaling (3e-4 × sqrt(96/64))、convnext は scale 適用可
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 3

# Augmentation
AUG_PROB = 0.5
AUG_GAIN_DB_RANGE = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

USE_MIXUP = True
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
MIXUP_HARD = False

FREQ_MASK_PARAM = 25
TIME_MASK_PARAM = 30
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

MIN_SAMPLE = 20

# Source mixing (R2: 3 sources)
SHARES = {"focal": 0.70, "labeled_sc": 0.10, "pseudo_sc": 0.20}
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "labeled_sc": 1.0, "pseudo_sc": 1.0}

NUM_WORKERS = 4
PERSISTENT_WORKERS = True

TRAIN_START = time.time()
MAX_RUNTIME_SEC = 22.0 * 3600

print(f"Backbone: {BACKBONE}")
print(f"Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}, win_length={WIN_LENGTH}")
print(f"Duration: {TRAIN_DURATION}s, samples={TRAIN_SAMPLES}")
print(f"Batch: {BATCH} | LR: {LR} | Epochs: {N_TOTAL_EPOCHS} | Folds: {FOLDS}")
print(f"SHARES: {SHARES}")
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell load_data — remove LabeledSC 4x aggregation (5s = no aggregation)
    if cid == "load_data":
        # Replace the 4x aggregation block with no-op (5s native)
        src_new = src_text
        # Remove the aggregation block
        old_agg = src_text
        # Find aggregation start and end
        if "# ★ exp082: Aggregate labeled SS 5s -> 20s windows" in src_text or "[exp082] Aggregating labeled SS" in src_text:
            # Replace the entire 4x agg block with passthrough
            old_lines = src_text.splitlines()
            new_lines = []
            in_agg_block = False
            for ln in old_lines:
                if "# ★ exp082: Aggregate labeled SS" in ln or "# ★ exp081: Aggregate labeled SS" in ln:
                    in_agg_block = True
                    new_lines.append("    # ★ exp083: 5s native = no LabeledSC aggregation (1:1)")
                    continue
                if in_agg_block:
                    # Detect end of aggregation block (next ungrouped section)
                    if "sc_files = sc_meta[" in ln or (ln.startswith("    sc_files") and "drop_duplicates" in ln):
                        in_agg_block = False
                        new_lines.append(ln)
                        continue
                    continue
                new_lines.append(ln)
            src_new = "\n".join(new_lines)
        # exp082 → exp083
        src_new = src_new.replace("exp082", "exp083")
        c["source"] = src_new.splitlines(keepends=True)
        continue

    # Cell train_loop — simplify Perch distill (5s native, no 4x averaging)
    if cid == "train_loop":
        src_new = src_text
        # Replace Perch 4x averaging block
        old = '''            if USE_PERCH_DISTILL and perch_teacher is not None:
                # ★ exp082 Perch distill: 4x 5s averaged for 20s alignment
                with torch.no_grad():
                    wav_full = wav.squeeze(1)
                    N = wav_full.shape[1]
                    if N == 160000:
                        # 5s legacy
                        perch_emb = perch_teacher.embed(wav_full).to(device)
                    elif N == 320000:
                        # 10s: 2x 5s averaged
                        emb_a = perch_teacher.embed(wav_full[:, :160000]).to(device)
                        emb_b = perch_teacher.embed(wav_full[:, 160000:]).to(device)
                        perch_emb = (emb_a + emb_b) * 0.5
                    elif N == 640000:
                        # 20s: 4x 5s Perch -> averaged
                        emb_a = perch_teacher.embed(wav_full[:, :160000]).to(device)
                        emb_b = perch_teacher.embed(wav_full[:, 160000:320000]).to(device)
                        emb_c = perch_teacher.embed(wav_full[:, 320000:480000]).to(device)
                        emb_d = perch_teacher.embed(wav_full[:, 480000:]).to(device)
                        perch_emb = (emb_a + emb_b + emb_c + emb_d) * 0.25
                    elif N > 160000:
                        start_off = (N - 160000) // 2
                        perch_emb = perch_teacher.embed(wav_full[:, start_off:start_off + 160000]).to(device)
                    else:
                        wav_pad = F.pad(wav_full, (0, 160000 - N))
                        perch_emb = perch_teacher.embed(wav_pad).to(device)'''
        new = '''            if USE_PERCH_DISTILL and perch_teacher is not None:
                # ★ exp083 Perch distill: 5s native = direct embed, no averaging
                with torch.no_grad():
                    wav_5s = wav.squeeze(1)
                    N = wav_5s.shape[1]
                    if N > 160000:
                        start_off = (N - 160000) // 2
                        wav_5s = wav_5s[:, start_off:start_off + 160000]
                    elif N < 160000:
                        wav_5s = F.pad(wav_5s, (0, 160000 - N))
                    perch_emb = perch_teacher.embed(wav_5s).to(device)'''
        if old in src_new:
            src_new = src_new.replace(old, new)
        src_new = src_new.replace("exp082", "exp083")
        c["source"] = src_new.splitlines(keepends=True)
        continue

    # Cell pseudo_setup — N_WINDOWS=12 (60s / 5s)
    if cid == "pseudo_setup":
        src_new = src_text.replace("N_WINDOWS    = 3    # exp082: 60s / 20s = 3",
                                    "N_WINDOWS    = 12   # exp083: 60s / 5s = 12")
        src_new = src_new.replace("exp082", "exp083")
        c["source"] = src_new.splitlines(keepends=True)
        continue

    # Cell upload_state — exp082 → exp083 in slug/title
    if cid == "upload_state":
        src_new = src_text.replace("birdclef2026-exp082-weights",
                                    "birdclef2026-exp083-weights")
        src_new = src_new.replace("birdclef2026 exp082 weights",
                                    "birdclef2026 exp083 weights")
        src_new = src_new.replace("exp082", "exp083")
        c["source"] = src_new.splitlines(keepends=True)
        continue

    # Other cells: generic exp082 → exp083 replacement
    if "exp082" in src_text:
        new_src = src_text.replace("exp082", "exp083")
        c["source"] = new_src.splitlines(keepends=True)

dst_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {dst_nb}")

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
        err_line = e.lineno or 1
        for i, ln in enumerate(clean.splitlines()[max(0, err_line-3):err_line+2]):
            print(f"    L{max(0, err_line-3)+i+1}: {ln[:150]}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
