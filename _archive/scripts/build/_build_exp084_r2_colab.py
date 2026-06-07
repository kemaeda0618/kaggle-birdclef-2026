"""Build exp084 R2 Colab NB based on exp083 R2 Colab NB.

Changes vs exp083 R2:
- Mel: Hi-freq (N_FFT=4096, HOP=512) → Hi-time (N_FFT=2048, HOP=256)
- Input time frames: 313 → 626 (2x)
- BATCH: 96 → 64 (2x input → reduce batch for VRAM safety)
- Output paths: exp083 → exp084
- Kaggle Dataset slug: exp083 → exp084

NOTE: Requires Drive R1 pseudo CSV at output/exp084/r1-pseudo/pseudo_labels.csv
      (Generate exp084 R1 pseudo separately if missing.)
"""
import json
import shutil
from pathlib import Path

src_nb = Path("experiment/exp083/notebook/nb_train_r2_colab.ipynb")
dst_dir = Path("experiment/exp084/notebook")
dst_dir.mkdir(parents=True, exist_ok=True)
dst_nb = dst_dir / "nb_train_r2_colab.ipynb"
shutil.copy(src_nb, dst_nb)

nb = json.loads(dst_nb.read_text(encoding="utf-8"))

for c in nb["cells"]:
    src_text = "".join(c.get("source", []))
    cid = c.get("id", "")

    # Header
    if cid == "hdr":
        new_src = '''# exp084 R2 — convnext_pico × 5s Hi-time mel × Multi-Iter Noisy Student (Colab Pro L4)

**exp084 R2 = R1 (val 0.9125) + R1 pseudo distill**

## Changes from exp083 R2 (base)
- Mel: Hi-freq (4096/512/256/20) → **Hi-time (2048/256/256/20)** ★
- Input time frames: 313 → **626** ★ (2x)
- BATCH: 96 → **64** (2x input → reduce batch for VRAM)
- Output: `output/exp084/r2/`, Kaggle slug `birdclef2026-exp084-weights`

## ★ Mel diversity 軸 (3 mel grid)

| | exp015 (Tucker) | exp083 (Hi-freq) | **exp084 (Hi-time)** |
|---|---|---|---|
| N_FFT | 2048 | 4096 | 2048 |
| HOP | 512 | 512 | **256** |
| Time frames (5s) | 313 | 313 | **626** |
| Strength | balanced | freq detail (insect) | **time detail (sharp event)** |

## 不変項目 (exp083 R2 から)
- Backbone: convnext_pico.d1_in1k
- TRAIN_DURATION: 5s
- LR: 3.7e-4 (sqrt scale base、BATCH=64 で実質 3.0e-4 程度)
- N_TOTAL_EPOCHS: 20
- SHARES: focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20
- ALPHA_DISTILL: 1.0、Perch v2 distill
- LabeledSC aggregation: なし (5s native)
- Perch distill: 1× 5s direct (no averaging)

## 想定時間 (Colab L4 24GB)
- DL data: 5-15 min
- R2 train: 20 epoch × ~12-18 min/ep = ~4-6h (input 2x で BATCH 半分、ep 時間ほぼ同等)
- Pseudo gen: ~15 min (5s native)
- Drive mirror + Kaggle upload: 5-10 min
- **合計 ~5-7h、Colab L4 1 session**

## 期待
- val_ns22: 0.93-0.94 (R1 0.9125 → R2 +0.02 期待)
- LB R2 single: 0.90-0.92
- ensemble bonus: time-res diversity で +0.003-0.008

## Drive 必須 assets (NB 開始前確認)
```
/content/drive/MyDrive/kaggle/birdclef2026/
├── output/exp084/
│   ├── r1/                    ← R1 ckpts (Drive 上に必要)
│   │   ├── ckpt_best_ns22.pth (118MB)
│   │   ├── ckpt_best_macro.pth
│   │   ├── ckpt_latest.pth
│   │   └── history.json
│   └── r1-pseudo/             ← R1 pseudo CSV (Drive 上に必要)
│       └── pseudo_labels.csv  (★ assert fails if missing)
├── kaggle.json
└── perch-onnx/perch_v2*.onnx
```

⚠️ exp084 R1 pseudo がまだ生成されていない場合:
- 別途 Kaggle で exp084 R1 pseudo gen NB を run (exp083 と同様の pattern)
- または Colab 上で R1 ckpt → train_soundscapes 推論で生成

## Output
- `r2/ckpt_best_ns22.pth` + その他 (Drive + Kaggle Dataset)
- `r2-pseudo/pseudo_labels.csv` (R3 用、optional)
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell setup — paths exp083 → exp084
    if cid == "setup":
        new_src = src_text.replace("exp083", "exp084")
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell config — change Hi-freq mel → Hi-time mel + BATCH
    if cid == "config":
        new_src = '''# ============================================================
# Cell 3: Imports + Config + Paths (★ exp084: 5s Hi-time mel + convnext_pico)
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
TRAIN_DURATION = 5    # 5s
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT          = 2048   # ★ exp084: Hi-time (was 4096 for exp083 Hi-freq)
HOP_LENGTH     = 256    # ★ exp084: 256 (was 512 for exp083、time res 2x)
N_MELS         = 256    # 同
FMIN           = 20     # 同
FMAX           = 16000
WIN_LENGTH     = N_FFT  # codebase convention (= N_FFT = 2048)

BACKBONE = "convnext_pico.d1_in1k"

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

N_FOLDS = 5
FOLDS = [0]
FORCE_FRESH = True       # R2 = warm start from R1 ckpt (cell 11)
N_TOTAL_EPOCHS = 20
BATCH = 64               # ★ exp084: 96→64 (input time 2x で VRAM 半分)
LR = 3.0e-4              # ★ BATCH=64 → LR 標準 (no scaling)
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
print(f"Time frames (5s): {TRAIN_SAMPLES // HOP_LENGTH + 1}")
print(f"Duration: {TRAIN_DURATION}s, samples={TRAIN_SAMPLES}")
print(f"Batch: {BATCH} | LR: {LR} | Epochs: {N_TOTAL_EPOCHS} | Folds: {FOLDS}")
print(f"SHARES: {SHARES}")
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell upload_state — exp083 → exp084 in slug/title
    if cid == "upload_state":
        src_new = src_text.replace("birdclef2026-exp083-weights",
                                    "birdclef2026-exp084-weights")
        src_new = src_new.replace("birdclef2026 exp083 weights",
                                    "birdclef2026 exp084 weights")
        src_new = src_new.replace("exp083", "exp084")
        c["source"] = src_new.splitlines(keepends=True)
        continue

    # All other cells: generic exp083 → exp084 replacement
    if "exp083" in src_text:
        new_src = src_text.replace("exp083", "exp084")
        c["source"] = new_src.splitlines(keepends=True)

dst_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {dst_nb}")

# Verify
nb = json.loads(dst_nb.read_text(encoding="utf-8"))
n_exp083 = sum(1 for c in nb["cells"] if "exp083" in "".join(c.get("source", [])))
print(f"Remaining 'exp083' references in cells: {n_exp083}")
if n_exp083 > 0:
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "exp083" in src:
            cid = c.get("id", "?")
            for ln in src.splitlines():
                if "exp083" in ln:
                    print(f"  cell {cid}: {ln.strip()[:150]}")

# Syntax check
import ast
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
