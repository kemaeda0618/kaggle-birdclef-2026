"""Build exp084 train R1 NB (Kaggle T4x2), based on exp083.

Differences vs exp083:
- N_FFT: 4096 → 2048
- HOP_LENGTH: 512 → 256 (★ time res 2x)
- WIN_LENGTH = N_FFT = 2048
- Input shape: (256, 313) → (256, 626) (time dim 2x)
- BATCH: 64 → 32 (T4 16GB safe with 2x input)
- LR: 3e-4 (no scaling, minor diff from exp083)
- Output paths/slugs: exp083 → exp084
"""
import json
import shutil
from pathlib import Path

# Setup exp084 dir
base_dir = Path("experiment/exp084/notebook")
base_dir.mkdir(parents=True, exist_ok=True)

# Copy exp083 R1 NB as base
src_nb = Path("experiment/exp083/notebook/nb_train_r1.ipynb")
dst_nb = base_dir / "nb_train_r1.ipynb"
shutil.copy(src_nb, dst_nb)

# Copy exp083 R1 push script
src_push = Path("experiment/exp083/notebook/_push_nb_train_r1.py")
dst_push = base_dir / "_push_nb_train_r1.py"
shutil.copy(src_push, dst_push)

# Modify NB
nb = json.loads(dst_nb.read_text(encoding="utf-8"))

for c in nb["cells"]:
    cid = c.get("id", "")
    src = "".join(c.get("source", []))

    # Header: exp083 → exp084
    if cid == "hdr":
        new_src = '''# exp084 — Self-resuming ConvNeXt-Pico SED training (Kaggle T4x2)

**Goal:** Train ConvNeXt-Pico SED with Perch v2 distillation + **Hi-time-res mel** (HOP=256),
25 epochs total, using Save & Run All sessions on Kaggle.

## ★ exp084 vs exp083 (base) の唯一の違い

**Mel spec: time-resolution 2x** (BC26 multi-taxa diversity 軸)

| Param | exp083 (Hi-freq) | **exp084 (Hi-time)** | 効果 |
|---|---|---|---|
| `N_FFT` | 4096 | **2048** | freq res down (Tucker と同) |
| `HOP_LENGTH` | 512 | **256** | **time res 2x (8ms/frame)** ★ |
| `WIN_LENGTH` | 4096 | **2048** | = N_FFT 規約 |
| `N_MELS` | 256 | 256 | 同 |
| `F_MIN` / `F_MAX` | 20 / 16000 | 同 | 同 |

**Input shape**: exp083 (256, 313) → **exp084 (256, 626)** (time dim 2x)
- backbone forward computation ~2x
- per-epoch time ~20-25 min (exp083 12 min の 2x)
- BATCH=32 (exp083 64 の半分、T4 16GB 安全)

## Diversity 軸 (exp015/083/084 trio)

| | exp015 (Tucker) | exp083 (Hi-freq) | **exp084 (Hi-time)** |
|---|---|---|---|
| N_FFT | 2048 | 4096 | 2048 |
| HOP | 512 | 512 | **256** |
| Strength | balanced | freq detail (insect harmonic) | **time detail (sharp event)** |

3-mel grid で time-freq trade-off を完全に carve out、ensemble bonus 期待。

## How self-resume works
- `maekeso/exp084-state` Dataset (R1 ckpts) で resume
- Per-session 11h budget で 25 epoch 完走 (1 session 想定、tight なら 2 sessions)

## Required input
- `birdclef-2026` (competition)
- `tuckerarrants/perch-v2-no-dft-onnx`
- `maekeso/exp084-state` (resume、初回 skip)
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell paths: exp083-state → exp084-state
    if cid == "paths":
        new_src = src.replace("exp083-state", "exp084-state")
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell config: change N_FFT, HOP, WIN_LENGTH, BATCH
    if cid == "config":
        new_src = '''# ============================================================
# Cell 3: Config — hyperparameters (★ exp084: HOP=256 のみ exp083 から diff、N_FFT も 2048 に戻す)
# ============================================================
NUM_CLASSES = 234
SR = 32000

# Duration / Mel ★ exp084: Hi-time-res mel
TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT          = 2048      # ★ exp084: 4096→2048 (Tucker と同、freq res down)
HOP_LENGTH     = 256       # ★ exp084: 512→256 (time res 2x、short event sharp)
N_MELS         = 256       # 同
FMIN           = 20        # 同
FMAX           = 16000     # 同
WIN_LENGTH     = N_FFT     # codebase convention (win_length = N_FFT)

# Model
BACKBONE = "convnext_pico.d1_in1k"

# Perch distillation
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

# Training
N_FOLDS = 5
FOLDS = [0]                  # 1-fold for now; expand later by editing this list
N_TOTAL_EPOCHS = 25
BATCH = 32                   # ★ exp084: 64→32 (T4 16GB safe with 2x time-dim input)
LR = 3e-4                    # 据え置き (exp083 と min diff)
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 2

# Augmentation
AUG_PROB = 0.5
AUG_GAIN_DB_RANGE = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

# Mixup
USE_MIXUP = True
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
MIXUP_HARD = False           # soft mixup (label blending)

# SpecAugment
FREQ_MASK_PARAM = 25
TIME_MASK_PARAM = 30
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

# Upsampling
MIN_SAMPLE = 20

# Source mixing
SHARES = {"focal": 0.85, "sc": 0.15}
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "sc": 1.0}

# Data loader
NUM_WORKERS = 4
PERSISTENT_WORKERS = True

print(f"Backbone: {BACKBONE}")
print(f"Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}, win_length={WIN_LENGTH}")
print(f"Input shape: (1, {N_MELS}, {TRAIN_SAMPLES // HOP_LENGTH + 1})")
print(f"Distill: ON (alpha={ALPHA_DISTILL})")
print(f"Batch: {BATCH} | Total epochs: {N_TOTAL_EPOCHS} | Folds: {FOLDS}")
print(f"LR: {LR} | WD: {WD} | warmup: {WARMUP_EPOCHS}ep")
print(f"Workers: {NUM_WORKERS}")
'''
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell resume_or_init: exp083-state → exp084-state in print
    if cid == "resume_or_init":
        new_src = src.replace("exp083-state", "exp084-state")
        c["source"] = new_src.splitlines(keepends=True)
        continue

    # Cell upload_state: exp083-state → exp084-state in slug/title
    if cid == "upload_state":
        new_src = src.replace("exp083-state", "exp084-state")
        new_src = new_src.replace("exp083 training state", "exp084 training state")
        new_src = new_src.replace("birdclef2026-exp083-train-r1", "birdclef2026-exp084-train-r1")
        c["source"] = new_src.splitlines(keepends=True)
        continue

dst_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {dst_nb}")

# Update push script
push_src = dst_push.read_text(encoding="utf-8")
push_src = push_src.replace("exp083", "exp084")
dst_push.write_text(push_src, encoding="utf-8")
print(f"OK Updated {dst_push}")

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
            print(f"    L{i+max(0, err_line-3)+1}: {ln[:150]}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
