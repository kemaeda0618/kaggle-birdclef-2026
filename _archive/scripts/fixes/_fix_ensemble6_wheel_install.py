"""Switch to openvino wheel install (internet=off, submit-ready).

Use cooolz/openvino-package dataset for offline install.
"""
import json
from pathlib import Path

# 1. Update NB cell 1 to install from wheel
P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "openvino" in src and "pip install" in src:
        new_src = '''# ============================================================
# Cell 1: Setup — install openvino from wheel (internet=off submit-ready)
# ============================================================
# ★ openvino wheel from cooolz/openvino-package Dataset (cp312、internet=off OK)
import os, sys
OV_WHEEL_DIR = '/kaggle/input/datasets/cooolz/openvino-package'
if not os.path.exists(OV_WHEEL_DIR):
    OV_WHEEL_DIR = '/kaggle/input/openvino-package'   # fallback
assert os.path.exists(OV_WHEEL_DIR), f'openvino wheel dataset not attached: {OV_WHEEL_DIR}'
import glob
ov_whl = glob.glob(f'{OV_WHEEL_DIR}/openvino-*.whl')
tel_whl = glob.glob(f'{OV_WHEEL_DIR}/openvino_telemetry-*.whl')
assert ov_whl and tel_whl, f'missing wheels in {OV_WHEEL_DIR}'
print(f'Installing from wheels: {[w.split("/")[-1] for w in ov_whl + tel_whl]}')

# --no-deps to avoid numpy upgrade (which breaks scipy)
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                       *ov_whl, *tel_whl])
print('openvino wheel installed')

import time, json, gc, glob as glob_mod
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
from scipy.ndimage import convolve1d
import soundfile as sf
import librosa
import openvino as ov
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cpu')
torch.set_num_threads(4)
print(f'torch={torch.__version__}, timm={timm.__version__}, ov={ov.__version__}')
print(f'numpy={np.__version__}, device={device}, num_threads={torch.get_num_threads()}')

GLOBAL_START = time.time()
'''
        c["source"] = new_src.splitlines(keepends=True)
        print("OK Updated cell 1: wheel-based openvino install")
        break

# Also fix glob reference (we used glob_mod above, but rest of NB uses glob)
# Check if any later cell uses `glob.glob`
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "glob.glob" in src and "import glob" not in src:
        # Add import glob at top of cell
        if not src.startswith("import"):
            new_src = "import glob\n" + src
            c["source"] = new_src.splitlines(keepends=True)

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# 2. Update push script: enable_internet=False + add cooolz/openvino-package to dataset_sources
push_path = Path("experiment/ensemble6/notebook/_push_nb_infer_ensemble6.py")
push_src = push_path.read_text(encoding="utf-8")
push_src = push_src.replace(
    '"enable_internet": True,    # ★ speed test 用 (pip install openvino)',
    '"enable_internet": False,   # ★ wheel install で submit-ready'
)
# Add cooolz/openvino-package to dataset_sources
old_ds_block = '''        "dataset_sources": [
            f"{USER}/birdclef2026-exp015-weights",'''
new_ds_block = '''        "dataset_sources": [
            "cooolz/openvino-package",   # ★ openvino wheel for offline install
            f"{USER}/birdclef2026-exp015-weights",'''
if old_ds_block in push_src:
    push_src = push_src.replace(old_ds_block, new_ds_block)
push_path.write_text(push_src, encoding="utf-8")
print("OK Updated push script: internet=False + openvino-package attached")

# Syntax check
import ast
nb = json.loads(P.read_text(encoding="utf-8"))
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
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")

# Save memory note
memory_note = '''---
name: reference-openvino-wheel-kaggle
description: Kaggle 公開 OpenVINO wheel dataset `cooolz/openvino-package` を offline install で使う方法 (cp312、submit-ready)
metadata:
  type: reference
---

# OpenVINO offline install (Kaggle submission)

## 公開 wheel Dataset (verified 2026-05-27)

**`cooolz/openvino-package`** (53MB)
- `openvino-2026.0.0-20965-cp312-cp312-manylinux_2_28_x86_64.whl` (53 MB)
- `openvino_telemetry-2025.2.0-py3-none-any.whl` (25 KB)
- Last updated: 2026-03-23

## Install code (internet=off compatible)

```python
import sys, subprocess, glob, os

OV_WHEEL_DIR = '/kaggle/input/datasets/cooolz/openvino-package'
ov_whl = glob.glob(f'{OV_WHEEL_DIR}/openvino-*.whl')
tel_whl = glob.glob(f'{OV_WHEEL_DIR}/openvino_telemetry-*.whl')

# --no-deps to avoid numpy upgrade (breaks scipy)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       '--no-index', '--no-deps',
                       *ov_whl, *tel_whl])
import openvino as ov  # works
```

## Push 設定 (submit-ready)

```python
meta = {
    "enable_internet": False,    # submission OK
    "dataset_sources": [
        "cooolz/openvino-package",   # ★ openvino wheels
        # 他の dataset...
    ],
}
```

## 注意点
- `--no-deps` 必須 (numpy 2.x へのアップグレードで scipy._no_nep50_warning bug 回避)
- Python cp312 のみ (Kaggle current env)
- onnxruntime ではない (別 dataset `romantamrazov/onnxruntime-1-24-4` 等)
'''
mem_path = Path("C:/Users/maeke/.claude/projects/C--Users-maeke-work-kaggle-birdclef-2026/memory/reference_openvino_wheel_kaggle.md")
mem_path.write_text(memory_note, encoding="utf-8")
print(f"\nOK Saved memory: {mem_path}")
