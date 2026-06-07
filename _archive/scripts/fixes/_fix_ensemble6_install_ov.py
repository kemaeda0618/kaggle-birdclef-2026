"""Add openvino install to cell 1 + change push to enable_internet=True (speed test).

For actual submission, need:
- Public openvino wheel dataset (memory: TBD find/create)
- internet=False
"""
import json
from pathlib import Path

P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# Update cell 1: add pip install + handle no-openvino fallback
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "import openvino as ov" in src:
        # Replace cell with pip install at top
        new_src = '''# ============================================================
# Cell 1: Setup — install openvino + imports
# ============================================================
# ★ Speed measurement mode: internet=True で pip install
# 本番 submission では internet=False + openvino wheel dataset 必要
!pip install -q openvino==2024.4.0

import os, sys, time, json, gc, glob
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

device = torch.device('cpu')   # Kaggle CPU sub
torch.set_num_threads(4)
print(f'torch={torch.__version__}, timm={timm.__version__}, ov={ov.__version__}')
print(f'Device: {device}, num_threads={torch.get_num_threads()}')

GLOBAL_START = time.time()
'''
        c["source"] = new_src.splitlines(keepends=True)
        print("OK Updated cell 1 with openvino install")
        break

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Update push script: enable_internet=True
push_path = Path("experiment/ensemble6/notebook/_push_nb_infer_ensemble6.py")
push_src = push_path.read_text(encoding="utf-8")
push_src = push_src.replace('"enable_internet": False,',
                             '"enable_internet": True,    # ★ speed test 用 (pip install openvino)')
push_path.write_text(push_src, encoding="utf-8")
print("OK Updated push script: internet=True")

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
