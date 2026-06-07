"""Generate diagnostic NB to find where it hangs/fails."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_diag.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}

nb = {
    "cells": [
        code_cell("c1", """import sys, time
t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] start", flush=True)

import torch
print(f"[{time.time()-t0:5.1f}s] torch={torch.__version__} cuda={torch.cuda.is_available()} devs={torch.cuda.device_count()}", flush=True)

import torch.nn as nn
print(f"[{time.time()-t0:5.1f}s] torch.nn OK", flush=True)

import timm
print(f"[{time.time()-t0:5.1f}s] timm={timm.__version__}", flush=True)

# Try simpler backbone first
print(f"[{time.time()-t0:5.1f}s] creating efficientnetv2_s...", flush=True)
model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0, global_pool="")
print(f"[{time.time()-t0:5.1f}s] model created, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
"""),

        code_cell("c2", """import time
t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] move model to cuda...", flush=True)
model = model.cuda()
print(f"[{time.time()-t0:5.1f}s] OK", flush=True)

print(f"[{time.time()-t0:5.1f}s] DataParallel?", flush=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"[{time.time()-t0:5.1f}s] DataParallel applied", flush=True)
else:
    print(f"[{time.time()-t0:5.1f}s] single GPU", flush=True)
"""),

        code_cell("c3", """import time
t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] torchaudio mel...", flush=True)
import torchaudio
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128,
).cuda()
db = torchaudio.transforms.AmplitudeToDB(top_db=80).cuda()
print(f"[{time.time()-t0:5.1f}s] mel created", flush=True)

# Dummy waveform
x = torch.randn(8, 32000*5, device="cuda")
print(f"[{time.time()-t0:5.1f}s] x: {x.shape}", flush=True)

# Forward mel
m = mel(x)
m = db(m)
print(f"[{time.time()-t0:5.1f}s] mel output: {m.shape}", flush=True)

# Forward model
x3 = m.unsqueeze(1).repeat(1, 3, 1, 1)
print(f"[{time.time()-t0:5.1f}s] x3: {x3.shape}", flush=True)
out = model(x3)
print(f"[{time.time()-t0:5.1f}s] model output: {out.shape}", flush=True)
"""),

        code_cell("c4", """import time
t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] backward test...", flush=True)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

with autocast():
    out = model(x3)
    loss = out.mean()
print(f"[{time.time()-t0:5.1f}s] forward in autocast OK, loss={loss.item():.4f}", flush=True)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
print(f"[{time.time()-t0:5.1f}s] backward + step OK", flush=True)
"""),

        code_cell("c5", """import time
t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] OGG read test...", flush=True)
import soundfile as sf
import pandas as pd
from pathlib import Path

BC_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
train_df = pd.read_csv(BC_ROOT / "train.csv")
print(f"[{time.time()-t0:5.1f}s] train.csv: {len(train_df)} rows", flush=True)

# Try first 5 OGG reads
for i in range(5):
    fname = train_df.iloc[i]["filename"]
    path = BC_ROOT / "train_audio" / fname
    wav, sr = sf.read(str(path), dtype="float32")
    print(f"[{time.time()-t0:5.1f}s] {fname}: {wav.shape} sr={sr}", flush=True)
"""),

        code_cell("c6", """import time
t0 = time.time()
print(f"[{time.time()-t0:5.1f}s] 100 OGG read benchmark...", flush=True)
read_times = []
for i in range(100):
    s = time.time()
    fname = train_df.iloc[i]["filename"]
    path = BC_ROOT / "train_audio" / fname
    wav, sr = sf.read(str(path), dtype="float32")
    read_times.append(time.time() - s)
import numpy as np
print(f"[{time.time()-t0:5.1f}s] 100 reads done", flush=True)
print(f"  avg: {np.mean(read_times)*1000:.1f}ms, max: {max(read_times)*1000:.1f}ms, min: {min(read_times)*1000:.1f}ms", flush=True)
print(f"  for batch 64 = {np.mean(read_times)*64:.1f}s per batch", flush=True)
print(f"  for 351 batches = {np.mean(read_times)*64*351/60:.1f}min per epoch", flush=True)
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
