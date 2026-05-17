"""CLAP smoke test NB — figure out the working API call pattern.

Lightweight: just loads CLAP and tries 3 API patterns to extract audio embedding
from a 5-sec dummy waveform. No real data, no submission. ~2 min.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

cells = []

cells.append({
    "cell_type": "markdown", "id": "hdr", "metadata": {}, "source": [
        "# CLAP API smoke test\n",
        "Try multiple API patterns to find the working one for transformers 새버전.\n"
    ],
})

CODE = r"""import sys, traceback
import numpy as np
import torch
import transformers

print(f"transformers: {transformers.__version__}")
print(f"torch:        {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import ClapModel, ClapProcessor
CLAP_NAME = "laion/clap-htsat-unfused"
print(f"\nLoading {CLAP_NAME}...")
proc = ClapProcessor.from_pretrained(CLAP_NAME)
model = ClapModel.from_pretrained(CLAP_NAME).to(DEVICE).eval()
SR = proc.feature_extractor.sampling_rate
print(f"SR = {SR}")

# Dummy audio batch
batch = [np.random.randn(SR * 5).astype(np.float32) for _ in range(3)]
inputs = proc(audio=batch, sampling_rate=SR, return_tensors="pt", padding=True)
inputs = {k: v.to(DEVICE) for k, v in inputs.items() if hasattr(v, 'to')}
print(f"\ninputs keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

# Method 1: get_audio_features
print("\n--- Method 1: model.get_audio_features(**inputs) ---")
with torch.no_grad():
    try:
        out = model.get_audio_features(**inputs)
        t = type(out).__name__
        print(f"  type: {t}")
        if torch.is_tensor(out):
            print(f"  RESULT: tensor shape={tuple(out.shape)}  ✅")
        else:
            attrs = [a for a in dir(out) if not a.startswith("_")][:25]
            print(f"  attrs: {attrs}")
            for attr in ("audio_embeds", "pooler_output", "last_hidden_state"):
                if hasattr(out, attr):
                    v = getattr(out, attr)
                    if torch.is_tensor(v):
                        print(f"  attr {attr}: {tuple(v.shape)}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

# Method 2: full forward with audio_embeds
print("\n--- Method 2: model(**inputs).audio_embeds ---")
with torch.no_grad():
    try:
        full_out = model(**inputs)
        print(f"  type: {type(full_out).__name__}")
        attrs = [a for a in dir(full_out) if not a.startswith("_")][:25]
        print(f"  attrs: {attrs}")
        if hasattr(full_out, "audio_embeds"):
            v = full_out.audio_embeds
            print(f"  audio_embeds: {tuple(v.shape)} ✅")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

# Method 3: model.audio_model(...) directly
print("\n--- Method 3: model.audio_model(**audio_inputs) ---")
with torch.no_grad():
    try:
        audio_inputs = {k: v for k, v in inputs.items()
                        if k in ("input_features", "is_longer")}
        audio_out = model.audio_model(**audio_inputs)
        print(f"  type: {type(audio_out).__name__}")
        for attr in ("pooler_output", "last_hidden_state"):
            if hasattr(audio_out, attr):
                v = getattr(audio_out, attr)
                if torch.is_tensor(v):
                    print(f"  {attr}: {tuple(v.shape)} ✅")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
"""

cells.append({
    "cell_type": "code", "id": "main", "metadata": {},
    "outputs": [], "execution_count": None,
    "source": CODE.split("\n"),
})
# fix newlines for nbformat
src_lines = CODE.split("\n")
cells[-1]["source"] = [s + "\n" for s in src_lines[:-1]] + ([src_lines[-1]] if src_lines[-1] else [])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = HERE / "nb_clap_smoke.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out}")
