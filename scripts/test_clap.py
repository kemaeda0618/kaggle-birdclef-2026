"""Smoke test for CLAP audio embedding extraction.

Run locally to verify the right API call before pushing to Kaggle.
"""
import numpy as np
import sys

print(f"Python: {sys.version}")
try:
    import torch
    import transformers
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
except ImportError as e:
    print(f"FAIL: {e}")
    sys.exit(1)

from transformers import ClapModel, ClapProcessor

CLAP_NAME = "laion/clap-htsat-unfused"
print(f"\nLoading {CLAP_NAME}...")
proc = ClapProcessor.from_pretrained(CLAP_NAME)
model = ClapModel.from_pretrained(CLAP_NAME).eval()
print(f"  proc.sampling_rate: {proc.feature_extractor.sampling_rate}")

# 5sec sample audio at 48kHz
SR = 48000
N = 3
batch = [np.random.randn(SR * 5).astype(np.float32) for _ in range(N)]

inputs = proc(audio=batch, sampling_rate=SR, return_tensors="pt", padding=True)
print(f"\ninputs keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n--- Method 1: model.get_audio_features(**inputs) ---")
try:
    out = model.get_audio_features(**inputs)
    print(f"  type: {type(out).__name__}")
    if torch.is_tensor(out):
        print(f"  tensor shape: {out.shape}")
    else:
        print(f"  attrs: {[a for a in dir(out) if not a.startswith('_')][:20]}")
        for attr in ("audio_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if torch.is_tensor(v):
                    print(f"    {attr}: {v.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- Method 2: model(**inputs).audio_embeds ---")
try:
    full_out = model(**inputs)
    print(f"  type: {type(full_out).__name__}")
    print(f"  attrs: {[a for a in dir(full_out) if not a.startswith('_')][:20]}")
    if hasattr(full_out, "audio_embeds"):
        ae = full_out.audio_embeds
        print(f"  audio_embeds: {ae.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- Method 3: model.audio_model(**audio_inputs).pooler_output ---")
try:
    audio_inputs = {k: v for k, v in inputs.items()
                    if k in ("input_features", "is_longer")}
    audio_out = model.audio_model(**audio_inputs)
    print(f"  type: {type(audio_out).__name__}")
    if hasattr(audio_out, "pooler_output"):
        po = audio_out.pooler_output
        print(f"  pooler_output: {po.shape}")
    if hasattr(audio_out, "last_hidden_state"):
        lhs = audio_out.last_hidden_state
        print(f"  last_hidden_state: {lhs.shape}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
