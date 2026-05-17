"""Generate exp010 AVES wav2vec2 ONNX export NB.

Exports wav2vec2-base (or earthspecies/aves-base-bio if available) to ONNX
for use in blend NB. Saves to /kaggle/working/wav2vec2_onnx/ for Dataset upload.

Runs on Kaggle CPU or T4 GPU, ~5-10 min.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr",
    "# AVES wav2vec2 ONNX export\n"
    "\n"
    "Export wav2vec2-base to ONNX for fast CPU inference in blend NB.\n"
    "Output: `/kaggle/working/wav2vec2_onnx/` (~50MB), upload as Dataset.\n"
    "\n"
    "Inference speed: torch CPU の 2-3x。"))

cells.append(code_cell("install",
    "import subprocess, sys, time\n"
    "START = time.time()\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                       'transformers', 'optimum[onnxruntime]'])\n"
    'print(f"Install done in {time.time()-START:.0f}s")'))

cells.append(code_cell("export", r"""import os, time
from pathlib import Path
import numpy as np

os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# Try AVES first, fall back to wav2vec2-base
fallbacks = ["earthspecies/aves-base-bio", "facebook/wav2vec2-base"]
chosen = None
for name in fallbacks:
    try:
        from transformers import AutoFeatureExtractor
        AutoFeatureExtractor.from_pretrained(name)
        chosen = name
        print(f"Will export: {name}")
        break
    except Exception as e:
        print(f"  {name} unavailable: {type(e).__name__}")
assert chosen is not None

OUT_DIR = Path("/kaggle/working/wav2vec2_onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Export via optimum
t0 = time.time()
from optimum.onnxruntime import ORTModelForFeatureExtraction
ort_model = ORTModelForFeatureExtraction.from_pretrained(chosen, export=True)
ort_model.save_pretrained(str(OUT_DIR))
print(f"Export done in {time.time()-t0:.0f}s")

# Also save processor for offline use in blend NB
from transformers import AutoFeatureExtractor
fe = AutoFeatureExtractor.from_pretrained(chosen)
fe.save_pretrained(str(OUT_DIR))

# Save metadata
import json as _json
(OUT_DIR / "_aves_meta.json").write_text(_json.dumps({
    "model_name": chosen, "sample_rate": fe.sampling_rate
}, indent=2), encoding="utf-8")

print(f"\nFiles saved to {OUT_DIR}:")
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")"""))

cells.append(code_cell("verify", r"""# Smoke test: load via onnxruntime and run dummy input
import time, numpy as np
import onnxruntime as ort
from transformers import AutoFeatureExtractor

print(f"onnxruntime: {ort.__version__}")

ONNX_PATH = OUT_DIR / "model.onnx"
print(f"Loading: {ONNX_PATH} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(ONNX_PATH), sess_options=so,
                             providers=["CPUExecutionProvider"])

print(f"Providers: {sess.get_providers()}")
for inp in sess.get_inputs():
    print(f"  Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
for out in sess.get_outputs():
    print(f"  Output: {out.name}, shape={out.shape}, dtype={out.type}")

fe = AutoFeatureExtractor.from_pretrained(str(OUT_DIR))
SR = fe.sampling_rate

# Smoke test: 12 windows of 5sec @ 16k
batch = [np.random.randn(SR * 5).astype(np.float32) for _ in range(12)]
t0 = time.time()
inputs = fe(batch, sampling_rate=SR, return_tensors="np", padding=True)
print(f"input_values shape: {inputs['input_values'].shape}, dtype={inputs['input_values'].dtype}")

# Run
input_name = sess.get_inputs()[0].name
out = sess.run(None, {input_name: inputs["input_values"]})
print(f"Output 0 shape: {out[0].shape} ({time.time()-t0:.2f}s for 12 windows)")

# Time 4 runs of 12 windows for averaging
t0 = time.time()
for _ in range(4):
    sess.run(None, {input_name: inputs["input_values"]})
print(f"4 runs: {time.time()-t0:.2f}s ({(time.time()-t0)/4*1000:.0f}ms/forward of 12 windows)")"""))

cells.append(code_cell("summary", r"""# === Done. Upload OUT_DIR/ as Kaggle Dataset ===
import os
print("Output files for Dataset upload:")
total = 0
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        sz = p.stat().st_size
        total += sz
        print(f"  {p.name}: {sz/1e6:.2f} MB")
print(f"\nTotal: {total/1e6:.1f} MB")
print(f"\nNext step: download /kaggle/working/wav2vec2_onnx/ → upload as Kaggle Dataset")
print(f"Suggested Dataset slug: birdclef2026-wav2vec2-onnx")"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_aves_onnx_export.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
