"""Extract just the InferenceConfig / model_groups portion of cell 9."""
import json, sys, io, os, re
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = r"experiment/exp010/notebook/_reference_babych/birdclef2025-1st-place-inference/birdclef2025-1st-place-inference.ipynb"
nb = json.load(open(p, encoding="utf-8"))
src = "".join(nb['cells'][9].get('source', []))
# Extract sections around key markers
markers = ["class InferenceConfig", "preds_weights", "model_groups", "model_configs", "tta_delta", "duration", "infer_duration", "n_mels", "hop_length", "n_fft", "preds_power", "smoothing"]
lines = src.split("\n")
# Find indexes of class definitions
print(f"=== Cell 9 ({len(src)} chars, {len(lines)} lines) ===\n")
# Find class InferenceConfig
for i, ln in enumerate(lines):
    if "class InferenceConfig" in ln or "class Config" in ln or "class TrainConfig" in ln or "_config = " in ln or "preds_weights" in ln or "preds_power" in ln:
        # print 25 lines from here
        end = min(i + 30, len(lines))
        print(f"--- line {i} ---")
        for j in range(i, end):
            print(f"  {lines[j]}")
        print()
