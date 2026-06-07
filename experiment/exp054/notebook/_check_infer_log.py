"""Check exp054 inference NB log + output."""
import json, os, re
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp054-stage2-infer"
print(f"=== {slug} status ===")
print(api.kernels_status(slug))

out = Path("/tmp/exp054_infer_log")
out.mkdir(parents=True, exist_ok=True)
try:
    api.kernels_output(slug, str(out))
except Exception as e:
    print(f"DL err: {str(e)[:200]}")

print("\n=== Files ===")
for f in out.rglob("*"):
    if f.is_file():
        print(f"  {f.relative_to(out)}  ({f.stat().st_size/1024:.1f} KB)")

# Read log
for f in out.rglob("*.log"):
    print(f"\n=== {f.name} ===")
    t = f.read_text(encoding="utf-8", errors="ignore")
    for line in t.split("\n"):
        if any(k in line for k in ["ckpt loaded", "val_ns22", "val_macro", "ep:", "Stage 1", "Test files", "PP applied", "ETA", "Inference done", "shape", "submission", "preds_array", "pred mean"]):
            m = re.search(r'"data":"([^"]+)"', line)
            if m: line = m.group(1)
            line = line.strip().replace("\\n", "")
            if line and len(line) < 250:
                print(f"  {line}")

# Check submission.csv
import pandas as pd
sub_path = out / "submission.csv"
if sub_path.exists():
    sub = pd.read_csv(sub_path)
    print(f"\n=== submission.csv ===")
    print(f"  shape: {sub.shape}")
    print(f"  cols: {sub.columns.tolist()[:5]} ... {sub.columns.tolist()[-3:]}")
    print(f"  head:")
    print(sub.head(3))
    pred_cols = [c for c in sub.columns if c != "row_id"]
    pred_array = sub[pred_cols].values
    print(f"\n  Prediction stats:")
    print(f"    shape: {pred_array.shape}")
    print(f"    mean: {pred_array.mean():.4f}")
    print(f"    median: {sub[pred_cols].median().median():.4f}")
    print(f"    max: {pred_array.max():.4f}")
    print(f"    min: {pred_array.min():.4f}")
    # Check if all zeros or all same
    n_unique = len(set(pred_array.flatten()[:1000]))
    print(f"    unique vals (first 1000): {n_unique}")
