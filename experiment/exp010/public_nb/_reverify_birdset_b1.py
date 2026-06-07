"""Re-verify dacquaviva/birdset-effnet-b1-xcl dataset metadata + content."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "dacquaviva/birdset-effnet-b1-xcl"

# Files
print("=== Dataset files ===")
files = api.dataset_list_files(slug).files
for f in files:
    size = getattr(f, "totalBytes", None) or getattr(f, "total_bytes", None) or 0
    print(f"  {f.name}  {size/1e6:.2f} MB")
total = sum(getattr(f, "totalBytes", 0) or getattr(f, "total_bytes", 0) or 0 for f in files)
print(f"Total: {total/1e6:.2f} MB")

# Already downloaded files
print("\n=== Local content ===")
local_dir = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\birdset_effnet_b1_xcl")
for f in sorted(local_dir.glob("*")):
    print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB")

# Inspect config.json key sections
print("\n=== config.json key fields ===")
cfg_path = local_dir / "config.json"
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    keys_of_interest = [
        "_name_or_path", "architectures", "model_type", "num_labels", "num_channels",
        "image_size", "depth_coefficient", "width_coefficient",
        "hidden_dim", "expand_ratios", "depthwise_padding",
        "hidden_act", "dropout_rate", "drop_connect_rate", "batch_norm_eps",
    ]
    for k in keys_of_interest:
        if k in cfg:
            val = cfg[k]
            if isinstance(val, (list, dict)):
                val = str(val)[:200]
            print(f"  {k}: {val}")
    # id2label size
    if "id2label" in cfg:
        n = len(cfg["id2label"])
        print(f"  id2label: {n} entries (Xeno-Canto bird species count)")
        first_3 = list(cfg["id2label"].items())[:3]
        print(f"    first 3: {first_3}")

# README key sections
print("\n=== README.md key info ===")
readme = local_dir / "README.md"
if readme.exists():
    txt = readme.read_text(encoding="utf-8")
    print(txt[:1500])
