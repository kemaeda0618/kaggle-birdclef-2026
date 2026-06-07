"""exp032: Manually upload new R1 ckpt (val 0.9095) to Kaggle Dataset V2.

前提: Drive 上に新 ckpt が保存されてる:
  /content/drive/MyDrive/kaggle/birdclef2026/output/exp032/r1/ckpt_best_ns22.pth

Drive ファイルを local に copy してから dataset_create_version で V2 upload。
Colab で実行するスクリプト。
"""
import json, os, shutil
from pathlib import Path

# Colab 環境前提
DRIVE_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp032/r1")
LOCAL_DIR = Path("/content/exp032_upload")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Copy files to local
files_to_upload = ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth", "history.json"]
for fn in files_to_upload:
    src = DRIVE_DIR / fn
    if src.exists():
        shutil.copy(src, LOCAL_DIR / fn)
        print(f"  copied: {fn} ({(LOCAL_DIR / fn).stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  MISSING: {src}")

# Verify ckpt content
import torch
state = torch.load(str(LOCAL_DIR / "ckpt_best_ns22.pth"), map_location="cpu", weights_only=False)
print(f"\nNew ckpt: epoch={state.get('epoch')}, "
      f"best_ns22={state.get('best_ns22'):.4f}, "
      f"best_macro={state.get('best_macro'):.4f}")

# Setup metadata for dataset_create_version
metadata = {
    "id": "maekeso/birdclef2026-exp032-weights",
    "title": "birdclef2026 exp032 weights",
    "licenses": [{"name": "CC0-1.0"}],
}
with open(LOCAL_DIR / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Upload as new version
os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print(f"\nUploading V2 to maekeso/birdclef2026-exp032-weights...")
try:
    result = api.dataset_create_version(
        folder=str(LOCAL_DIR),
        version_notes=f"exp032 R1 mel-fixed (n_mels=224, hop=312) val_ns22={state['best_ns22']:.4f}",
        quiet=False,
        dir_mode="zip",
    )
    print(f"  result: {result}")
except Exception as e:
    print(f"  ERROR: {str(e)[:400]}")
