"""Push exp081 R2 NB (Colab Pro G4)."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Note: This NB is intended to run on Colab (Drive mount, kaggle API).
# This push script is provided in case user wants to also push to Kaggle for reference.
# For actual training, user runs on Colab manually.

print("Note: exp081 NB is Colab-targeted. To run:")
print("  1. Upload nb_train_r2.ipynb to Colab")
print("  2. Place kaggle.json at /content/drive/MyDrive/kaggle/birdclef2026/")
print("  3. Ensure exp017 R1 pseudo exists at:")
print("     /content/drive/MyDrive/kaggle/birdclef2026/output/exp017/r1-pseudo/pseudo_labels.csv")
print("  4. Run all cells (~5h on G4 96GB)")
print()
print(f"NB file: experiment/exp081/notebook/nb_train_r2.ipynb")
