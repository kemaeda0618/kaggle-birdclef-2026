"""Push exp058 NB1 (pseudo gen) to Colab (manual upload) or for reference.

Note: This NB is intended for Colab Blackwell execution.
Colab does not support automatic push via Kaggle API.
Manual workflow: upload nb_pseudo.ipynb to Colab Drive folder, open and run.

Drive path: /content/drive/MyDrive/kaggle/birdclef2026/notebooks/exp058/nb_pseudo.ipynb (recommended)
"""
import json, os, sys, io
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_pseudo.ipynb")
print(f"NB local: {NB}")
print(f"  size: {NB.stat().st_size / 1024:.1f} KB")

print("\n=== Manual upload to Colab ===")
print("1. Open Google Drive: kaggle/birdclef2026/notebooks/exp058/")
print(f"2. Upload {NB.name}")
print("3. Open in Colab and run cells sequentially")
print("\n=== Cell flow ===")
print("  Cell 1: install + drive mount")
print("  Cell 2: DL XC Part 1/2/3 + exp020 R2 ckpts (~5-10 min, ~30 GB)")
print("  Cell 3: imports + GPU info")
print("  Cell 4: BC2026 species/taxonomy")
print("  Cell 5-6: load existing XC metadata + scan audio files")
print("  Cell 7: XC API metadata refresh (~10 min, adds recordist)")
print("  Cell 8: filter quality A/B + per-recordist cap N=10")
print("  Cell 9-10: model + load 5 fold ckpts")
print("  Cell 11-12: pseudo gen (~30 min - 2h depending on filtered count)")
print("  Cell 13: save pseudo .npz + index.csv")
print("  Cell 14: Kaggle Dataset upload")
print("  Cell 15-16: summary + auto-disconnect")
