"""Make kaggle.json path more flexible:
1. Try multiple common locations
2. If not found, use files.upload() prompt
3. Save to Drive for next time
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_INSTALL = '''# ============================================================
# Cell 1: Install + Drive mount + Kaggle API setup
# ============================================================
!pip install -q timm kaggle soundfile librosa torchaudio tqdm

from google.colab import drive
drive.mount("/content/drive")

import os, json, shutil
from pathlib import Path

# Drive paths
DRIVE_ROOT = Path("/content/drive/MyDrive/birdclef2026/exp052")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = DRIVE_ROOT / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)

# Kaggle credentials - try multiple locations
KAGGLE_JSON_CANDIDATES = [
    Path("/content/drive/MyDrive/birdclef2026/kaggle.json"),   # 推奨
    Path("/content/drive/MyDrive/kaggle.json"),                 # Drive root
    Path("/content/drive/MyDrive/.kaggle/kaggle.json"),
    Path("/content/kaggle.json"),                               # Colab session
    Path.home() / ".kaggle" / "kaggle.json",                   # default
]

KAGGLE_JSON_PATH = next((p for p in KAGGLE_JSON_CANDIDATES if p.exists()), None)

if KAGGLE_JSON_PATH is None:
    print("kaggle.json not found in:")
    for p in KAGGLE_JSON_CANDIDATES:
        print(f"  - {p}")
    print("\\nUpload kaggle.json now:")
    from google.colab import files
    uploaded = files.upload()
    if "kaggle.json" in uploaded:
        # Save to Drive for next time
        target = Path("/content/drive/MyDrive/birdclef2026/kaggle.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded["kaggle.json"])
        KAGGLE_JSON_PATH = target
        print(f"OK Saved to {target}")
    else:
        raise FileNotFoundError("kaggle.json not uploaded")

print(f"Using kaggle.json from: {KAGGLE_JSON_PATH}")

# Copy to ~/.kaggle/
os.makedirs(Path.home() / ".kaggle", exist_ok=True)
shutil.copy(KAGGLE_JSON_PATH, Path.home() / ".kaggle" / "kaggle.json")
os.chmod(Path.home() / ".kaggle" / "kaggle.json", 0o600)

# Verify Kaggle API
!kaggle datasets list -m 1 2>&1 | head -5
print("OK Kaggle API ready")
'''

for c in nb["cells"]:
    if c.get("id") == "install":
        c["source"] = NEW_INSTALL.splitlines(keepends=True)
        print("[install] OK kaggle.json fallback added")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
