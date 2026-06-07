"""Fix Kaggle API auth in Stage 1 NB.

Issue: kaggle.json has KGAT_ Bearer token, CLI tries Basic auth -> 403
Fix: Set KAGGLE_API_TOKEN env var (CLAUDE.md memory `feedback_kaggle_push.md`)

Also: dl_data cell needs fail-fast on download error (currently prints "done" even on 403)
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# === Fix 1: Cell 1 (install) — use KAGGLE_API_TOKEN env var ===
NEW_INSTALL = '''# ============================================================
# Cell 1: Install + Drive mount + Kaggle API setup
# ============================================================
!pip install -q timm kaggle soundfile librosa torchaudio tqdm

from google.colab import drive
drive.mount("/content/drive")

import os, json, shutil
from pathlib import Path

# Drive paths
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/exp052")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = DRIVE_ROOT / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)

# Kaggle credentials - try multiple locations
KAGGLE_JSON_CANDIDATES = [
    Path("/content/drive/MyDrive/kaggle/birdclef2026/kaggle.json"),  # ★ 標準
    Path("/content/drive/MyDrive/kaggle.json"),
    Path("/content/drive/MyDrive/.kaggle/kaggle.json"),
    Path("/content/kaggle.json"),
    Path.home() / ".kaggle" / "kaggle.json",
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
        target = Path("/content/drive/MyDrive/kaggle/birdclef2026/kaggle.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded["kaggle.json"])
        KAGGLE_JSON_PATH = target
        print(f"OK Saved to {target}")
    else:
        raise FileNotFoundError("kaggle.json not uploaded")

print(f"Using kaggle.json from: {KAGGLE_JSON_PATH}")

# ★ KGAT_ Bearer token を KAGGLE_API_TOKEN env var に設定 (CLAUDE.md memory)
# Basic auth (kaggle.json 直接) だと 401/403 になる
creds = json.loads(KAGGLE_JSON_PATH.read_text())
if "key" in creds:
    token = creds["key"]
    os.environ["KAGGLE_API_TOKEN"] = token
    print(f"OK KAGGLE_API_TOKEN env var set (token starts: {token[:10]}...)")
else:
    raise ValueError(f"kaggle.json missing 'key' field: {creds.keys()}")

# Also copy to ~/.kaggle/ for backup (CLI fall back behavior)
os.makedirs(Path.home() / ".kaggle", exist_ok=True)
shutil.copy(KAGGLE_JSON_PATH, Path.home() / ".kaggle" / "kaggle.json")
os.chmod(Path.home() / ".kaggle" / "kaggle.json", 0o600)

# Verify Kaggle API (should NOT 403)
print("\\nTesting Kaggle API...")
import subprocess
r = subprocess.run("kaggle datasets list -m 1", shell=True, capture_output=True, text=True,
                   env={**os.environ})
print(r.stdout[:500])
if "403" in r.stderr or "401" in r.stderr or "Forbidden" in r.stderr:
    print(f"⚠ Auth error: {r.stderr[:500]}")
    raise RuntimeError("Kaggle API auth failed - check KGAT token in kaggle.json")
else:
    print("OK Kaggle API ready")
'''

# === Fix 2: Cell 2 (dl_data) — fail-fast on download error ===
NEW_DL_DATA = '''# ============================================================
# Cell 2: Download XC Datasets (Part 1 + Part 2) — fail-fast on error
# ============================================================
import subprocess

DATA_ROOT = Path("/content/xc_data")
DATA_ROOT.mkdir(exist_ok=True)

part1_dir = DATA_ROOT / "part1"
part2_dir = DATA_ROOT / "part2"

def kaggle_dl(slug, target_dir):
    target_dir.mkdir(exist_ok=True, parents=True)
    print(f"\\nDownloading {slug} -> {target_dir}")
    cmd = f"kaggle datasets download -d {slug} -p {target_dir} --unzip"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       env={**os.environ})
    print(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0 or "403" in r.stderr or "401" in r.stderr or "Forbidden" in r.stderr:
        print(f"FAIL: {r.stderr[:500]}")
        raise RuntimeError(f"Kaggle DL failed for {slug}")
    print(f"OK {slug}")

# Part 1
p1_meta = part1_dir / "xc_api" / "_metadata.csv"
if not p1_meta.exists():
    kaggle_dl("maekeso/birdclef2026-xc-api-dl-part1", part1_dir)
else:
    print(f"Part 1 already exists: {p1_meta}")

# Part 2
p2_meta = part2_dir / "xc_api_part2" / "_metadata.csv"
if not p2_meta.exists():
    kaggle_dl("maekeso/birdclef2026-xc-api-dl-part2", part2_dir)
else:
    print(f"Part 2 already exists: {p2_meta}")

# Verify
import pandas as pd
assert p1_meta.exists(), f"Part 1 metadata not found: {p1_meta}"
assert p2_meta.exists(), f"Part 2 metadata not found: {p2_meta}"
p1_df = pd.read_csv(p1_meta)
p2_df = pd.read_csv(p2_meta)
print(f"\\nPart 1: {len(p1_df)} files, {p1_df['scientific_name'].nunique()} species")
print(f"Part 2: {len(p2_df)} files, {p2_df['scientific_name'].nunique()} species")
print(f"Total: {len(p1_df) + len(p2_df)} files, {p1_df['scientific_name'].nunique() + p2_df['scientific_name'].nunique()} species")
'''

for c in nb["cells"]:
    if c.get("id") == "install":
        c["source"] = NEW_INSTALL.splitlines(keepends=True)
        print("[install] OK KAGGLE_API_TOKEN env var fix applied")
    if c.get("id") == "dl_data":
        c["source"] = NEW_DL_DATA.splitlines(keepends=True)
        print("[dl_data] OK fail-fast error check applied")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
