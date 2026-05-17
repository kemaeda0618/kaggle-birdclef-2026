"""Generate exp012 nb_train_colab.ipynb for Google Colab Pro (A100/V100).

Strategy:
  - Same Tucker pipeline as Kaggle nb_train.ipynb
  - All 5 folds in one notebook (Colab Pro 24h session, A100 ~5-15min/epoch)
  - Drive-mounted checkpoint for persistence across disconnects
  - kaggle CLI for data download (cache to Drive to avoid re-DL)
  - At end: zip ONNX outputs and upload as Kaggle dataset for inference NB

Run:
  python _gen_nb_train_colab.py  -> writes nb_train_colab.ipynb
Then: open nb_train_colab.ipynb in Colab Pro, Runtime -> Change runtime type -> A100
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# Reuse Tucker reference for cell content
ref_nb = json.loads((HERE / "_reference_tucker.ipynb").read_text(encoding="utf-8"))


def code_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}


def cell_src(c):
    s = c["source"]
    return s if isinstance(s, str) else "".join(s)


cells = []

# ============================================================
# 0. Header
# ============================================================
cells.append(md_cell(r"""# exp012 — Distilled SED Training (Colab Pro version)

Tucker SED pipeline 5-fold training on Colab Pro (A100/V100). All folds in one
notebook. Outputs: 5 ONNX files uploaded to Kaggle Dataset for inference NB.

## Prerequisites
1. Colab Pro subscription (A100 or V100 recommended)
2. `kaggle.json` in Google Drive at `/MyDrive/kaggle.json` (with KGAT_ token in `key` field)
3. Drive mounted (cell 1 will prompt)

## Runtime: select GPU
Runtime -> Change runtime type -> Hardware accelerator: **A100** (or V100)

## Outputs (in Drive at `/MyDrive/birdclef-2026/exp012/output/`)
- `fold{0..4}_best_macro.pt`
- `fold{0..4}_history.json`
- `sed_distill_fold{0..4}.onnx`
- `exp012_onnx_bundle.zip` for Kaggle upload

## Time estimate (A100)
~10 min/epoch × 25 epochs × 5 folds = ~21 hours total. Fits in one Colab Pro
24h session. With V100, expect ~30-50% slower.
""", "hdr"))

# ============================================================
# 1. Setup: install + drive + kaggle config
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Setup: install + Drive + Kaggle credentials
# ============================================================
!pip install -q timm onnx onnxruntime-gpu librosa kaggle scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, shutil, json
from pathlib import Path

KAGGLE_JSON_DRIVE = Path("/content/drive/MyDrive/kaggle.json")
assert KAGGLE_JSON_DRIVE.exists(), \
    "Place kaggle.json (with KGAT_ token in 'key') at /MyDrive/kaggle.json"

# Setup ~/.kaggle for kaggle CLI
KAGGLE_CFG_DIR = Path.home() / ".kaggle"
KAGGLE_CFG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(str(KAGGLE_JSON_DRIVE), str(KAGGLE_CFG_DIR / "kaggle.json"))
os.chmod(str(KAGGLE_CFG_DIR / "kaggle.json"), 0o600)

# Also export for SDK that uses KAGGLE_API_TOKEN
key = json.loads(KAGGLE_JSON_DRIVE.read_text())["key"]
if key.startswith("KGAT_"):
    os.environ["KAGGLE_API_TOKEN"] = key

# Working dirs
WORK_ROOT     = Path("/content/drive/MyDrive/birdclef-2026")
DATA_DIR      = WORK_ROOT / "data"            # cached downloads (skip re-DL)
EXP_DIR       = WORK_ROOT / "exp012"
OUT_DIR       = EXP_DIR / "output"
LOCAL_DATA    = Path("/content/data")          # local copy for fast I/O
LOCAL_OUT     = Path("/content/output")        # local checkpoints (mirror to Drive)
for p in [WORK_ROOT, DATA_DIR, EXP_DIR, OUT_DIR, LOCAL_DATA, LOCAL_OUT]:
    p.mkdir(parents=True, exist_ok=True)

print(f"Drive Out: {OUT_DIR}")
print(f"Local Out: {LOCAL_OUT}")
""", "setup"))

# ============================================================
# 2. Data download (cached, skip if already on Drive)
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Download data once, cache to Drive, copy to local for fast I/O
# ============================================================
import subprocess, time

def maybe_download(target_dir_drive, target_dir_local, dl_cmd, marker_file):
    # If marker file doesn't exist on Drive, download. Then sync to local.
    drive_marker = target_dir_drive / marker_file
    if not drive_marker.exists():
        print(f"Downloading -> {target_dir_drive}")
        target_dir_drive.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        subprocess.run(dl_cmd, cwd=str(target_dir_drive), shell=True, check=True)
        print(f"  done in {time.time()-t0:.0f}s")
    else:
        print(f"Cached on Drive: {target_dir_drive}")
    # Sync to local fast disk
    target_dir_local.mkdir(parents=True, exist_ok=True)
    if not (target_dir_local / marker_file).exists():
        print(f"  copying to {target_dir_local} ...")
        t0 = time.time()
        # rsync if available, else cp
        rc = subprocess.run(f"rsync -a --info=progress2 {target_dir_drive}/ {target_dir_local}/",
                            shell=True).returncode
        if rc != 0:
            subprocess.run(f"cp -r {target_dir_drive}/* {target_dir_local}/", shell=True, check=True)
        print(f"  copied in {time.time()-t0:.0f}s")

# 1) Birdclef-2026 competition data
maybe_download(
    DATA_DIR / "competition",
    LOCAL_DATA / "competition",
    "kaggle competitions download -c birdclef-2026 -q && unzip -q -o birdclef-2026.zip && rm birdclef-2026.zip",
    "train.csv",
)

# 2) Tucker waveform cache
maybe_download(
    DATA_DIR / "waveform-cache",
    LOCAL_DATA / "waveform-cache",
    "kaggle datasets download -d tuckerarrants/birdclef-2026-waveform-cache -q && "
    "unzip -q -o birdclef-2026-waveform-cache.zip && rm birdclef-2026-waveform-cache.zip",
    "waveform_cache/audio_cache_meta.csv",
)

# 3) Perch v2 ONNX
maybe_download(
    DATA_DIR / "perch-onnx",
    LOCAL_DATA / "perch-onnx",
    "kaggle datasets download -d tuckerarrants/perch-v2-no-dft-onnx -q && "
    "unzip -q -o perch-v2-no-dft-onnx.zip && rm perch-v2-no-dft-onnx.zip",
    "perch_v2_no_dft.onnx",
)

print("\nData ready:")
for p in sorted(LOCAL_DATA.rglob("*"))[:30]:
    if p.is_file():
        print(f"  {p.relative_to(LOCAL_DATA)} ({p.stat().st_size/1e6:.1f}MB)")
""", "data_dl"))

# ============================================================
# 3. Imports + config (modified for Colab paths)
# ============================================================
c4_src = cell_src(ref_nb["cells"][4])
import re
c4_src = c4_src.replace('MODE = "infer"', 'MODE = "train"')
c4_src = re.sub(
    r"if\s+MODE\s*==\s*['\"]train['\"][^\n]*\n\s+DEBUG\s*=\s*True[^\n]*\n\s*else\s*:\s*\n\s+DEBUG\s*=\s*False",
    "DEBUG = False  # Colab Pro: full training",
    c4_src, count=1)
c4_src = c4_src.replace("FOLDS  = [0, 1, 2, 3, 4]", "FOLDS  = [0, 1, 2, 3, 4]  # all 5 folds in one Colab session")
c4_src = c4_src.replace("BATCH  = 16 if DEBUG else 64", "BATCH  = 64  # A100 has 40GB, V100 has 16GB (drop to 32 if V100 OOM)")
# Replace Kaggle-specific paths with Colab/local paths
c4_src = c4_src.replace(
    'COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")',
    'COMP_DIR = Path("/content/data/competition")')
c4_src = c4_src.replace(
    'WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")',
    'WAVEFORM_CACHE_DIR = Path("/content/data/waveform-cache/waveform_cache")')
c4_src = c4_src.replace(
    'PERCH_ONNX_PATH = Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx")',
    'PERCH_ONNX_PATH = Path("/content/data/perch-onnx/perch_v2_no_dft.onnx")')
c4_src = c4_src.replace(
    'OUT_DIR = Path("/kaggle/working")',
    'OUT_DIR = Path("/content/output")  # local fast disk; we mirror to Drive after each fold')

cells.append(code_cell(c4_src, "config"))

# ============================================================
# 4. Load data
# ============================================================
cells.append(code_cell(cell_src(ref_nb["cells"][7]), "load_data"))

# (skip cell 8 DEBUG override)

# ============================================================
# 5. Model definitions
# ============================================================
cells.append(code_cell(cell_src(ref_nb["cells"][10]), "model_defs"))

# ============================================================
# 6. Data pipeline
# ============================================================
cells.append(code_cell(cell_src(ref_nb["cells"][12]), "data_pipeline"))

# ============================================================
# 7. Training function (Tucker original — Colab has 24h, no need for resume cell)
# ============================================================
cells.append(code_cell(cell_src(ref_nb["cells"][14]), "train_fn"))

# ============================================================
# 8. Patch train_fold to mirror checkpoints to Drive after each epoch
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Patch: mirror best checkpoints to Drive after each epoch
# (so a Colab disconnect doesn't lose progress)
# ============================================================
import shutil

DRIVE_OUT_DIR = Path("/content/drive/MyDrive/birdclef-2026/exp012/output")
DRIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)

_orig_train_fold = train_fold

def train_fold(fold_k):
    print(f"=== train_fold({fold_k}) ===")
    # Pre-load existing best checkpoints from Drive if present (resume support)
    for fname in [f"fold{fold_k}_best_ns22.pt", f"fold{fold_k}_best_macro.pt",
                  f"fold{fold_k}_history.json"]:
        d = DRIVE_OUT_DIR / fname
        l = OUT_DIR / fname
        if d.exists() and not l.exists():
            shutil.copy(str(d), str(l))
            print(f"  pre-loaded {fname} from Drive")

    result = _orig_train_fold(fold_k)

    # Mirror outputs to Drive after fold
    for src in OUT_DIR.glob(f"fold{fold_k}_*"):
        dst = DRIVE_OUT_DIR / src.name
        shutil.copy2(str(src), str(dst))
        print(f"  mirrored {src.name} -> Drive ({dst.stat().st_size/1e6:.1f}MB)")
    return result

print("OK Drive-mirroring train_fold ready")
""", "drive_mirror"))

# ============================================================
# 9. Fold loop + ONNX export (Tucker cell 16)
# ============================================================
fold_loop_src = cell_src(ref_nb["cells"][16])
cells.append(code_cell(fold_loop_src, "fold_loop"))

# ============================================================
# 10. Mirror ONNX to Drive
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Copy ONNX outputs to Drive
# ============================================================
import shutil
for src in OUT_DIR.glob("sed_distill_fold*.onnx"):
    dst = DRIVE_OUT_DIR / src.name
    shutil.copy2(str(src), str(dst))
    print(f"  mirrored {src.name} -> Drive ({dst.stat().st_size/1e6:.1f}MB)")

print("\nFinal Drive contents:")
for p in sorted(DRIVE_OUT_DIR.iterdir()):
    print(f"  {p.name} ({p.stat().st_size/1e6:.1f}MB)")
""", "mirror_onnx"))

# ============================================================
# 11. Bundle ONNX into Kaggle dataset
# ============================================================
cells.append(code_cell(r"""# ============================================================
# Create Kaggle Dataset with the 5-fold ONNX bundle
# (run only once; for re-uploads use kaggle datasets version)
# ============================================================
import json, subprocess

DATASET_USER  = "maekeso"
DATASET_SLUG  = "birdclef2026-exp012-sed-onnx"
DATASET_TITLE = "birdclef2026 exp012 sed onnx"

bundle_dir = Path("/content/exp012_onnx_bundle")
bundle_dir.mkdir(parents=True, exist_ok=True)
for src in DRIVE_OUT_DIR.glob("sed_distill_fold*.onnx"):
    shutil.copy2(str(src), str(bundle_dir / src.name))

meta = {
    "title": DATASET_TITLE,
    "id":    f"{DATASET_USER}/{DATASET_SLUG}",
    "licenses": [{"name": "CC0-1.0"}],
}
(bundle_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

# First upload (will fail with "already exists" if you re-run; use version below in that case)
print("Trying initial dataset create...")
r = subprocess.run(f"kaggle datasets create -p {bundle_dir} -q",
                   shell=True, capture_output=True, text=True)
print("stdout:", r.stdout)
print("stderr:", r.stderr)

if "already exists" in (r.stdout + r.stderr).lower():
    print("\nDataset already exists, creating new version...")
    r2 = subprocess.run(
        f"kaggle datasets version -p {bundle_dir} -m 'exp012 5-fold ONNX update' --dir-mode zip -q",
        shell=True, capture_output=True, text=True)
    print("stdout:", r2.stdout)
    print("stderr:", r2.stderr)

print(f"\nUse in Kaggle inference NB:")
print(f"  dataset_sources: ['{DATASET_USER}/{DATASET_SLUG}']")
print(f"  path on Kaggle: /kaggle/input/{DATASET_SLUG}/")
""", "upload_kaggle"))

cells.append(md_cell(r"""## Done

Use the uploaded Kaggle dataset `maekeso/birdclef2026-exp012-sed-onnx` as `dataset_sources` in your inference notebook (`nb_submit.ipynb`). The 5 ONNX files are:
- `sed_distill_fold0.onnx`
- ...
- `sed_distill_fold4.onnx`

Inference: ensemble (mean of 5-fold logits) + 5-tap Gaussian smoothing + sigmoid → submission.csv. Expected LB ~0.917.
""", "footer"))

# Build notebook
nb_out = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_train_colab.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote: {out_path}")
print(f"Cells: {len(cells)}")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
