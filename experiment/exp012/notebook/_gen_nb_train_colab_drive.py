"""Generate exp012 nb_train_colab_drive.ipynb — zip-streaming version.

Tucker waveform-cache zip is ~98GB and unzipped ~100GB. Total 200GB exceeds
Colab Pro disk (150GB). Strategy: keep the zip, read individual .pt files
directly from zip via zipfile.ZipFile (no extraction).

Cell layout:
  1. Setup (Drive mount, kaggle.json, PROJECT_ROOT)
  2. DL waveform-cache zip + perch-onnx (no full extract; just CSVs from zip)
  3. Install onnxruntime wheel
  4. Imports + config (paths point to /content/data)
  5-7. Load data → Model → Data pipeline (Tucker original)
  8. Patch load_int16 to read from zip
  9. Training fn
  10. Drive output mirror patch
  11. Fold loop
  12. Mirror ONNX to Drive
  13. Upload Kaggle dataset

Disk usage: ~100GB (zip 97.7GB + CSVs + perch-onnx + outputs).

Run: python _gen_nb_train_colab_drive.py  -> writes nb_train_colab_drive.ipynb
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
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

# ===== Header =====
cells.append(md_cell(r"""# exp012 — Distilled SED Training (Colab Pro, zip-streaming)

Tucker waveform-cache zip (97.7GB) is downloaded but NOT extracted. Training
reads .pt files directly from zip via `zipfile.ZipFile`, fitting in Colab Pro's
150GB disk.

## Disk usage
| 項目 | サイズ |
|---|---|
| waveform-cache zip (no extract) | 97.7GB |
| Extracted CSVs from zip | <100MB |
| perch-onnx | 1.5GB |
| competition CSVs | <50MB |
| Output (mirrored to Drive) | <500MB |
| **合計** | **~100GB** |

## Prerequisites
1. Local project at `C:\Users\maeke\work\kaggle\birdclef-2026` synced to Drive
2. `kaggle.json` (with KGAT_ token in `key`) at `/content/drive/MyDrive/kaggle.json`

## Time (Colab Pro A100)
- Cell 2 zip DL: 30-45 min
- 5-fold training: 4-8h (zip read overhead +10-30%)
- Total: ~5-9h, fits in 24h Colab Pro session

## 注意
- `num_workers=0` 維持 (Tucker original)。1+ にする場合 zip パッチ要修正
- 切断時は zip 再 DL 必要 → 1 session で完走推奨
""", "hdr"))

# ===== Setup =====
cells.append(code_cell(r"""# ============================================================
# Setup: Drive mount + project root + kaggle.json
# ============================================================
!pip install -q timm onnx onnxruntime-gpu librosa kaggle scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

# Auto-find project folder
print("Searching for birdclef-2026 on Drive...")
r = subprocess.run(
    ["find", "/content/drive/MyDrive", "-maxdepth", "6", "-name", "birdclef-2026", "-type", "d"],
    capture_output=True, text=True)
candidates = [Path(p) for p in r.stdout.strip().split("\n") if p]
print("Candidates:")
for c in candidates: print(f"  {c}")

PROJECT_ROOT = None
for c in candidates:
    if (c / "experiment" / "exp012").exists():
        PROJECT_ROOT = c; break
if PROJECT_ROOT is None and candidates:
    PROJECT_ROOT = candidates[0]
assert PROJECT_ROOT is not None, "birdclef-2026 not found on Drive"
print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")

# kaggle.json setup
KJ_CANDIDATES = [
    Path("/content/drive/MyDrive/kaggle.json"),
    PROJECT_ROOT / "kaggle.json",
]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
assert KJ is not None, f"kaggle.json not found in {KJ_CANDIDATES}"
KAGGLE_CFG = Path.home() / ".kaggle"
KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
creds = json.loads(KJ.read_text())
if creds.get("key", "").startswith("KGAT_"):
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]
print(f"kaggle.json: {KJ}")

# Working dirs
EXP_DIR = PROJECT_ROOT / "experiment" / "exp012"
OUT_DIR_DRIVE = EXP_DIR / "output"
OUT_DIR_DRIVE.mkdir(parents=True, exist_ok=True)

LOCAL_DATA = Path("/content/data")
LOCAL_OUT  = Path("/content/output")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUT.mkdir(parents=True, exist_ok=True)

print(f"\nDrive output: {OUT_DIR_DRIVE}")
print(f"Local data:   {LOCAL_DATA}")
print(f"Local output: {LOCAL_OUT}")
""", "setup"))

# ===== Data DL (zip kept, CSVs extracted) =====
cells.append(code_cell(r"""# ============================================================
# Data DL — zip-streaming strategy
# - waveform-cache: zip だけ DL、解凍せず保持。CSV だけ zip から extract
# - perch-onnx: 普通に DL+unzip (1.5GB)
# ============================================================
import time, zipfile, shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
print("kaggle authenticated")

# 既存の壊れた状態をクリーンアップ
WC_DIR = LOCAL_DATA / "waveform-cache"
if WC_DIR.exists():
    print(f"Removing existing {WC_DIR}")
    shutil.rmtree(WC_DIR)
WC_DIR.mkdir(parents=True, exist_ok=True)

# 1) Competition CSVs (Drive から copy)
comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
    src = PROJECT_ROOT / fn
    dst = comp / fn
    if src.exists() and not dst.exists():
        shutil.copy2(str(src), str(dst))
        print(f"  competition/{fn} OK (from Drive)")

# 2) waveform-cache zip だけ DL (97.7GB, 30-45分)
WC_ZIP = WC_DIR / "birdclef-2026-waveform-cache.zip"
if not WC_ZIP.exists():
    print(f"\nDownloading waveform-cache zip (no extract)...")
    t0 = time.time()
    api.dataset_download_files(
        "tuckerarrants/birdclef-2026-waveform-cache",
        path=str(WC_DIR),
        unzip=False,           # 解凍しない
        quiet=False)
    print(f"  done in {time.time()-t0:.0f}s, {WC_ZIP.stat().st_size/1e9:.1f}GB")
else:
    print(f"\nzip already present: {WC_ZIP.stat().st_size/1e9:.1f}GB")

# 3) CSV だけ zip から extract (小さい、高速)
print("\nExtracting CSVs from zip (no audio files)...")
with zipfile.ZipFile(str(WC_ZIP)) as zf:
    for info in zf.infolist():
        if info.filename.endswith(".csv"):
            target = WC_DIR / info.filename
            if not target.exists():
                zf.extract(info.filename, str(WC_DIR))
                print(f"  extracted {info.filename} ({info.file_size/1e6:.2f}MB)")

# 4) Perch ONNX (1.5GB, 普通に unzip)
po_dir = LOCAL_DATA / "perch-onnx"
if not (po_dir / "perch_v2_no_dft.onnx").exists():
    po_dir.mkdir(parents=True, exist_ok=True)
    print("\nDownloading perch-onnx...")
    t0 = time.time()
    api.dataset_download_files(
        "tuckerarrants/perch-v2-no-dft-onnx",
        path=str(po_dir),
        unzip=True,
        quiet=False)
    print(f"  done in {time.time()-t0:.0f}s")
else:
    print("\nperch-onnx already present")

# 5) Disk summary
print("\n=== /content disk ===")
r = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
print(r.stdout)

print("=== /content/data summary ===")
for sub in sorted(LOCAL_DATA.iterdir()):
    if sub.is_dir():
        n = sum(1 for _ in sub.rglob("*") if _.is_file())
        sz = sum(_.stat().st_size for _ in sub.rglob("*") if _.is_file()) / 1e9
        print(f"  {sub.name}/  ({n} files, {sz:.1f}GB)")
""", "dl_data"))

# ===== Install onnxruntime wheel =====
cells.append(code_cell(r"""# ============================================================
# Install Tucker's bundled onnxruntime wheel (matches Perch ONNX format)
# ============================================================
WHEEL = next(LOCAL_DATA.rglob("onnxruntime-*.whl"), None)
if WHEEL is not None:
    !pip install -q {WHEEL}
    print(f"Installed {WHEEL.name}")
else:
    print("No bundled wheel; using onnxruntime-gpu installed earlier")
""", "install_ort"))

# ===== Imports + config =====
c4_src = cell_src(ref_nb["cells"][4])
c4_src = c4_src.replace('MODE = "infer"', 'MODE = "train"')
c4_src = re.sub(
    r"if\s+MODE\s*==\s*['\"]train['\"][^\n]*\n\s+DEBUG\s*=\s*True[^\n]*\n\s*else\s*:\s*\n\s+DEBUG\s*=\s*False",
    "DEBUG = False  # Colab Pro: full training",
    c4_src, count=1)
c4_src = c4_src.replace("FOLDS  = [0, 1, 2, 3, 4]",
                        "FOLDS  = [0, 1, 2, 3, 4]  # all 5 folds in one Colab Pro session")
c4_src = c4_src.replace("BATCH  = 16 if DEBUG else 64",
                        "BATCH  = 64  # A100 40GB; drop to 32 if V100 OOM")
c4_src = c4_src.replace(
    'COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")',
    'COMP_DIR = Path("/content/data/competition")')
c4_src = c4_src.replace(
    'WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")',
    'WAVEFORM_CACHE_DIR = Path("/content/data/waveform-cache/waveform_cache")  # 仮想 path; 実際は zip から読む')
c4_src = c4_src.replace(
    'PERCH_ONNX_PATH = Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx")',
    'PERCH_ONNX_PATH = Path("/content/data/perch-onnx/perch_v2_no_dft.onnx")')
c4_src = c4_src.replace(
    'OUT_DIR = Path("/kaggle/working")',
    'OUT_DIR = Path("/content/output")  # Drive にミラーされる')

cells.append(code_cell(c4_src, "config"))

# ===== Load data =====
cells.append(code_cell(cell_src(ref_nb["cells"][7]), "load_data"))

# ===== Model defs =====
cells.append(code_cell(cell_src(ref_nb["cells"][10]), "model_defs"))

# ===== Data pipeline (Tucker original) =====
cells.append(code_cell(cell_src(ref_nb["cells"][12]), "data_pipeline"))

# ===== ZIP STREAMING PATCH (the new key cell) =====
cells.append(code_cell(r"""# ============================================================
# 重要: load_int16 を zip 直読みに置き換え
# waveform-cache を解凍せず zip のまま使うパッチ
# ============================================================
import zipfile, io, threading
from pathlib import Path
import torch

_WC_ZIP_PATH = Path("/content/data/waveform-cache/birdclef-2026-waveform-cache.zip")
assert _WC_ZIP_PATH.exists(), f"zip missing: {_WC_ZIP_PATH}"

# zip 内パスのプレフィックスを除去するため、サンプルから推定
with zipfile.ZipFile(str(_WC_ZIP_PATH)) as _zf_probe:
    _names_sample = [n for n in _zf_probe.namelist()[:50] if n.endswith(".pt")]
    if _names_sample:
        print(f"sample zip pt path: {_names_sample[0]}")

# 各 worker (or main thread) で独立した ZipFile を持つ
# ZipFile は thread-safe ではないので、thread.local + lock
_zf_local = threading.local()

def _get_zip():
    z = getattr(_zf_local, "z", None)
    if z is None:
        z = zipfile.ZipFile(str(_WC_ZIP_PATH), "r")
        _zf_local.z = z
    return z

# Tucker original の load_int16 を上書き
def load_int16(path):
    # path は WAVEFORM_CACHE_DIR / cache_file 形式 → zip 内 path に変換
    p = path if isinstance(path, Path) else Path(path)
    # WAVEFORM_CACHE_DIR より下の相対 path
    try:
        rel = p.relative_to(Path("/content/data/waveform-cache"))
    except ValueError:
        rel = p
    rel_str = str(rel).replace("\\", "/")
    z = _get_zip()
    with z.open(rel_str) as f:
        data = f.read()
    waveform_int16 = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return waveform_int16.float() / 32767.0

# 動作確認
import pandas as pd
meta_path = Path("/content/data/waveform-cache/waveform_cache/audio_cache_meta.csv")
if meta_path.exists():
    meta = pd.read_csv(meta_path)
    cf = meta.iloc[0]["cache_file"]
    # WAVEFORM_CACHE_DIR を使う (= /content/data/waveform-cache/waveform_cache)
    test_path = WAVEFORM_CACHE_DIR / cf
    print(f"Test cache_file: {cf}")
    print(f"  full path: {test_path}")
    sample = load_int16(test_path)
    print(f"  sample shape={sample.shape}, dtype={sample.dtype}")
    print(f"  min={sample.min():.4f}, max={sample.max():.4f}")
else:
    print("audio_cache_meta.csv not found — extract from zip in Cell 2 first")

# キャッシュをクリア (古い load_int16 で読まれてた可能性のある)
_FC.clear() if "_FC" in dir() else None
_SC_CACHE.clear() if "_SC_CACHE" in dir() else None
print("\nload_int16 patched — reads directly from zip")
""", "zip_patch"))

# ===== Training fn =====
cells.append(code_cell(cell_src(ref_nb["cells"][14]), "train_fn"))

# ===== Drive output mirror patch =====
cells.append(code_cell(r"""# ============================================================
# Patch: mirror outputs to Drive after each fold (resumability)
# ============================================================
import shutil
from pathlib import Path

DRIVE_OUT_DIR = OUT_DIR_DRIVE
_orig_train_fold = train_fold

def train_fold(fold_k):
    print(f"=== train_fold({fold_k}) ===")
    for fname in [f"fold{fold_k}_best_ns22.pt", f"fold{fold_k}_best_macro.pt",
                  f"fold{fold_k}_history.json"]:
        d = DRIVE_OUT_DIR / fname
        l = OUT_DIR / fname
        if d.exists() and not l.exists():
            shutil.copy(str(d), str(l))
            print(f"  pre-loaded {fname} from Drive")

    result = _orig_train_fold(fold_k)

    for src in OUT_DIR.glob(f"fold{fold_k}_*"):
        dst = DRIVE_OUT_DIR / src.name
        shutil.copy2(str(src), str(dst))
        print(f"  mirrored {src.name} -> Drive ({dst.stat().st_size/1e6:.1f}MB)")
    return result

print("OK Drive-mirroring train_fold ready")
""", "drive_mirror"))

# ===== Fold loop + ONNX export =====
cells.append(code_cell(cell_src(ref_nb["cells"][16]), "fold_loop"))

# ===== Mirror ONNX =====
cells.append(code_cell(r"""# Mirror ONNX outputs to Drive
import shutil
from pathlib import Path

for src in OUT_DIR.glob("sed_distill_fold*.onnx"):
    dst = DRIVE_OUT_DIR / src.name
    shutil.copy2(str(src), str(dst))
    print(f"  mirrored {src.name} -> Drive ({dst.stat().st_size/1e6:.1f}MB)")

print("\nFinal Drive contents:")
for p in sorted(DRIVE_OUT_DIR.iterdir()):
    print(f"  {p.name} ({p.stat().st_size/1e6:.1f}MB)")
""", "mirror_onnx"))

# ===== Upload Kaggle dataset =====
cells.append(code_cell(r"""# ============================================================
# Bundle 5-fold ONNX → Kaggle Dataset (for inference NB)
# ============================================================
import json, shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

DATASET_USER  = "maekeso"
DATASET_SLUG  = "birdclef2026-exp012-sed-onnx"
DATASET_TITLE = "birdclef2026 exp012 sed onnx"

bundle = Path("/content/exp012_onnx_bundle")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)
for src in DRIVE_OUT_DIR.glob("sed_distill_fold*.onnx"):
    shutil.copy2(str(src), str(bundle / src.name))

(bundle / "dataset-metadata.json").write_text(json.dumps({
    "title": DATASET_TITLE,
    "id":    f"{DATASET_USER}/{DATASET_SLUG}",
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))

print("Trying initial dataset create...")
try:
    api.dataset_create_new(folder=str(bundle), dir_mode="skip", quiet=False)
    print("Created new dataset")
except Exception as e:
    msg = str(e)
    print(f"create_new failed: {msg[:200]}")
    if "already exists" in msg.lower() or "404" in msg:
        print("\nCreating new version...")
        api.dataset_create_version(folder=str(bundle), version_notes="exp012 5-fold ONNX update",
                                   dir_mode="skip", quiet=False)
        print("Version created")

print(f"\nFor Kaggle inference NB:")
print(f"  dataset_sources: ['{DATASET_USER}/{DATASET_SLUG}']")
""", "upload_kaggle"))

cells.append(md_cell(r"""## 完了

`maekeso/birdclef2026-exp012-sed-onnx` を Kaggle 推論NB の `dataset_sources` に指定して提出 → **LB 0.917+** 期待。
""", "footer"))

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

out_path = HERE / "nb_train_colab_drive.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote: {out_path}")
print(f"Cells: {len(cells)}")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
