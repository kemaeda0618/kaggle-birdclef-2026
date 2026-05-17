"""Generate exp013 NB2 v2: HGNetV2-B0 + Tucker recipe + Perch distill + pseudo (Drive-based).

R1 v2 学生 SED training。Tucker recipe を踏襲しつつ:
1. Backbone を **HGNetV2-B0** に変更 (Tucker EffNet との arch diversity)
2. **Tucker 97GB waveform-cache zip を廃止**、train_audio/SS は Drive から copy
3. `load_focal` / `load_sc_waveform_from` を **.ogg 直読** にパッチ

総 disk 使用: ~22 GB (vs 旧 112 GB)、Colab Pro 150GB に余裕。

データソース:
- Tucker waveform-cache zip (97GB, zip-streaming):
  - focal: train_audio (~46k, hard label primary+secondary)
  - labeled_sc: 739 windows from 66 labeled soundscapes (hard label)
- 新規: BC2026 train_soundscapes (~10GB, .ogg, 10,658 files):
  - pseudo_sc: NB1 v2 pseudo CSV (soft label)

Source mix shares: focal=0.7, labeled_sc=0.10, pseudo_sc=0.20

Disk usage on Colab Pro:
- waveform-cache zip (no extract):  98 GB
- train_soundscapes audio:          10 GB
- perch-onnx + bundled wheel:        2 GB
- pseudo CSV + competition CSVs:    <1 GB
- output (mirrored to Drive):       <1 GB
- Total:                           ~112 GB (fits in 150GB Colab Pro disk)

Folds: initially [0] for AUC validation, expand to [0..4] after.

Cell layout:
  1. Setup (Drive mount, kaggle.json)
  2. Data DL (zip-stream, train_soundscapes, pseudo CSV)
  3. Install ORT wheel
  4. Imports + Config (Tucker base, modified for Colab + pseudo)
  5. Load data (incl. pseudo CSV)
  6. Model defs (Tucker verbatim)
  7. Data pipeline (Tucker + new PseudoScDS)
  8. Build active datasets (3 sources)
  9. ZIP streaming patch
  10. Training loop
  11. Fold loop (training + ckpt save)
  12. ONNX export
  13. Drive mirror
  14. (Optional) Kaggle Dataset upload

Run: python _gen_nb_pl2_sed_train_v2.py  -> writes nb_pl2_sed_train_v2.ipynb
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
ref_nb = json.loads(
    (HERE.parent.parent / "exp012" / "notebook" / "_reference_tucker.ipynb").read_text(encoding="utf-8")
)


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

# =============================================================================
# Header
# =============================================================================
cells.append(md_cell(r"""# exp013 NB2 v2 — R1 v2: HGNetV2-B0 + Tucker recipe + Perch distill + pseudo

Tucker recipe を **HGNetV2-B0 backbone** に変更し、arch diversity で R1 v1 失敗 (Tucker クローン化) を回避。
**Drive-based 構成**: Tucker 97GB zip 廃止、train_audio/SS は Drive から copy、.ogg 直読。

## なぜ HGNetV2-B0?
- Tucker は EffNet-B0 SED、我々の v8 にも入ってる → 同 arch だと blend で重複
- **HGNetV2-B0** は別 arch (GhostConv 系)、Tucker と独立な特徴抽出経路
- **Natsume が BC2026 Discussion (5/10) で実証**: hgnetv2_b0 solo 0.931 → +pseudo concat で **0.943 (+0.012)** 達成
- BC2025 12位 223223 も HGNetV2 採用

## データソース (Drive 経由、97GB DL 廃止)
| Source | サイズ | 取得方法 | ラベル | Share |
|---|---|---|---|---|
| **train_audio** (.ogg) | ~15 GB | Drive copy ~10-15 min | focal=hard | 0.85 |
| **train_soundscapes** (.ogg) | ~5 GB | Drive copy ~3-5 min | sc=hard / pseudo soft | 0.05 / 0.10 |
| Tucker meta CSVs | 2.5 MB | Drive copy (audio_cache_meta + sc_cache_meta) | - | - |
| Pseudo CSV (NB1 v2 出力) | 370 MB | Kaggle DL ~1 min | soft | (sc に統合) |
| Perch ONNX + wheel | 1.5 GB | Kaggle DL ~3 min | distillation teacher | - |
| **合計 disk** | **~22 GB** (vs 旧 112 GB) | | | |

## 学習
- Backbone: **hgnetv2_b0.ssld_stage2_ft_in1k** (SSLD pretraining)
- Distillation: ON (alpha=1.0, MSE to Perch v2 1536-d、Tucker 流の stop-grad 構造)
- Loss: 0.5 * BCE_clip + 0.5 * BCE_frame_max + 1.0 * MSE_distill
- **MIXUP_HARD = False** (soft label を保持、pseudo の soft 性に整合)
- **LR = 3e-4** (HGNet は 5e-4 不安定報告)
- Folds: **[0] のみ** (AUC 確認後 [1..4] 追加)
- 25 epoch, batch=64 (A100 40GB)

## 出力
- `/content/output/r1v2_fold{k}_best_ns22.pth`
- `/content/output/r1v2_fold{k}_best_macro.pth`
- `/content/output/r1v2_fold{k}.onnx`
- すべて `/content/drive/MyDrive/birdclef2026/exp013/r1v2/` にミラー

## 想定時間 (Colab Pro A100, 1 fold)
- データ DL: 30-50 min (zip 97GB が支配的)
- 学習 25 epoch: 1.5-2.5h
- ONNX export + Drive mirror: 5 min
- **合計 1 fold: ~2-3.5h**
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — Drive mount, project root, kaggle.json
# ============================================================
!pip install -q timm onnx onnxruntime-gpu librosa kaggle scipy soundfile

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

# ★ ユーザー指定の Drive folder (memory: user_colab_drive_folder.md)
DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
assert DRIVE_INPUT_DIR.exists(), f"Drive input folder missing: {DRIVE_INPUT_DIR}"

DRIVE_OUTPUT_DIR = DRIVE_INPUT_DIR / "output" / "exp013" / "r1v2"
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Drive input:  {DRIVE_INPUT_DIR}")
print(f"Drive output: {DRIVE_OUTPUT_DIR}")

# Search for project code (experiment/exp013) on Drive — may be different from input data dir
print("\nSearching for project code on Drive...")
r = subprocess.run(
    ["find", "/content/drive/MyDrive", "-maxdepth", "6", "-name", "birdclef-2026", "-type", "d"],
    capture_output=True, text=True)
candidates = [Path(p) for p in r.stdout.strip().split("\n") if p]
PROJECT_ROOT = None
for c in candidates:
    if (c / "experiment" / "exp013").exists():
        PROJECT_ROOT = c; break
if PROJECT_ROOT is None and candidates:
    PROJECT_ROOT = candidates[0]
print(f"PROJECT_ROOT (code): {PROJECT_ROOT}")

# kaggle.json setup
KJ_CANDIDATES = [
    Path("/content/drive/MyDrive/kaggle.json"),
    DRIVE_INPUT_DIR / "kaggle.json",
]
if PROJECT_ROOT is not None:
    KJ_CANDIDATES.append(PROJECT_ROOT / "kaggle.json")
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
OUT_DIR_DRIVE = DRIVE_OUTPUT_DIR   # alias for backward compat in later cells

LOCAL_DATA = Path("/content/data")
LOCAL_OUT  = Path("/content/output")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUT.mkdir(parents=True, exist_ok=True)

print(f"\nDrive output: {OUT_DIR_DRIVE}")
print(f"Local data:   {LOCAL_DATA}")
print(f"Local output: {LOCAL_OUT}")
""", "setup"))

# =============================================================================
# Cell 2: Data DL — zip-stream + train_soundscapes + pseudo CSV
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Data prep — Drive copy + Kaggle download (NO Tucker waveform-cache)
# ============================================================
# v2 (Drive-based): Tucker 97GB zip 廃止、train_audio/SS は Drive から copy、.ogg 直読
import time, zipfile, shutil, subprocess
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
print("kaggle authenticated")

T0_total = time.time()

# 1) Competition CSVs (Drive から copy — DRIVE_INPUT_DIR 優先、無ければ PROJECT_ROOT)
comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
    src_candidates = [DRIVE_INPUT_DIR / fn]
    if PROJECT_ROOT is not None:
        src_candidates.append(PROJECT_ROOT / fn)
    src = next((s for s in src_candidates if s.exists()), None)
    dst = comp / fn
    if src is not None and not dst.exists():
        shutil.copy2(str(src), str(dst))
        print(f"  competition/{fn} OK (from Drive: {src.parent.name})")
    elif src is None:
        # Fallback: download from Kaggle
        print(f"  {fn} not on Drive, DL from Kaggle...")
        api.competition_download_file("birdclef-2026", file_name=fn, path=str(comp))
        z = comp / (fn + ".zip")
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(comp)
            z.unlink()

# 2) Tucker meta CSVs (Drive にあれば copy、なければ Kaggle DL)
#    waveform-cache の zip 自体は不要、CSV だけ欲しい
WC_DIR = LOCAL_DATA / "waveform-cache"
(WC_DIR / "waveform_cache").mkdir(parents=True, exist_ok=True)
META_NEEDED = ["audio_cache_meta.csv", "soundscape_cache_meta.csv", "soundscape_file_meta.csv"]
# Try Drive paths
DRIVE_META_CANDIDATES = [
    DRIVE_INPUT_DIR / "tucker_meta",
]
if PROJECT_ROOT is not None:
    DRIVE_META_CANDIDATES += [
        PROJECT_ROOT / "experiment" / "exp013" / "notebook" / "tucker_meta_test",
        PROJECT_ROOT / "tucker_meta",
    ]
for fn in META_NEEDED:
    dst = WC_DIR / "waveform_cache" / fn
    if dst.exists():
        print(f"  meta {fn} already present")
        continue
    src_found = None
    for cand_dir in DRIVE_META_CANDIDATES:
        src = cand_dir / fn
        if src.exists():
            src_found = src
            break
    if src_found:
        shutil.copy2(str(src_found), str(dst))
        print(f"  meta {fn} OK (from Drive)")
    else:
        # Fallback: download from Tucker dataset
        print(f"  meta {fn} not on Drive, DL from Kaggle...")
        try:
            api.dataset_download_file(
                "tuckerarrants/birdclef-2026-waveform-cache",
                file_name=f"waveform_cache/{fn}",
                path=str(WC_DIR / "waveform_cache"),
            )
            z = WC_DIR / "waveform_cache" / (fn + ".zip")
            if z.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(WC_DIR / "waveform_cache")
                z.unlink()
        except Exception as e:
            print(f"    failed: {e}")
            if fn != "soundscape_file_meta.csv":
                raise  # required
            print(f"    {fn} optional, will disable focal-SC mixup")

# 3) train_audio (Drive → /content/data)
TA_DIR = LOCAL_DATA / "train_audio"
DRIVE_TA = DRIVE_INPUT_DIR / "train_audio"
if not TA_DIR.exists() or sum(1 for _ in TA_DIR.rglob("*.ogg")) < 1000:
    if DRIVE_TA.exists():
        n_drive = sum(1 for _ in DRIVE_TA.rglob("*.ogg"))
        print(f"\nDrive train_audio: {n_drive} .ogg files at {DRIVE_TA}")
        print(f"Copying to local /content/data/train_audio (~15GB, ~10-15 min)...")
        t0 = time.time()
        shutil.copytree(str(DRIVE_TA), str(TA_DIR), dirs_exist_ok=True)
        n = sum(1 for _ in TA_DIR.rglob("*.ogg"))
        print(f"  done in {time.time()-t0:.0f}s, {n} .ogg files (drive had {n_drive})")
    else:
        raise FileNotFoundError(
            f"train_audio not on Drive at {DRIVE_TA}. "
            f"Drive アップロード未完了 or path 違いの可能性")
else:
    n = sum(1 for _ in TA_DIR.rglob("*.ogg"))
    print(f"\ntrain_audio already present in /content/data: {n} .ogg files")

# 4) train_soundscapes (Drive → /content/data)
TS_DIR = LOCAL_DATA / "train_soundscapes"
DRIVE_TS = DRIVE_INPUT_DIR / "train_soundscapes"
if not TS_DIR.exists() or sum(1 for _ in TS_DIR.glob("*.ogg")) < 1000:
    if DRIVE_TS.exists():
        n_drive = sum(1 for _ in DRIVE_TS.glob("*.ogg"))
        print(f"\nDrive train_soundscapes: {n_drive} .ogg files at {DRIVE_TS}")
        print(f"Copying to local (~5GB, ~3-5 min)...")
        t0 = time.time()
        shutil.copytree(str(DRIVE_TS), str(TS_DIR), dirs_exist_ok=True)
        n_ogg = sum(1 for _ in TS_DIR.glob("*.ogg"))
        print(f"  done in {time.time()-t0:.0f}s, {n_ogg} .ogg files (drive had {n_drive})")
    else:
        raise FileNotFoundError(
            f"train_soundscapes not on Drive at {DRIVE_TS}. "
            f"Drive アップロード未完了 or path 違いの可能性")
else:
    n_ogg = sum(1 for _ in TS_DIR.glob("*.ogg"))
    print(f"\ntrain_soundscapes already present in /content/data: {n_ogg} .ogg files")

# 5) Pseudo CSV (NB1 v2 出力)
PSEUDO_DIR = LOCAL_DATA / "pseudo-r1-v2"
PSEUDO_CSV = PSEUDO_DIR / "pseudo_labels.csv"
if not PSEUDO_CSV.exists():
    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading pseudo-r1-v2 CSV...")
    t0 = time.time()
    api.dataset_download_files(
        "maekeso/birdclef2026-exp013-pseudo-r1-v2",
        path=str(PSEUDO_DIR), unzip=True, quiet=False)
    # Dataset may unzip to a nested folder; locate the CSV
    if not PSEUDO_CSV.exists():
        cands = list(PSEUDO_DIR.rglob("pseudo_labels.csv"))
        if cands:
            shutil.move(str(cands[0]), str(PSEUDO_CSV))
    print(f"  done in {time.time()-t0:.0f}s, {PSEUDO_CSV.stat().st_size/1e6:.1f}MB")
else:
    print(f"\npseudo CSV already present: {PSEUDO_CSV.stat().st_size/1e6:.1f}MB")

# 6) Perch ONNX (1.5GB)
po_dir = LOCAL_DATA / "perch-onnx"
if not (po_dir / "perch_v2_no_dft.onnx").exists():
    po_dir.mkdir(parents=True, exist_ok=True)
    print("\nDownloading perch-onnx...")
    t0 = time.time()
    api.dataset_download_files(
        "tuckerarrants/perch-v2-no-dft-onnx",
        path=str(po_dir), unzip=True, quiet=False)
    print(f"  done in {time.time()-t0:.0f}s")
else:
    print("\nperch-onnx already present")

# 7) Disk summary
print(f"\n=== Total DL time: {(time.time()-T0_total)/60:.1f} min ===")
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

# =============================================================================
# Cell 3: Install bundled ORT wheel
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Install Tucker's bundled onnxruntime wheel (matches Perch ONNX format)
# ============================================================
WHEEL = next(LOCAL_DATA.rglob("onnxruntime-*.whl"), None)
if WHEEL is not None:
    !pip install -q {WHEEL}
    print(f"Installed {WHEEL.name}")
else:
    print("No bundled wheel; using onnxruntime-gpu installed earlier")
""", "install_ort"))

# =============================================================================
# Cell 4: Imports + Config
# Take Tucker's cell 4, modify paths, FOLDS, BATCH, MODE/DEBUG, add pseudo config.
# =============================================================================
c4_src = cell_src(ref_nb["cells"][4])
c4_src = c4_src.replace('MODE = "infer"', 'MODE = "train"')
c4_src = re.sub(
    r"if\s+MODE\s*==\s*['\"]train['\"][^\n]*\n\s+DEBUG\s*=\s*True[^\n]*\n\s*else\s*:\s*\n\s+DEBUG\s*=\s*False",
    "DEBUG = False  # Colab Pro: full training",
    c4_src, count=1)
c4_src = c4_src.replace(
    "FOLDS  = [0, 1, 2, 3, 4]",
    "FOLDS  = [0]  # initial 1-fold AUC check; expand to [0,1,2,3,4] after")
c4_src = c4_src.replace(
    "BATCH  = 16 if DEBUG else 64",
    "BATCH  = 64  # A100 40GB; drop to 32 if V100 OOM")
c4_src = c4_src.replace(
    'COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")',
    'COMP_DIR = Path("/content/data/competition")')
c4_src = c4_src.replace(
    'WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")',
    'WAVEFORM_CACHE_DIR = Path("/content/data/waveform-cache/waveform_cache")  # virtual; actual reads via zip')
c4_src = c4_src.replace(
    'PERCH_ONNX_PATH = Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx")',
    'PERCH_ONNX_PATH = Path("/content/data/perch-onnx/perch_v2_no_dft.onnx")')
c4_src = c4_src.replace(
    'OUT_DIR = Path("/kaggle/working")',
    'OUT_DIR = Path("/content/output")  # mirrored to Drive')

# ★ Backbone を HGNetV2-B0 に変更 (Tucker と arch diversity 確保)
c4_src = c4_src.replace(
    'BACKBONE_NAME = "tf_efficientnet_b0.ns_jft_in1k"',
    'BACKBONE_NAME = "hgnetv2_b0.ssld_stage2_ft_in1k"  # ★ Tucker と arch 違い、Natsume 実証 +0.012')

# ★ Soft mixup mode (現 True=union → False=weighted blend、soft pseudo に整合)
c4_src = c4_src.replace(
    "MIXUP_HARD         = True    # union labels (hard) vs weighted blend (soft)",
    "MIXUP_HARD         = False   # ★ soft label を保持 (pseudo の soft 性を活かす)")

# ★ LR を少し下げる (HGNetV2 は Salman/Boredom 報告で 3e-4 程度が安定)
c4_src = c4_src.replace(
    "LR     = 5e-4",
    "LR     = 3e-4  # ★ HGNetV2 は 5e-4 だと不安定報告あり、保守的に")

# Replace Tucker's SHARES/SOURCE_WEIGHTS/ACTIVE_SOURCES block with v2 config (3 sources)
c4_src = c4_src.replace(
    'ACTIVE_SOURCES = ["focal", "sc"]\nSHARES = {"focal": 0.9, "sc": 0.1}\nSOURCE_WEIGHTS = {\n    "focal":         1.0,\n    "focal_missing": 0.0,\n    "sc":            1.0,\n}',
    """# --- v2: pseudo SC source ---
USE_PSEUDO_SC      = True
PSEUDO_SC_SHARE    = 0.20

ACTIVE_SOURCES = ["focal", "sc", "pseudo_sc"]
# ★ Tucker original (focal 0.9, sc 0.1) に近い保守的比率、pseudo は薄く追加
SHARES = {"focal": 0.85, "sc": 0.05, "pseudo_sc": 0.10}
SOURCE_WEIGHTS = {
    "focal":         1.0,
    "focal_missing": 0.0,
    "sc":            1.0,
    "pseudo_sc":     0.5,   # noisier than hard labels
}

# --- pseudo CSV path (NB1 v2 output) ---
PSEUDO_CSV_PATH    = Path("/content/data/pseudo-r1-v2/pseudo_labels.csv")
TRAIN_SC_DIR       = Path("/content/data/train_soundscapes")""")

cells.append(code_cell(c4_src, "config"))

# =============================================================================
# Cell 5: Load data — Tucker original + load pseudo CSV
# =============================================================================
load_data_src = cell_src(ref_nb["cells"][7])

# Append pseudo CSV loading at end
pseudo_load = r"""

# =================================================================
# v2 ADDITION — Load pseudo CSV (soft labels on train_soundscapes)
# =================================================================
print(f"\nLoading pseudo CSV: {PSEUDO_CSV_PATH}")
t0 = time.time()
pseudo_df_full = pd.read_csv(PSEUDO_CSV_PATH)
print(f"  loaded {len(pseudo_df_full)} rows in {time.time()-t0:.1f}s")
print(f"  columns: {list(pseudo_df_full.columns[:5])} ... ({pseudo_df_full.shape[1]} total)")

# Identify label columns. Expected schema: filename, start_sec, end_sec, <234 species cols in PRIMARY_LABELS order>
_meta_cols = [c for c in pseudo_df_full.columns if c not in PRIMARY_LABELS]
_label_cols_in_csv = [c for c in pseudo_df_full.columns if c in PRIMARY_LABELS]
print(f"  meta cols: {_meta_cols}")
print(f"  matching label cols: {len(_label_cols_in_csv)} / {NUM_CLASSES}")
assert len(_label_cols_in_csv) == NUM_CLASSES, \
    f"pseudo CSV must contain all {NUM_CLASSES} species cols; got {len(_label_cols_in_csv)}"

# Reorder/select label columns to match PRIMARY_LABELS order strictly
Y_pseudo = pseudo_df_full[PRIMARY_LABELS].values.astype(np.float32)
print(f"  Y_pseudo: {Y_pseudo.shape}, range=[{Y_pseudo.min():.4f}, {Y_pseudo.max():.4f}], mean={Y_pseudo.mean():.4f}")

# Keep meta-only DataFrame for indexing
pseudo_meta = pseudo_df_full[_meta_cols].copy().reset_index(drop=True)
# Ensure required columns exist
for col in ["filename", "start_sec"]:
    assert col in pseudo_meta.columns, f"pseudo CSV missing required col '{col}'"
# Normalize filename column (ensure .ogg suffix and the file exists check at __getitem__ time)
pseudo_meta["filename"] = pseudo_meta["filename"].astype(str)
pseudo_meta["start_sec"] = pseudo_meta["start_sec"].astype(float)

# Quick sanity: first 3 rows
print(f"  pseudo_meta head:\n{pseudo_meta.head(3)}")
print(f"\nPseudo SC: {len(pseudo_meta)} rows on train_soundscapes")
"""
load_data_src = load_data_src + pseudo_load

cells.append(code_cell(load_data_src, "load_data"))

# =============================================================================
# Cell 6: Model defs (Tucker verbatim)
# =============================================================================
cells.append(code_cell(cell_src(ref_nb["cells"][10]), "model_defs"))

# =============================================================================
# Cell 7: Data pipeline (Tucker original) + new PseudoScDS
# =============================================================================
dp_src = cell_src(ref_nb["cells"][12])

pseudo_ds = r'''

# =================================================================
# v2 ADDITION -- PseudoScDS: read .ogg directly with soundfile
# =================================================================
import soundfile as sf

class PseudoScDS(Dataset):
    """Pseudo-labeled soundscape windows. Reads .ogg directly with soundfile,
    seeking to start_sec for efficiency.

    Returns (waveform, soft_label, weight, mask, source_tag).
    soft_label is float32 in [0, 1] (post NB1 gamma correction).
    """
    def __init__(self, meta_df, Y_soft, audio_dir, aug=False):
        self.meta = meta_df.reset_index(drop=True)
        self.Y = Y_soft.astype(np.float32)
        self.audio_dir = Path(audio_dir)
        self.aug = aug
        self.filenames = self.meta["filename"].values
        self.start_secs = self.meta["start_sec"].values

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        fn = self.filenames[i]
        if not fn.endswith(".ogg"):
            fn = fn + ".ogg"
        path = self.audio_dir / fn
        start_sec = float(self.start_secs[i])

        try:
            wav, sr = sf.read(
                str(path),
                start=int(start_sec * SR),
                frames=TRAIN_SAMPLES,
                dtype="float32",
                always_2d=False,
            )
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            if len(wav) < TRAIN_SAMPLES:
                wav = np.pad(wav, (0, TRAIN_SAMPLES - len(wav)))
            else:
                wav = wav[:TRAIN_SAMPLES]
        except Exception:
            wav = np.zeros(TRAIN_SAMPLES, dtype=np.float32)

        if self.aug:
            wav = apply_aug(wav)

        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # (1, T)
        soft_lb = torch.from_numpy(self.Y[i])
        return (wav_t, soft_lb,
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "pseudo_sc")


print("OK PseudoScDS defined (reads .ogg via soundfile.read with seek)")
'''
dp_src = dp_src + pseudo_ds

cells.append(code_cell(dp_src, "data_pipeline"))

# =============================================================================
# Cell 8: Build active datasets (3 sources) — override Tucker's build_active_datasets
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: build_active_datasets v2 — adds pseudo_sc as 3rd source
# ============================================================

def build_active_datasets(fold_k):
    items = []
    if USE_FOCAL:
        fds = FocalDS(audio_cache_meta[audio_cache_meta["fold"] != fold_k],
                      LABEL2IDX, secondary_lookup=focal_secondary_labels,
                      sc_mixup_sources=sc_mixup_sources if USE_FOCAL_SC_MIXUP else None,
                      fold_k=fold_k, aug=True)
        items.append(("focal", fds, len(fds)))
    if USE_LABELED_SC:
        vm = sc_cache_meta["fold"].values == fold_k
        sc_train_df = sc_cache_meta[~vm].reset_index(drop=True)
        Y_tr = Y_SC[~vm]
        sds = ScDS(Y_tr, sc_train_df, aug=True)
        items.append(("sc", sds, len(sds)))
    if USE_PSEUDO_SC:
        # Pseudo SC: not fold-filtered (unlabeled, no leakage concern)
        pds = PseudoScDS(pseudo_meta, Y_pseudo, TRAIN_SC_DIR, aug=True)
        items.append(("pseudo_sc", pds, len(pds)))
    return items

print("OK build_active_datasets v2 ready (focal + sc + pseudo_sc)")
""", "build_active"))

# =============================================================================
# Cell 9: ZIP streaming patch (same as exp012)
# =============================================================================
cells.append(code_cell(r'''# ============================================================
# Cell 9: .ogg patch — replace load_focal/load_sc to read .ogg directly from /content/data
# ============================================================
# v2 (Drive-based): Tucker .pt cache 廃止、cache_file → 元 .ogg をマッピング
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from pathlib import Path

TRAIN_AUDIO_DIR = Path("/content/data/train_audio")

# Build lookup: cache_file (Tucker .pt name) → original .ogg filename
print("Building cache_file → .ogg lookup ...")
_CACHE_FILE_TO_OGG_FOCAL = {row["cache_file"]: row["filename"] for _, row in audio_cache_meta.iterrows()}
_CACHE_FILE_TO_OGG_SC = {row["cache_file"]: row["filename"] for _, row in sc_cache_meta.iterrows()}
print(f"  focal lookup: {len(_CACHE_FILE_TO_OGG_FOCAL)} entries")
print(f"  sc lookup:    {len(_CACHE_FILE_TO_OGG_SC)} entries")

# Patch load_focal: read .ogg from train_audio
def load_focal(p):
    # Read .ogg from /content/data/train_audio. p is cache_file (audio/audio_xxx.pt)
    if p in _FC: return _FC[p]
    ogg_rel = _CACHE_FILE_TO_OGG_FOCAL.get(p)
    if ogg_rel is None:
        return None
    ogg_path = TRAIN_AUDIO_DIR / ogg_rel
    if not ogg_path.exists():
        return None
    try:
        wav, sr = sf.read(str(ogg_path), dtype="int16")
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype("int16")
        if sr != SR:
            import librosa
            wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=SR)
            wav = wav.astype("float32") / 32767.0   # already float
            a = wav
        else:
            a = wav.astype("float32") / 32767.0   # int16 → [-1, 1]
    except Exception:
        return None
    if len(_FC) >= 2000:
        _FC.pop(next(iter(_FC)))
    _FC[p] = a
    return a

# Patch load_sc_waveform_from: read .ogg from train_soundscapes
TRAIN_SC_DIR_LOCAL = Path("/content/data/train_soundscapes")
def load_sc_waveform_from(cache_dir, cache_file):
    # Read .ogg from /content/data/train_soundscapes. cache_dir ignored (kept for API compat)
    key = str(cache_dir) + "::" + str(cache_file)
    if key in _SC_CACHE: return _SC_CACHE[key]
    ogg_rel = _CACHE_FILE_TO_OGG_SC.get(cache_file)
    if ogg_rel is None:
        return None
    ogg_path = TRAIN_SC_DIR_LOCAL / ogg_rel
    if not ogg_path.exists():
        return None
    try:
        wav, sr = sf.read(str(ogg_path), dtype="int16")
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype("int16")
        if sr != SR:
            import librosa
            wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=SR)
            a = wav.astype("float32")
        else:
            a = wav.astype("float32") / 32767.0
    except Exception:
        return None
    if len(_SC_CACHE) >= 200:
        _SC_CACHE.pop(next(iter(_SC_CACHE)))
    _SC_CACHE[key] = a
    return a

# Sanity check
print("\nSanity check load_focal:")
cf0 = audio_cache_meta.iloc[0]["cache_file"]
sample = load_focal(cf0)
if sample is not None:
    print(f"  {cf0} -> .ogg={_CACHE_FILE_TO_OGG_FOCAL[cf0]}, "
          f"shape={sample.shape}, min={sample.min():.4f}, max={sample.max():.4f}")
else:
    print(f"  FAILED to load {cf0}")

print("\nSanity check load_sc_waveform_from:")
cf_sc = sc_cache_meta.iloc[0]["cache_file"]
sample_sc = load_sc_waveform_from(None, cf_sc)
if sample_sc is not None:
    print(f"  {cf_sc} -> .ogg={_CACHE_FILE_TO_OGG_SC[cf_sc]}, "
          f"shape={sample_sc.shape}, min={sample_sc.min():.4f}, max={sample_sc.max():.4f}")
else:
    print(f"  FAILED to load {cf_sc}")

# Clear caches (in case prior loader populated)
_FC.clear()
_SC_CACHE.clear()
print("\nload_focal / load_sc_waveform_from patched -> reads .ogg directly")
''', "ogg_patch"))

# =============================================================================
# Cell 10: Training function (Tucker verbatim — already source-weight aware)
# =============================================================================
cells.append(code_cell(cell_src(ref_nb["cells"][14]), "train_fn"))

# =============================================================================
# Cell 11: Drive mirror patch around train_fold (resumability)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 11: Drive mirror — auto-mirror checkpoints after each fold
# ============================================================
import shutil
from pathlib import Path

DRIVE_OUT_DIR = OUT_DIR_DRIVE
_orig_train_fold = train_fold

def train_fold(fold_k):
    print(f"=== train_fold({fold_k}) ===")
    # Pre-load any existing ckpts from Drive (resume support)
    for fname in [f"r1v2_fold{fold_k}_best_ns22.pth",
                  f"r1v2_fold{fold_k}_best_macro.pth",
                  f"r1v2_fold{fold_k}_history.json"]:
        d = DRIVE_OUT_DIR / fname
        l = OUT_DIR / fname
        if d.exists() and not l.exists():
            shutil.copy(str(d), str(l))
            print(f"  pre-loaded {fname} from Drive")

    result = _orig_train_fold(fold_k)

    # Mirror new outputs back to Drive
    for src in OUT_DIR.glob(f"r1v2_fold{fold_k}_*"):
        dst = DRIVE_OUT_DIR / src.name
        shutil.copy2(str(src), str(dst))
        print(f"  mirrored {src.name} -> Drive ({dst.stat().st_size/1e6:.1f}MB)")
    return result

print("OK Drive-mirroring train_fold ready")
""", "drive_mirror"))

# =============================================================================
# Cell 12: Fold loop — train each fold + save with r1v2_ prefix
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 12: Fold loop — train each fold, save ckpt + history
# ============================================================
import json as _json

if MODE != "train":
    print("Skipping training (MODE='infer')")
    oof_ns22 = None
    all_hist = {}
else:
    oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
    all_hist = {}
    for fold_k in FOLDS:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_k}  (R1 v2)")
        print(f"{'='*60}")
        torch.manual_seed(SEED + fold_k)
        np.random.seed(SEED + fold_k)
        random.seed(SEED + fold_k)

        vm = sc_cache_meta["fold"].values == fold_k
        val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)

        best_ns22_state, best_macro_state, hist = train_fold(fold_k)
        all_hist[fold_k] = hist

        # Save PyTorch checkpoints (.pth)
        if best_ns22_state is not None:
            torch.save(best_ns22_state, OUT_DIR / f"r1v2_fold{fold_k}_best_ns22.pth")
            print(f"  saved best_ns22 (auc={max(hist['ns22_macro']):.4f})")
        if best_macro_state is not None:
            torch.save(best_macro_state, OUT_DIR / f"r1v2_fold{fold_k}_best_macro.pth")
            print(f"  saved best_macro (auc={max(hist['macro']):.4f})")

        # Save history (without val_preds to keep size small)
        hist_serializable = {k: (v if k != "val_preds" else f"<{len(v)} arrays omitted>")
                             for k, v in hist.items()}
        with open(OUT_DIR / f"r1v2_fold{fold_k}_history.json", "w") as f:
            _json.dump(hist_serializable, f, indent=2, default=str)

        # OOF blend prediction with best_macro
        if best_macro_state is not None:
            mel_tf = MelSpecTransform().to(device)
            val_wavs_k = _load_val_waveforms(val_sc_df_k)
            m_oof = make_model()
            m_oof.load_state_dict(best_macro_state, strict=False)
            oof_ns22[vm] = _predict_from_waveforms(m_oof, mel_tf, val_wavs_k)["blend"]
            del m_oof
            torch.cuda.empty_cache()

print("\nFold loop done")
""", "fold_loop"))

# =============================================================================
# Cell 13: ONNX export
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 13: ONNX export — clip + framewise outputs (no distill head)
# ============================================================
import onnxruntime as ort

if MODE != "train":
    print("Skipping ONNX export (MODE='infer')")
else:
    INF_N_MELS = N_MELS  # full mel; Tucker used 128 in original; we keep N_MELS for accuracy
    INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1

    class SEDExportWrapper(nn.Module):
        # Inference-only wrapper: takes mel, returns (clip_logits, framewise_logits)
        # with framewise transposed to (B, T, num_classes) for downstream blend NB.
        def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, in_chans=1,
                num_classes=0, global_pool="", drop_path_rate=0.1,
            )
            self.gem_freq = GeMFreqPool(p_init=3.0)
            self.dense_drop1 = nn.Dropout(0.25)
            self.dense_conv = nn.Conv1d(backbone_dim, hidden_dim, 1)
            self.dense_relu = nn.ReLU(inplace=True)
            self.dense_drop2 = nn.Dropout(0.5)
            self.att = nn.Conv1d(hidden_dim, num_classes, 1)
            self.cla = nn.Conv1d(hidden_dim, num_classes, 1)

        def forward(self, mel):
            h = self.backbone(mel)
            h = self.gem_freq(h)
            h = self.dense_drop1(h)
            h = self.dense_conv(h)
            h = self.dense_relu(h)
            h = self.dense_drop2(h)
            norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
            framewise = self.cla(h)
            clip = torch.sum(norm_att * framewise, dim=2)
            return clip, framewise.permute(0, 2, 1)

    def load_and_remap_state(export_model, trained_state):
        remap = {}
        for k, v in trained_state.items():
            if k.startswith("distill_head."):
                continue
            if k == "dense.1.weight":
                remap["dense_conv.weight"] = v.unsqueeze(-1)
            elif k == "dense.1.bias":
                remap["dense_conv.bias"] = v
            else:
                remap[k] = v
        export_model.load_state_dict(remap, strict=False)

    for fold_k in FOLDS:
        ckpt_path = OUT_DIR / f"r1v2_fold{fold_k}_best_macro.pth"
        if not ckpt_path.exists():
            print(f"  skip fold {fold_k}: ckpt missing")
            continue
        print(f"\nExporting fold {fold_k}...")
        state = torch.load(str(ckpt_path), map_location="cpu")

        # Determine backbone_dim from a reference model
        m_ref = make_model()
        backbone_dim = m_ref.backbone_dim
        del m_ref
        torch.cuda.empty_cache()

        export_model = SEDExportWrapper(BACKBONE_NAME, NUM_CLASSES, backbone_dim).to(device)
        load_and_remap_state(export_model, state)
        export_model.eval()

        dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)
        onnx_path = OUT_DIR / f"r1v2_fold{fold_k}.onnx"
        torch.onnx.export(
            export_model, dummy_mel, str(onnx_path),
            input_names=["mel"],
            output_names=["clip_logits", "framewise_logits"],
            dynamic_axes={"mel": {0: "batch"},
                          "clip_logits": {0: "batch"},
                          "framewise_logits": {0: "batch"}},
            opset_version=14,
        )

        # Verify
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onx_out = sess.run(None, {"mel": dummy_mel.cpu().numpy()})
        with torch.no_grad():
            ref_clip, _ = export_model(dummy_mel)
        diff = np.abs(ref_clip.cpu().numpy() - onx_out[0]).max()
        print(f"  ONNX verify: max|diff|={diff:.3e}")
        assert diff < 1e-3, f"ONNX export diverged: {diff}"
        del sess

        size_mb = onnx_path.stat().st_size / 1e6
        print(f"  exported {onnx_path.name} ({size_mb:.1f} MB)")
        del export_model
        torch.cuda.empty_cache()

    print("\nONNX export done")
""", "onnx_export"))

# =============================================================================
# Cell 14: Drive mirror final outputs (ckpt + ONNX)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 14: Mirror all final outputs to Drive
# ============================================================
import shutil

for src in OUT_DIR.glob("r1v2_*"):
    dst = DRIVE_OUT_DIR / src.name
    shutil.copy2(str(src), str(dst))
    print(f"  mirrored {src.name} -> Drive ({dst.stat().st_size/1e6:.1f}MB)")

print("\nFinal Drive contents:")
for p in sorted(DRIVE_OUT_DIR.iterdir()):
    print(f"  {p.name} ({p.stat().st_size/1e6:.1f}MB)")
""", "drive_final"))

# =============================================================================
# Cell 15: Optional Kaggle Dataset upload
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 15 (optional): Upload ONNX bundle to Kaggle Dataset
# ============================================================
import json as _json, shutil, tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

DATASET_USER  = "maekeso"
DATASET_SLUG  = "birdclef2026-exp013-r1-v2-student-sed"
DATASET_TITLE = "BirdCLEF2026 exp013 R1 v2 Student SED"

bundle = Path("/content/exp013_r1v2_onnx_bundle")
if bundle.exists(): shutil.rmtree(bundle)
bundle.mkdir(parents=True)

n_copied = 0
for src in DRIVE_OUT_DIR.glob("r1v2_fold*.onnx"):
    shutil.copy2(str(src), str(bundle / src.name))
    n_copied += 1
    # External data file (.onnx.data) for large models
    extdata = src.with_suffix(src.suffix + ".data")
    if extdata.exists():
        shutil.copy2(str(extdata), str(bundle / extdata.name))

if n_copied == 0:
    print("No ONNX files to upload; skipping")
else:
    (bundle / "dataset-metadata.json").write_text(_json.dumps({
        "title": DATASET_TITLE,
        "id":    f"{DATASET_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    print(f"Uploading {n_copied} ONNX files...")
    try:
        api.dataset_create_new(folder=str(bundle), dir_mode="skip",
                               public=False, quiet=False)
        print(f"Created dataset: {DATASET_USER}/{DATASET_SLUG}")
    except Exception as e:
        print(f"create_new failed: {str(e)[:200]}")
        try:
            api.dataset_create_version(folder=str(bundle),
                                        version_notes="exp013 R1 v2 student SED",
                                        dir_mode="skip", quiet=False)
            print(f"Updated existing version: {DATASET_USER}/{DATASET_SLUG}")
        except Exception as e2:
            print(f"update also failed: {str(e2)[:200]}")
            print("Manual upload may be required.")

    print(f"\nFor blend NB: dataset_sources: ['{DATASET_USER}/{DATASET_SLUG}']")
""", "upload_kg"))

cells.append(md_cell(r"""## 完了

`maekeso/birdclef2026-exp013-r1-v2-student-sed` を blend NB の `dataset_sources` に指定して提出。

### 検証チェックリスト
- [ ] Cell 5 で pseudo_meta + Y_pseudo が正しく load された
- [ ] Cell 9 zip-streaming で sample 読み込み成功
- [ ] Cell 12 fold 0 の `non_s22_macro` が 0.85 以上 (R1 v1 の 0.67 を上回ること)
- [ ] Cell 13 ONNX verify で max|diff| < 1e-3
- [ ] Drive にミラー完了後、blend NB に dataset 指定して LB submission
""", "footer"))

# =============================================================================
# Notebook assembly
# =============================================================================
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

out_path = HERE / "nb_pl2_sed_train_v2.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
