"""Generate exp017 R3 NB: eca_nfnet_l0 SED + R2 pseudo (Colab Pro Blackwell).

Round 3 of Multi-Iterative Noisy Student:
- R2 student (exp017 eca_nfnet_l0、val_ns22 0.9249) が生成した pseudo を 3rd source として使用
- R3 student も同 backbone (eca_nfnet_l0)、新規初期化 (Babych 流)
- NFNet 系は sqrt LR scaling 厳禁 ([[feedback_nfnet_lr_no_sqrt_scaling]]): batch=192 でも LR=3e-4 据え置き
- 学習 → pseudo R3 生成 → 重み Kaggle Dataset upload まで 1 ipynb で完結

データソース (R2 と同じレシピ、pseudo input のみ R2 由来):
- focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20 (Babych BC25 1位レシピ準拠)
- pseudo_sc は source_weight 0.5 (hard label より noisy)

入出力:
- 入力: Drive `output/exp017/r2-pseudo/pseudo_labels.csv` (R2 pseudo)
        Drive train_audio / train_soundscapes (Kaggle API 直 DL 経由)
- 出力 ckpt: Drive `output/exp017/r3/`
- 出力 pseudo: Drive `output/exp017/r3-pseudo/pseudo_labels.csv`
- Kaggle Dataset: `maekeso/birdclef2026-exp017-weights` (version up、R3 重み追加 r3_*)

Run: python _gen_nb_train_r3.py  ->  writes nb_train_r3.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


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


cells = []

# =============================================================================
# Header
# =============================================================================
cells.append(md_cell(r"""# exp017 R3 — eca_nfnet_l0 SED + R2 pseudo (Colab Pro Blackwell)

**Round 3 of Multi-Iterative Noisy Student** (Babych BC2025 1位レシピ準拠)。

## What's new in R3
- R2 student (val_ns22 **0.9249**) が生成した pseudo CSV を 3rd training source として使用
- Source mix: **focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20** (R2 と同じ)
- pseudo_sc は source_weight 0.5 (hard label より noisy)
- R3 student は新規初期化 (R2 の重みは流用しない、Babych 流)
- Backbone は R1/R2 と同じ eca_nfnet_l0 (~24M)
- **NFNet rule**: batch=192 でも LR=**3e-4** 据え置き ([[feedback_nfnet_lr_no_sqrt_scaling]])
- N_TOTAL_EPOCHS=20 ([[feedback_default_epochs_20]]、best ckpt 自動保存で peak 取り損ねなし)

## 1 ipynb 統合構成 ([[feedback_train_pseudo_same_ipynb]])
1-13: train phase (R3 学習)
14-17: pseudo phase (R3 pseudo CSV 生成、R4 候補)
18:   upload phase (重み Kaggle Dataset 化、r3_* prefix)
19:   runtime.unassign() (unit 節約自動切断)

## Drive 構成
- 入力 ckpt source: 不要 (R3 は新規初期化)
- 入力 pseudo: `/content/drive/MyDrive/kaggle/birdclef2026/output/exp017/r2-pseudo/pseudo_labels.csv`
- 出力 R3 ckpt: `output/exp017/r3/`
- 出力 R3 pseudo: `output/exp017/r3-pseudo/pseudo_labels.csv`

## 想定時間 (Colab Pro Blackwell)
- 学習 20 epoch: ~3-3.5h (R2 と同じ、pseudo_sc 含む)
- pseudo R3 生成: 15-20 min
- 重み upload: 1-2 min
- **合計: ~3.5-4h**

## 期待 (diminishing returns)
- val_ns22 0.928-0.935 (R2 0.9249 から +0.003-0.010)
- LB 推定 +0.003-0.008 over R2 (Babych 経験則、R3 で頭打ち傾向)
- R2 pseudo の bias を増幅するリスクあり、best_ckpt 自動保存で peak 取得が重要
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — Drive mount, pip install, kaggle.json
# ============================================================
!pip install -q timm onnxruntime-gpu librosa soundfile scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

DRIVE_INPUT_DIR  = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_R2_PSEUDO  = DRIVE_INPUT_DIR / "output" / "exp017" / "r2-pseudo" / "pseudo_labels.csv"
DRIVE_OUTPUT_DIR = DRIVE_INPUT_DIR / "output" / "exp017" / "r3"
DRIVE_R3_PSEUDO_DIR = DRIVE_INPUT_DIR / "output" / "exp017" / "r3-pseudo"
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_R3_PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
assert DRIVE_INPUT_DIR.exists(), f"Drive input folder missing: {DRIVE_INPUT_DIR}"
assert DRIVE_R2_PSEUDO.exists(), f"R2 pseudo CSV missing: {DRIVE_R2_PSEUDO}"
print(f"Drive input:    {DRIVE_INPUT_DIR}")
print(f"R2 pseudo CSV:  {DRIVE_R2_PSEUDO} ({DRIVE_R2_PSEUDO.stat().st_size/1e6:.1f}MB)")
print(f"Drive R3 ckpt:  {DRIVE_OUTPUT_DIR}")
print(f"Drive R3 pseudo:{DRIVE_R3_PSEUDO_DIR}")

# kaggle.json
KJ_CANDIDATES = [
    DRIVE_INPUT_DIR / "kaggle.json",
    Path("/content/drive/MyDrive/kaggle.json"),
]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
if KJ is not None:
    KAGGLE_CFG = Path.home() / ".kaggle"
    KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
    os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
    creds = json.loads(KJ.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]
    print(f"kaggle.json: {KJ}")
else:
    print("kaggle.json not found")

LOCAL_DATA = Path("/content/data")
LOCAL_OUT  = Path("/content/output")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
print(f"Local data: {LOCAL_DATA}")
print(f"Local out:  {LOCAL_OUT}")
""", "setup"))

# =============================================================================
# Cell 2: Data prep — Kaggle API direct DL + R2 pseudo copy
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Data prep — Kaggle API direct DL + R2 pseudo copy (with tqdm)
# ============================================================
import time, zipfile, subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm.auto import tqdm

api = KaggleApi(); api.authenticate()
print("kaggle authenticated")

T0_total = time.time()

TA_DIR = LOCAL_DATA / "train_audio"
TS_DIR = LOCAL_DATA / "train_soundscapes"

need_dl = (
    not TA_DIR.exists() or sum(1 for _ in TA_DIR.rglob("*.ogg")) < 40000 or
    not TS_DIR.exists() or sum(1 for _ in TS_DIR.glob("*.ogg")) < 10000
)

if need_dl:
    print(f"\n[1/2] Downloading birdclef-2026 (~25GB) via Kaggle API...")
    t0 = time.time()
    api.competition_download_files(
        "birdclef-2026",
        path=str(LOCAL_DATA),
        force=False, quiet=False,
    )
    print(f"  DL done in {(time.time()-t0)/60:.1f} min")
    zips = list(LOCAL_DATA.glob("birdclef-2026*.zip"))
    assert zips
    zip_path = zips[0]
    print(f"\n[2/2] Extracting {zip_path.name} ({zip_path.stat().st_size/1e9:.1f}GB)...")
    t_extract = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        total_bytes = sum(i.file_size for i in infos)
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="extract", smoothing=0.05, mininterval=1.0)
        for info in infos:
            zf.extract(info, LOCAL_DATA)
            pbar.update(info.file_size)
        pbar.close()
    print(f"  extracted in {(time.time()-t_extract)/60:.1f} min")
    zip_path.unlink()
else:
    print("Competition data already present locally")

n_ta = sum(1 for _ in TA_DIR.rglob("*.ogg")) if TA_DIR.exists() else 0
n_ts = sum(1 for _ in TS_DIR.glob("*.ogg")) if TS_DIR.exists() else 0
print(f"\n  train_audio: {n_ta}, train_soundscapes: {n_ts}")

# Competition CSVs
comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
    src_in_root = LOCAL_DATA / fn
    dst = comp / fn
    if src_in_root.exists() and not dst.exists():
        shutil.copy2(str(src_in_root), str(dst))

# Perch ONNX
po_dir = LOCAL_DATA / "perch-onnx"
po_dir.mkdir(parents=True, exist_ok=True)
PERCH_PATH = po_dir / "perch_v2_no_dft.onnx"
if not PERCH_PATH.exists():
    drive_perch = list(DRIVE_INPUT_DIR.rglob("perch_v2*.onnx"))
    if drive_perch:
        src = drive_perch[0]; sz = src.stat().st_size
        print(f"\nCopying Perch ONNX from Drive ({sz/1e9:.2f}GB)...")
        with open(src, "rb") as fin, open(PERCH_PATH, "wb") as fout:
            pbar = tqdm(total=sz, unit="B", unit_scale=True, unit_divisor=1024,
                        desc="Perch copy", mininterval=1.0)
            while True:
                buf = fin.read(8*1024*1024)
                if not buf: break
                fout.write(buf); pbar.update(len(buf))
            pbar.close()
    else:
        print("\nPerch ONNX not on Drive, DL from Kaggle...")
        api.dataset_download_files("tuckerarrants/perch-v2-no-dft-onnx",
                                    path=str(po_dir), unzip=True, quiet=False)
        if not PERCH_PATH.exists():
            hits = list(po_dir.rglob("perch_v2*.onnx"))
            if hits: shutil.move(str(hits[0]), str(PERCH_PATH))
assert PERCH_PATH.exists()
print(f"Perch ONNX: {PERCH_PATH.stat().st_size/1e9:.2f}GB")

# R2 pseudo CSV (Drive → local for fast access)
LOCAL_PSEUDO_DIR = LOCAL_DATA / "r2-pseudo"
LOCAL_PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_PSEUDO_CSV = LOCAL_PSEUDO_DIR / "pseudo_labels.csv"
if not LOCAL_PSEUDO_CSV.exists():
    sz = DRIVE_R2_PSEUDO.stat().st_size
    print(f"\nCopying R2 pseudo CSV from Drive ({sz/1e6:.1f}MB)...")
    with open(DRIVE_R2_PSEUDO, "rb") as fin, open(LOCAL_PSEUDO_CSV, "wb") as fout:
        pbar = tqdm(total=sz, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="pseudo copy", mininterval=1.0)
        while True:
            buf = fin.read(8*1024*1024)
            if not buf: break
            fout.write(buf); pbar.update(len(buf))
        pbar.close()
print(f"R2 pseudo CSV: {LOCAL_PSEUDO_CSV.stat().st_size/1e6:.1f}MB")

print(f"\n=== Total prep time: {(time.time()-T0_total)/60:.1f} min ===")
r = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
print(r.stdout)
""", "dl_data"))

# =============================================================================
# Cell 3: Imports + Config + Paths
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 3: Imports + Config + Paths
# ============================================================
import os, time, json, gc, random, math
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

# ===== Paths =====
BASE = LOCAL_DATA / "competition"
TA_DIR = LOCAL_DATA / "train_audio"
TS_DIR = LOCAL_DATA / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
LABELS_PATH = BASE / "train_soundscapes_labels.csv"
PERCH_PATH = LOCAL_DATA / "perch-onnx" / "perch_v2_no_dft.onnx"
PSEUDO_CSV_PATH = LOCAL_DATA / "r2-pseudo" / "pseudo_labels.csv"
OUT_DIR = LOCAL_OUT
print(f"BASE: {BASE}")
print(f"TA_DIR: {TA_DIR.exists()}, TS_DIR: {TS_DIR.exists()}")
print(f"PERCH: {PERCH_PATH.exists()}, PSEUDO: {PSEUDO_CSV_PATH.exists()}")

# ===== Reproducibility =====
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ===== Config =====
NUM_CLASSES = 234
SR = 32000
TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 256
FMIN           = 20
FMAX           = 16000

BACKBONE = "eca_nfnet_l0"

USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

N_FOLDS = 5
FOLDS = [0]
FORCE_FRESH = True       # ★ R3 は新規初期化 (Babych 流)、Drive 上に既存 R3 ckpt があっても無視
# ★ R3 on Blackwell: VRAM 96GB で batch=192、ただし NFNet rule で LR は 3e-4 据え置き
# [[feedback_nfnet_lr_no_sqrt_scaling]]: NFNet は AGC 無し環境で sqrt LR scaling すると overshoot
N_TOTAL_EPOCHS = 20      # ★ [[feedback_default_epochs_20]]、best ckpt 自動保存で peak 取り損ねなし
BATCH = 192              # ★ Blackwell 96GB の VRAM 活用、R1 v2 と同じ
LR = 3e-4                # ★ NFNet rule: sqrt scaling 厳禁、batch 増しても 3e-4 据え置き
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 3

AUG_PROB = 0.5
AUG_GAIN_DB_RANGE = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

USE_MIXUP = True
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
MIXUP_HARD = False

FREQ_MASK_PARAM = 25
TIME_MASK_PARAM = 30
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

MIN_SAMPLE = 20

# ★ R3: 3-source mix (R2 と同じ Babych BC25 1位 recipe)
SHARES = {"focal": 0.70, "labeled_sc": 0.10, "pseudo_sc": 0.20}
SOURCE_WEIGHTS = {
    "focal":          1.0,
    "focal_missing":  0.0,
    "labeled_sc":     1.0,
    "pseudo_sc":      0.5,   # noisier than hard labels
}

NUM_WORKERS = 8           # ★ Blackwell + batch=128 はデータ供給律速、4→8 で +20-40%
PERSISTENT_WORKERS = True

TRAIN_START = time.time()
MAX_RUNTIME_SEC = 22.0 * 3600

print(f"Backbone: {BACKBONE}")
print(f"Batch: {BATCH} | Total epochs: {N_TOTAL_EPOCHS} | Folds: {FOLDS}")
print(f"LR: {LR} | WD: {WD} | warmup: {WARMUP_EPOCHS}ep")
print(f"Source mix: {SHARES}")
print(f"Source weights: {SOURCE_WEIGHTS}")
print(f"Max runtime: {MAX_RUNTIME_SEC/3600:.1f}h")
""", "config"))

# =============================================================================
# Cell 4: Load data (incl. R2 pseudo)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Load data — train.csv, taxonomy, labeled SS, R2 pseudo CSV
# ============================================================
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES

taxonomy = pd.read_csv(TAXO_PATH)
label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                          taxonomy["class_name"].astype(str)))
TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                            if label_to_taxon.get(l, "") == t])
               for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}

train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[train_df["primary_label"].astype(str).isin(LABEL2IDX)].reset_index(drop=True)
train_df["filename"] = train_df["filename"].astype(str)
print(f"Focal train.csv: {len(train_df)} rows")

def _check_exists(fn):
    return (TA_DIR / fn).exists()
print("Checking focal file existence...")
_t0 = time.time()
train_df["exists"] = train_df["filename"].map(_check_exists)
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"  {len(train_df)} focal files exist ({time.time()-_t0:.1f}s)")
train_df["original_idx"] = np.arange(len(train_df))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fold
print(f"Focal fold distribution: {train_df['fold'].value_counts().sort_index().to_dict()}")

focal_secondary_labels = {}
for idx, row in train_df.iterrows():
    sec = row.get("secondary_labels", "")
    if pd.isna(sec) or sec in ("", "[]"):
        continue
    try:
        sec_list = eval(sec) if isinstance(sec, str) else []
    except Exception:
        continue
    valid = [s for s in sec_list if s in LABEL2IDX]
    if valid:
        focal_secondary_labels[int(row["original_idx"])] = valid
print(f"Focal secondary labels: {len(focal_secondary_labels)} files")

counts = train_df["primary_label"].value_counts()
rare_species = counts[counts < MIN_SAMPLE].index.tolist()
extra_rows = []
for sp in rare_species:
    sp_rows = train_df[train_df["primary_label"] == sp]
    n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
    for _ in range(n_copies):
        extra_rows.append(sp_rows)
n_before = len(train_df)
if extra_rows:
    train_df = pd.concat([train_df] + extra_rows, ignore_index=True)
print(f"Upsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): {n_before} -> {len(train_df)}")

# Labeled SS
if LABELS_PATH.exists():
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    if sc_labels_raw["start"].dtype == object:
        sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    else:
        sc_labels_raw["start_sec"] = sc_labels_raw["start"].astype(int)

    sc_meta = (sc_labels_raw[["filename", "start_sec"]]
               .drop_duplicates()
               .reset_index(drop=True))
    if "site" in sc_labels_raw.columns:
        site_map = sc_labels_raw.groupby("filename")["site"].first().to_dict()
        sc_meta["site"] = sc_meta["filename"].map(site_map).fillna("UNK")
    else:
        sc_meta["site"] = "UNK"

    Y_SC = np.zeros((len(sc_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_meta.iterrows():
        matches = sc_labels_raw[(sc_labels_raw["filename"] == row["filename"]) &
                                 (sc_labels_raw["start_sec"] == row["start_sec"])]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    print(f"Soundscape labels: {len(sc_meta)} windows, {int(Y_SC.sum())} positives, "
          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")

    sc_files = sc_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_meta["fold"] = sc_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    non_s22_mask_sc = (sc_meta["site"].values != "S22")
    print(f"Non-S22: {non_s22_mask_sc.sum()}/{len(sc_meta)}")
else:
    print("LABELS_PATH missing")
    sc_meta = pd.DataFrame(columns=["filename", "start_sec", "site", "fold"])
    Y_SC = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    non_s22_mask_sc = np.zeros(0, dtype=bool)

# ★ R2 pseudo CSV (R3 の 3rd source)
print(f"\nLoading R2 pseudo CSV: {PSEUDO_CSV_PATH}")
_t0 = time.time()
pseudo_df_full = pd.read_csv(PSEUDO_CSV_PATH)
print(f"  loaded {len(pseudo_df_full)} rows in {time.time()-_t0:.1f}s")

_label_cols_in_csv = [c for c in pseudo_df_full.columns if c in LABEL2IDX]
assert len(_label_cols_in_csv) == NUM_CLASSES, (
    f"pseudo CSV must contain all {NUM_CLASSES} species cols; got {len(_label_cols_in_csv)}"
)
Y_PSEUDO = pseudo_df_full[PRIMARY_LABELS].values.astype(np.float32)
pseudo_meta = pseudo_df_full[["filename", "start_sec"]].copy().reset_index(drop=True)
pseudo_meta["filename"] = pseudo_meta["filename"].astype(str)
pseudo_meta["start_sec"] = pseudo_meta["start_sec"].astype(float)
print(f"  Y_pseudo: {Y_PSEUDO.shape}, range=[{Y_PSEUDO.min():.4f}, {Y_PSEUDO.max():.4f}], "
      f"mean={Y_PSEUDO.mean():.4f}")
print(f"  pseudo_meta: {len(pseudo_meta)} rows, {pseudo_meta['filename'].nunique()} files")

print("\nOK data loaded")
""", "load_data"))

# =============================================================================
# Cell 5: Model
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Model — Mel + SpecAugment + Perch teacher + DistillHead + BirdSEDModel
# ============================================================
import onnxruntime as ort

class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))

class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    def forward(self, mel):
        for _ in range(NUM_FREQ_MASKS): mel = self.freq_mask(mel)
        for _ in range(NUM_TIME_MASKS): mel = self.time_mask(mel)
        return mel

class PerchTeacher:
    def __init__(self, onnx_path, device_str="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device_str == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._embed_idx = None
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self._embed_idx = i; break
        if self._embed_idx is None:
            self._embed_idx = 1
        print(f"PerchTeacher: embed_idx={self._embed_idx}, providers={self.session.get_providers()}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()

class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))

class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)

class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
            print(f"Backbone out: {tuple(feat.shape)}  (C={self.backbone_dim})")

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.cla.weight)
        self.att.bias.data.fill_(0.)
        self.cla.bias.data.fill_(0.)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        elif return_framewise:
            return clip_logits, fw
        elif return_distill:
            return clip_logits, distill_emb
        return clip_logits

def make_model():
    m = BirdSEDModel().to(device)
    m = m.to(memory_format=torch.channels_last)
    return m

print("OK model defs ready")
""", "model"))

# =============================================================================
# Cell 6: Datasets (focal + labeled_sc + pseudo_sc)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Datasets — FocalDS, LabeledSCDS, PseudoScDS, samplers
# ============================================================
import soundfile as sf
import librosa
from functools import lru_cache

# LRU cache for full 60s SS waveforms (12 windows × 5s same file → cache hit)
# ★ workers=8 で並列度上げる時、cache 容量も増やす (per-worker、各 ~4GB RAM)
@lru_cache(maxsize=512)
def _load_full_audio_cached(path_str):
    try:
        wav, sr = sf.read(path_str, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None

def _load_ogg(path, n_samples_target, start_sec=None):
    if start_sec is not None:
        full = _load_full_audio_cached(str(path))
        if full is None:
            return None
        start = int(start_sec * SR)
        end = start + n_samples_target
        if end <= len(full):
            return full[start:end].copy()
        out = np.zeros(n_samples_target, dtype=np.float32)
        avail = full[start:start + n_samples_target]
        out[:len(avail)] = avail
        return out
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None

def extract_chunk_np(waveform, start_sample, n_samples):
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[start_sample:start_sample + n_samples]

def apply_aug(w):
    if np.random.random() < AUG_PROB:
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))
    return w


class FocalDS(Dataset):
    def __init__(self, df, l2i, secondary_lookup=None, aug=False):
        self.df = df.reset_index(drop=True)
        self.l2i = l2i
        self.aug = aug
        self.secondary_lookup = secondary_lookup
        self.filenames = self.df["filename"].values
        self.primary = self.df["primary_label"].astype(str).values
        self.original_idx = self.df["original_idx"].values if "original_idx" in self.df.columns else None

    def __len__(self): return len(self.df)

    def _load_chunk(self, i):
        fn = self.filenames[i]
        path = TA_DIR / fn
        wav = _load_ogg(path, n_samples_target=None)
        if wav is None:
            return None, None
        if self.aug and len(wav) > TRAIN_SAMPLES:
            start = np.random.randint(0, len(wav) - TRAIN_SAMPLES + 1)
        else:
            start = 0
        chunk = extract_chunk_np(wav, start, TRAIN_SAMPLES)
        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        if self.primary[i] in self.l2i:
            lb[self.l2i[self.primary[i]]] = 1.0
        if self.secondary_lookup is not None and self.original_idx is not None:
            for s in self.secondary_lookup.get(int(self.original_idx[i]), []):
                if s in self.l2i: lb[self.l2i[s]] = 1.0
        return chunk, lb

    def __getitem__(self, i):
        ch1, lb1 = self._load_chunk(i)
        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal_missing")

        if USE_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            ch2, lb2 = None, None
            for _ in range(3):
                j = np.random.randint(len(self.df))
                ch2, lb2 = self._load_chunk(j)
                if ch2 is not None: break
            if ch2 is not None:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                if self.aug: ch_mix = apply_aug(ch_mix)
                lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                return (torch.from_numpy(ch_mix).unsqueeze(0),
                        torch.from_numpy(lb.astype(np.float32)),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")

        if self.aug: ch1 = apply_aug(ch1)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")


class LabeledSCDS(Dataset):
    def __init__(self, Y, sc_df, aug=False):
        self.Y = Y
        self.df = sc_df.reset_index(drop=True)
        self.aug = aug
        self.filenames = self.df["filename"].astype(str).values
        self.start_secs = self.df["start_sec"].astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        fn = self.filenames[i]
        if not fn.endswith(".ogg"): fn = fn + ".ogg"
        path = TS_DIR / fn
        wav = _load_ogg(path, n_samples_target=TRAIN_SAMPLES,
                         start_sec=float(self.start_secs[i]))
        if wav is None:
            wav = np.zeros(TRAIN_SAMPLES, dtype=np.float32)
        if len(wav) < TRAIN_SAMPLES:
            wav = np.pad(wav, (0, TRAIN_SAMPLES - len(wav)))
        else:
            wav = wav[:TRAIN_SAMPLES]
        if self.aug:
            wav = apply_aug(wav)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "labeled_sc")


class PseudoScDS(Dataset):
    '''Pseudo-labeled SS windows from prior round. Soft labels.'''
    def __init__(self, meta_df, Y_soft, audio_dir, aug=False):
        self.meta = meta_df.reset_index(drop=True)
        self.Y = Y_soft.astype(np.float32)
        self.audio_dir = Path(audio_dir)
        self.aug = aug
        self.filenames = self.meta["filename"].values
        self.start_secs = self.meta["start_sec"].values

    def __len__(self): return len(self.meta)

    def __getitem__(self, i):
        fn = str(self.filenames[i])
        if not fn.endswith(".ogg"):
            fn = fn + ".ogg"
        path = self.audio_dir / fn
        start_sec = float(self.start_secs[i])
        wav = _load_ogg(path, n_samples_target=TRAIN_SAMPLES, start_sec=start_sec)
        if wav is None:
            wav = np.zeros(TRAIN_SAMPLES, dtype=np.float32)
        if len(wav) < TRAIN_SAMPLES:
            wav = np.pad(wav, (0, TRAIN_SAMPLES - len(wav)))
        else:
            wav = wav[:TRAIN_SAMPLES]
        if self.aug:
            wav = apply_aug(wav)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i]),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "pseudo_sc")


class MixSamp(torch.utils.data.Sampler):
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.rng = np.random.default_rng(seed)
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs:
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
    def __len__(self): return self.nst
    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0: continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch

def collate_m(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            [b[4] for b in batch])

def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)

print("OK datasets ready (focal + labeled_sc + pseudo_sc)")
""", "dataset"))

# =============================================================================
# Cell 7: Eval helpers
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 7: Eval — AUC + validation predictor
# ============================================================
def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (np.mean(aucs) if aucs else float("nan")), len(aucs)


def full_eval(y_true, y_pred, ns22_mask, taxon_masks):
    r = {}
    a, n = compute_macro_auc(y_true, y_pred)
    r["macro_auc_all"] = round(float(a) if not np.isnan(a) else 0.0, 4)
    r["n_all"] = n
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22_mask)
    r["non_s22_macro"] = round(float(a) if not np.isnan(a) else 0.0, 4)
    r["n_ns22"] = n
    per_taxon = {}
    for t, cm in taxon_masks.items():
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22_mask, class_mask=cm)
        per_taxon[t] = round(float(a) if not np.isnan(a) else 0.0, 4)
    r["per_taxon"] = per_taxon
    return r


def _load_val_waveforms(val_sc_df):
    wavs = []
    for _, row in val_sc_df.iterrows():
        fn = str(row["filename"])
        if not fn.endswith(".ogg"): fn = fn + ".ogg"
        wav = _load_ogg(TS_DIR / fn, n_samples_target=VAL_SAMPLES,
                         start_sec=float(row["start_sec"]))
        if wav is None or len(wav) == 0:
            wav = np.zeros(VAL_SAMPLES, dtype=np.float32)
        if len(wav) < VAL_SAMPLES:
            wav = np.pad(wav, (0, VAL_SAMPLES - len(wav)))
        else:
            wav = wav[:VAL_SAMPLES]
        wavs.append(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0))
    return wavs


def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    with torch.no_grad():
        for s in range(0, len(wav_list), batch_size):
            batch = torch.stack(wav_list[s:s+batch_size]).to(device)
            mel = mel_transform(batch)
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            mel = mel.to(memory_format=torch.channels_last)
            with autocast():
                clip_logits, framewise = model(mel, return_framewise=True)
                frame_max = framewise.max(dim=1).values
                p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()
                p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
                p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip); preds_fmax.append(p_fmax); preds_blend.append(p_blend)
    return {"clip": np.concatenate(preds_clip),
            "fmax": np.concatenate(preds_fmax),
            "blend": np.concatenate(preds_blend)}

print("OK eval helpers ready")
""", "eval"))

# =============================================================================
# Cell 8: Build active datasets (3-source)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: build_active_datasets (R3: 3 sources)
# ============================================================
def build_active_datasets(fold_k):
    items = []
    fds = FocalDS(train_df[train_df["fold"] != fold_k],
                  LABEL2IDX, secondary_lookup=focal_secondary_labels, aug=True)
    items.append(("focal", fds, len(fds)))
    if len(sc_meta) > 0:
        vm = sc_meta["fold"].values == fold_k
        sc_train_df = sc_meta[~vm].reset_index(drop=True)
        Y_tr = Y_SC[~vm]
        sds = LabeledSCDS(Y_tr, sc_train_df, aug=True)
        items.append(("labeled_sc", sds, len(sds)))
    # Pseudo SC (not fold-filtered — unlabeled-derived, no leakage)
    pds = PseudoScDS(pseudo_meta, Y_PSEUDO, TS_DIR, aug=True)
    items.append(("pseudo_sc", pds, len(pds)))
    return items

print("OK build_active_datasets ready (focal + labeled_sc + pseudo_sc)")
""", "build_active"))

# =============================================================================
# Cell 9: init / resume from Drive
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 9: init — fresh R3 start, optionally resume from Drive
# ============================================================
FOLD_K = FOLDS[0]
print(f"This run trains fold {FOLD_K}")

active = build_active_datasets(FOLD_K)
NAMES, DATASETS, SIZES = zip(*active)
NAMES, DATASETS, SIZES = list(NAMES), list(DATASETS), list(SIZES)
print(f"Streams: {dict(zip(NAMES, SIZES))}")

mds = ConcatDataset(DATASETS)
N_STEPS_PER_EP = max(100, int(sum(SIZES) / BATCH))
print(f"steps/epoch: {N_STEPS_PER_EP}")

model = make_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scaler = GradScaler()

warmup_steps = N_STEPS_PER_EP * WARMUP_EPOCHS
total_steps  = N_STEPS_PER_EP * N_TOTAL_EPOCHS
warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=MIN_LR)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

start_epoch = 0
best_ns22 = -1.0
best_macro = -1.0
history = []
resumed = False

# Optional resume from Drive R3
DRIVE_LATEST = DRIVE_OUTPUT_DIR / "ckpt_latest.pth"
if FORCE_FRESH and DRIVE_LATEST.exists():
    print(f"\nFORCE_FRESH=True: ignoring existing R3 ckpt {DRIVE_LATEST}, starting fresh")
elif DRIVE_LATEST.exists():
    print(f"\nFound Drive R3 ckpt: {DRIVE_LATEST}")
    try:
        state = torch.load(str(DRIVE_LATEST), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(DRIVE_LATEST), map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    start_epoch = int(state["epoch"])
    best_ns22 = float(state.get("best_ns22", -1.0))
    best_macro = float(state.get("best_macro", -1.0))
    history = state.get("history", [])
    resumed = True
    print(f"=== RESUMED from epoch {start_epoch}/{N_TOTAL_EPOCHS} ===")
    print(f"    best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}, history_len={len(history)}")
    src = DRIVE_OUTPUT_DIR / "history.json"
    if src.exists():
        shutil.copy(str(src), str(OUT_DIR / "history.json"))
else:
    print("\nNo Drive R3 ckpt; starting fresh R3")

if start_epoch >= N_TOTAL_EPOCHS:
    print(f"\nAlready trained to epoch {start_epoch}/{N_TOTAL_EPOCHS}.")
""", "resume_or_init"))

# =============================================================================
# Cell 10: Val prep
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 10: Val prep — load val waveforms once
# ============================================================
if len(sc_meta) > 0:
    vm = sc_meta["fold"].values == FOLD_K
    val_sc_df = sc_meta[vm].reset_index(drop=True)
    Y_val = Y_SC[vm]
    ns22_val = non_s22_mask_sc[vm]
    print(f"Val: {len(val_sc_df)} SS windows for fold {FOLD_K}")
    print(f"  loading val waveforms...")
    _t0 = time.time()
    val_wavs = _load_val_waveforms(val_sc_df)
    print(f"  loaded in {time.time()-_t0:.1f}s")
else:
    val_sc_df = pd.DataFrame()
    Y_val = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    ns22_val = np.zeros(0, dtype=bool)
    val_wavs = []
    print("No labeled SS for validation")
""", "val_prep"))

# =============================================================================
# Cell 11: Train loop with Drive mirror per epoch
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 11: Training loop — Drive mirror per epoch
# ============================================================
mel_transform = MelSpecTransform().to(device)
spec_augment = SpecAugment().to(device)
perch_teacher = PerchTeacher(PERCH_PATH, "cuda" if torch.cuda.is_available() else "cpu") \
                if USE_PERCH_DISTILL else None

def mirror_to_drive(local_path: Path, dst_name: str = None):
    try:
        name = dst_name or local_path.name
        shutil.copy2(str(local_path), str(DRIVE_OUTPUT_DIR / name))
    except Exception as e:
        print(f"  [WARN] mirror_to_drive failed for {local_path.name}: {e}")

epoch_times = []
trained_this_session = 0
ep = start_epoch

for epoch in range(start_epoch, N_TOTAL_EPOCHS):
    elapsed = time.time() - TRAIN_START
    if epoch_times:
        est_next = max(epoch_times[-3:])
        if elapsed + est_next * 1.3 > MAX_RUNTIME_SEC:
            print(f"\n[stop] Time budget exhausted before epoch {epoch+1}/{N_TOTAL_EPOCHS}")
            break

    ep_start = time.time()
    model.train()

    smp = MixSamp(SIZES, NAMES, SHARES, BATCH, N_STEPS_PER_EP, seed=42 + epoch)
    train_loader = DataLoader(
        mds, batch_sampler=smp, collate_fn=collate_m,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        pin_memory=True,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,   # 2→4
    )

    el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
    for batch_idx, (wav, lb, wt, mk, sr) in enumerate(train_loader):
        wav, lb, wt, mk = wav.to(device, non_blocking=True), lb.to(device, non_blocking=True), \
                          wt.to(device, non_blocking=True), mk.to(device, non_blocking=True)
        sw = mk_sw(sr).to(device, non_blocking=True)

        with torch.no_grad():
            mel = mel_transform(wav)
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            mel = spec_augment(mel)
            mel = mel.to(memory_format=torch.channels_last)

        with autocast():
            if USE_PERCH_DISTILL:
                clip_logits, framewise, distill_emb = model(mel, return_framewise=True,
                                                              return_distill=True)
            else:
                clip_logits, framewise = model(mel, return_framewise=True)

            frame_max_logits = framewise.max(dim=1).values
            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
            bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
            bce = 0.5 * bce_clip + 0.5 * bce_frame
            ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
            cls_loss = (ps * sw).mean()

            if USE_PERCH_DISTILL and perch_teacher is not None:
                with torch.no_grad():
                    wav_5s = wav.squeeze(1)
                    N = wav_5s.shape[1]
                    if N > 160000:
                        start_off = (N - 160000) // 2
                        wav_5s = wav_5s[:, start_off:start_off + 160000]
                    elif N < 160000:
                        wav_5s = F.pad(wav_5s, (0, 160000 - N))
                    perch_emb = perch_teacher.embed(wav_5s).to(device)
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss
            else:
                distill_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        el += float(loss.item())
        el_cls += float(cls_loss.item())
        el_dist += float(distill_loss.item())
        nb_count += 1

        if batch_idx % 20 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    ep{epoch+1:02d} batch {batch_idx:4d}/{N_STEPS_PER_EP}  "
                  f"loss={loss.item():.4f}  cls={cls_loss.item():.4f}  "
                  f"dist={distill_loss.item():.4f}  lr={cur_lr:.2e}", flush=True)

    train_loss_avg = el / max(nb_count, 1)
    cls_loss_avg = el_cls / max(nb_count, 1)
    dist_loss_avg = el_dist / max(nb_count, 1)

    if len(val_wavs) > 0:
        val_preds_dict = _predict_from_waveforms(model, mel_transform, val_wavs)
        val_preds = val_preds_dict["blend"]
        r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
        val_metrics = {
            "ns22": r["non_s22_macro"],
            "macro": r["macro_auc_all"],
            "per_taxon": r["per_taxon"],
        }
    else:
        val_metrics = {"ns22": float("nan"), "macro": float("nan"), "per_taxon": {}}

    ep_elapsed = time.time() - ep_start
    epoch_times.append(ep_elapsed)
    trained_this_session += 1
    ep = epoch + 1

    state_to_save = {
        "epoch": ep,
        "fold": FOLD_K,
        "n_total_epochs": N_TOTAL_EPOCHS,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_ns22": best_ns22,
        "best_macro": best_macro,
        "history": history,
    }
    torch.save(state_to_save, OUT_DIR / "ckpt_latest.pth")
    torch.save(state_to_save, OUT_DIR / f"ckpt_ep{ep:02d}.pth")

    is_best_ns22 = (not math.isnan(val_metrics["ns22"])) and val_metrics["ns22"] > best_ns22
    is_best_macro = (not math.isnan(val_metrics["macro"])) and val_metrics["macro"] > best_macro
    if is_best_ns22:
        best_ns22 = val_metrics["ns22"]
        state_to_save["best_ns22"] = best_ns22
        torch.save(state_to_save, OUT_DIR / "ckpt_best_ns22.pth")
    if is_best_macro:
        best_macro = val_metrics["macro"]
        state_to_save["best_macro"] = best_macro
        torch.save(state_to_save, OUT_DIR / "ckpt_best_macro.pth")

    cur_lr = optimizer.param_groups[0]["lr"]
    history.append({
        "epoch": ep,
        "train_loss": round(train_loss_avg, 5),
        "cls_loss": round(cls_loss_avg, 5),
        "dist_loss": round(dist_loss_avg, 5),
        "val_ns22": val_metrics["ns22"],
        "val_macro": val_metrics["macro"],
        "val_per_taxon": val_metrics["per_taxon"],
        "lr": cur_lr,
        "elapsed_sec": round(ep_elapsed, 1),
    })
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    mirror_to_drive(OUT_DIR / "ckpt_latest.pth")
    mirror_to_drive(OUT_DIR / "history.json")
    if is_best_ns22:
        mirror_to_drive(OUT_DIR / "ckpt_best_ns22.pth")
    if is_best_macro:
        mirror_to_drive(OUT_DIR / "ckpt_best_macro.pth")

    total_elapsed = time.time() - TRAIN_START
    print(f"\n=== Ep {ep}/{N_TOTAL_EPOCHS}: "
          f"train_loss={train_loss_avg:.4f} cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
          f"val_ns22={val_metrics['ns22']:.4f} val_macro={val_metrics['macro']:.4f} "
          f"({ep_elapsed:.0f}s, total {total_elapsed/60:.1f}min) ===\n")

    gc.collect()
    torch.cuda.empty_cache()

trained_up_to = ep
print(f"\n=== Session done: trained epochs {start_epoch+1}-{trained_up_to} this session ===")
print(f"    Cumulative: {trained_up_to}/{N_TOTAL_EPOCHS} epochs done")
""", "train_loop"))

# =============================================================================
# Cell 12: Final Drive mirror (all per-epoch ckpts)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 12: Final Drive mirror — all per-epoch ckpts
# ============================================================
print("Mirroring all artifacts to Drive...")
n_copied = 0
for f in sorted(OUT_DIR.glob("*")):
    if not f.is_file(): continue
    if f.suffix not in {".pth", ".json", ".txt"}: continue
    try:
        shutil.copy2(str(f), str(DRIVE_OUTPUT_DIR / f.name))
        n_copied += 1
        print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
    except Exception as e:
        print(f"  [WARN] {f.name}: {e}")
print(f"\nMirrored {n_copied} files to {DRIVE_OUTPUT_DIR}")
""", "drive_mirror"))

# =============================================================================
# Cell 13: Session summary
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 13: Session summary (train phase)
# ============================================================
total_time = time.time() - TRAIN_START
print(f"\n{'='*60}")
print(f"R3 Train Session summary")
print(f"{'='*60}")
print(f"  Resume status:        {'RESUMED' if resumed else 'FRESH START'}")
print(f"  Resumed from epoch:   {start_epoch}")
print(f"  Trained this session: {trained_this_session} epoch(s)")
print(f"  Cumulative:           {trained_up_to}/{N_TOTAL_EPOCHS} epochs")
print(f"  Best ns22 AUC:        {best_ns22:.4f}")
print(f"  Best macro AUC:       {best_macro:.4f}")
print(f"  Total session time:   {total_time/60:.1f} min")
print(f"  Drive output:         {DRIVE_OUTPUT_DIR}")
if trained_up_to >= N_TOTAL_EPOCHS:
    print(f"\n  >>> R3 Training COMPLETE — proceed to pseudo phase <<<")
else:
    print(f"\n  ... R3 not yet complete; re-run NB to continue.")
    print(f"  ... Skip pseudo/upload cells until training completes.")
""", "summary"))

# =============================================================================
# Cell 14: Pseudo phase header
# =============================================================================
cells.append(md_cell(r"""## ─────── R3 Pseudo Phase ───────

R3 学生の best ckpt で全 train_soundscapes 推論 → R3 pseudo CSV を Drive に保存。
exp017 R4 等で使う想定 (Babych 経験則では R4 はほぼゼロ gain)。
""", "pseudo_hdr"))

# =============================================================================
# Cell 15: Pseudo setup + free VRAM
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 15: Pseudo setup — free VRAM, load best ckpt
# ============================================================
import glob
from scipy.ndimage import gaussian_filter1d

# Post-processing config (R1 と同じ)
N_WINDOWS    = 12
CHUNK_N      = TRAIN_SAMPLES
GAUSS_SIGMA  = 0.65
POWER_GAMMA  = 1.2

# Free VRAM from training
del optimizer, scheduler, scaler, mel_transform, spec_augment
if perch_teacher is not None:
    del perch_teacher
del model
gc.collect()
torch.cuda.empty_cache()

# Best ckpt path
CKPT_PATH = OUT_DIR / "ckpt_best_ns22.pth"
if not CKPT_PATH.exists():
    CKPT_PATH = OUT_DIR / "ckpt_latest.pth"
assert CKPT_PATH.exists(), f"No ckpt at {OUT_DIR}"
print(f"Using ckpt: {CKPT_PATH}")
print(f"R3 pseudo dir: {DRIVE_R3_PSEUDO_DIR}")

try:
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(CKPT_PATH), map_location="cpu")

print(f"  epoch={state.get('epoch')}, "
      f"best_ns22={state.get('best_ns22', float('nan')):.4f}, "
      f"best_macro={state.get('best_macro', float('nan')):.4f}")

model = BirdSEDModel().to(device)
model.load_state_dict(state["model_state"], strict=False)
model.eval()
model = model.to(memory_format=torch.channels_last)
print(f"OK model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

mel_tf = MelSpecTransform().to(device)
""", "pseudo_setup"))

# =============================================================================
# Cell 16: Inference on all train_soundscapes
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 16: Pseudo R3 inference — all train_soundscapes
# ============================================================
import soundfile as sf
import librosa
from tqdm.auto import tqdm

def load_audio_32k_mono(path, target_samples=60 * SR):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < target_samples:
        wav = np.pad(wav, (0, target_samples - len(wav)))
    elif len(wav) > target_samples:
        wav = wav[:target_samples]
    return wav.astype(np.float32)

def file_to_chunks(path):
    wav = load_audio_32k_mono(path, target_samples=N_WINDOWS * CHUNK_N)
    return wav.reshape(N_WINDOWS, CHUNK_N).astype(np.float32)

sc_files = sorted(glob.glob(str(TS_DIR / "*.ogg")))
print(f"train_soundscapes: {len(sc_files)} files")
assert len(sc_files) > 0

all_filenames, all_start_secs, all_end_secs, all_probs = [], [], [], []
t0 = time.time()

with torch.no_grad():
    for fi, fpath in enumerate(tqdm(sc_files, desc="infer", mininterval=2.0)):
        stem = Path(fpath).stem
        try:
            chunks = file_to_chunks(fpath)
        except Exception as e:
            print(f"WARN: {stem}: {e}")
            chunks = np.zeros((N_WINDOWS, CHUNK_N), dtype=np.float32)

        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)
        mel = mel_tf(wav_t)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        mel = mel.to(memory_format=torch.channels_last)

        with autocast():
            clip_logits, framewise = model(mel, return_framewise=True)
            frame_max = framewise.max(dim=1).values
            p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()
            p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
        probs_file = 0.5 * p_clip + 0.5 * p_fmax
        probs_file = gaussian_filter1d(probs_file, sigma=GAUSS_SIGMA, axis=0,
                                        mode="nearest").astype(np.float32)

        all_probs.append(probs_file)
        for wi in range(N_WINDOWS):
            all_filenames.append(stem)
            all_start_secs.append(wi * TRAIN_DURATION)
            all_end_secs.append((wi + 1) * TRAIN_DURATION)

prob_mat = np.concatenate(all_probs, axis=0).astype(np.float32)
filenames_arr = np.array(all_filenames)
start_secs_arr = np.array(all_start_secs, dtype=np.float32)
end_secs_arr   = np.array(all_end_secs,   dtype=np.float32)
print(f"\nInference: {prob_mat.shape}, mean={prob_mat.mean():.4f}, max={prob_mat.max():.4f} "
      f"in {(time.time()-t0)/60:.1f} min")
""", "pseudo_infer"))

# =============================================================================
# Cell 17: Postproc + save CSV to Drive
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 17: Pseudo R3 postproc + save CSV + Drive mirror
# ============================================================
print(f"Pre-PT: mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}")

prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)

print(f"Post-PT (γ={POWER_GAMMA}): mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")

row_max = prob_mat.max(axis=1)
print(f"row_max stats (diagnostic, NO filter): 50%ile={np.percentile(row_max,50):.4f}, "
      f"99%ile={np.percentile(row_max,99):.4f}")

df = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)
df.insert(0, "filename",  filenames_arr)
df.insert(1, "start_sec", start_secs_arr)
df.insert(2, "end_sec",   end_secs_arr)

local_csv = OUT_DIR / "pseudo_labels.csv"
df.to_csv(local_csv, index=False)
print(f"\nLocal: {local_csv} ({local_csv.stat().st_size/1024/1024:.1f}MB)")
print(f"  shape: {df.shape}, files: {df['filename'].nunique()}")

# Mirror to Drive
drive_csv = DRIVE_R3_PSEUDO_DIR / "pseudo_labels.csv"
sz = local_csv.stat().st_size
print(f"\nMirroring to Drive: {drive_csv} ({sz/1e6:.1f}MB)")
t0 = time.time()
with open(local_csv, "rb") as fin, open(drive_csv, "wb") as fout:
    pbar = tqdm(total=sz, unit="B", unit_scale=True, unit_divisor=1024,
                desc="Drive mirror", mininterval=1.0)
    while True:
        buf = fin.read(8*1024*1024)
        if not buf: break
        fout.write(buf); pbar.update(len(buf))
    pbar.close()
print(f"  done in {time.time()-t0:.0f}s")

meta = {
    "backbone": BACKBONE,
    "round": "R3",
    "ckpt": str(CKPT_PATH.name),
    "epoch": state.get("epoch"),
    "best_ns22": state.get("best_ns22"),
    "best_macro": state.get("best_macro"),
    "n_files": int(df["filename"].nunique()),
    "n_rows": int(len(df)),
    "gauss_sigma": GAUSS_SIGMA,
    "power_gamma": POWER_GAMMA,
}
(DRIVE_R3_PSEUDO_DIR / "pseudo_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(f"  meta written: {DRIVE_R3_PSEUDO_DIR}/pseudo_meta.json")

# Free pseudo memory before upload
del prob_mat, df, all_probs, all_filenames
gc.collect(); torch.cuda.empty_cache()
""", "pseudo_save"))

# =============================================================================
# Cell 18: Kaggle Dataset upload (R3 weights)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 18: Upload R3 weights to Kaggle Dataset (version up existing slug)
# ============================================================
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

USER  = "maekeso"
SLUG  = "birdclef2026-exp017-weights"
TITLE = "birdclef2026 exp017 weights"

# R3 ckpts to upload (R3 best ns22 / macro / latest + history)
# 注: 同じ Dataset slug に R3 ckpt も追加する (version up、R1/R2 は保持)
TARGETS = [
    ("ckpt_best_ns22.pth", "r3_ckpt_best_ns22.pth"),
    ("ckpt_best_macro.pth", "r3_ckpt_best_macro.pth"),
    ("ckpt_latest.pth", "r3_ckpt_latest.pth"),
    ("history.json", "r3_history.json"),
]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_copied = 0
    for src_name, dst_name in TARGETS:
        src = OUT_DIR / src_name
        if not src.exists():
            print(f"  skip (missing): {src_name}")
            continue
        shutil.copy2(str(src), str(td / dst_name))
        n_copied += 1
        print(f"  staged: {dst_name}  ({src.stat().st_size/1e6:.1f}MB)")

    # Pull existing R1 + R2 files from the previous Dataset version into the new version
    # (otherwise Kaggle replaces — version up = full replacement)
    print(f"\nFetching R1 + R2 files from existing Dataset version to preserve them...")
    try:
        tmp_dl = td / "_r1_existing"
        tmp_dl.mkdir(exist_ok=True)
        api.dataset_download_files(f"{USER}/{SLUG}", path=str(tmp_dl), unzip=True, quiet=False)
        for f in tmp_dl.iterdir():
            if f.is_file() and not f.name.startswith("r3_") and not (td / f.name).exists():
                shutil.copy2(str(f), str(td / f.name))
                print(f"  preserved existing: {f.name}")
        shutil.rmtree(tmp_dl)
    except Exception as e:
        print(f"  [WARN] could not preserve existing files: {str(e)[:200]}")
        print(f"  R3 will be uploaded as a fresh version (R1/R2 files may be replaced)")

    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = f"R3 best_ns22={best_ns22:.4f} (R2 was 0.9249)"
    uploaded = False

    # Pre-check existence (Kaggle returns 403 not 404 for missing dataset)
    try:
        api.dataset_view(f"{USER}/{SLUG}")
        dataset_exists = True
        print(f"\nDataset {USER}/{SLUG} exists → create new VERSION")
    except Exception:
        dataset_exists = False
        print(f"\nDataset {USER}/{SLUG} not found → CREATE new dataset")

    if dataset_exists:
        try:
            api.dataset_create_version(folder=str(td),
                                        version_notes=version_notes,
                                        dir_mode="zip", quiet=False)
            print(f"\nOK Uploaded (new version) — {version_notes}")
            uploaded = True
        except Exception as e:
            print(f"  dataset_create_version error: {str(e)[:400]}")
    else:
        try:
            api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
            print(f"\nOK Created (first time)")
            uploaded = True
        except Exception as e:
            print(f"  dataset_create_new error: {str(e)[:400]}")

    assert uploaded, "Upload failed — check logs above"
    print(f"\nDataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
    print(f"\nNext: ローカル Windows で push_nb_infer.py を実行 → Kaggle で submit")
""", "upload_state"))

# =============================================================================
# Cell 19: Terminate Colab runtime (save compute units)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 19: Terminate Colab runtime — save Pro+ compute units
# ============================================================
# ここまで完走したら Blackwell を即座に切る。Drive 出力済 + Kaggle Dataset push 済で残作業なし。
print("All R3 phases complete. Terminating Colab runtime in 5s to free compute units...")
import time as _t
_t.sleep(5)
from google.colab import runtime
runtime.unassign()
""", "terminate"))

# =============================================================================
# Assemble notebook
# =============================================================================
nb_out = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb_train_r3.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
