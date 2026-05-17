"""Generate exp020 NB: eca_nfnet_l0 5-fold + TTA + Perch cache (Colab Pro Blackwell).

Design:
- Base: exp017 R2 NB を 5-fold + Perch cache + Pseudo TTA に拡張
- Perch cache: maekeso/birdclef2026-perch-emb-cache (1.1 GB) を /content に展開
- 5-fold loop: 1 NB 内で fold 0-4 を順次学習、各 fold で R1 → pseudo → R2 → pseudo
- TTA: pseudo 生成時に delta-shift TTA (3 view: 0s, +0.5s, -0.5s) を適用
- Resume: 各 fold の R2 ckpt が Drive にあれば skip、session 切れ対応
- Output: Kaggle Dataset `maekeso/birdclef2026-exp020-weights-5fold` (新規) に全 fold ckpt upload
- Backbone: eca_nfnet_l0 (NFNet rule: LR=3e-4 据え置き、sqrt scaling 無し)

Estimated time (Blackwell, 5-fold with Perch cache + Pseudo TTA + B-set params):
- Per fold: R1 ~25 min + pseudo+TTA ~22 min + R2 ~1.1h (15 ep) + pseudo+TTA ~22 min ≈ 2.2h
- 5-fold total: ~11h, Pro+ 24h session 内で余裕完走
- B 案: N_EPOCHS_R2=15 (20→15), WARMUP=2 (3→2), NUM_WORKERS=12 (8→12)

Run: python _gen_nb_train_5fold_tta.py  ->  writes nb_train_5fold_tta.ipynb
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
# Cell 0: Header
# =============================================================================
cells.append(md_cell(r"""# exp020 — eca_nfnet_l0 5-fold + TTA + Perch cache (Colab Pro Blackwell)

**Goal:** exp017 R2 (LB 0.921) の構成を **5-fold ensemble + Perch cache 高速化 + Pseudo gen TTA** に拡張。
gold (0.95+) 突破の本命 path。

## 主要変更点 (exp017 R2 比)

| 項目 | exp017 R2 (基準) | **exp020 (新)** |
|---|---|---|
| Fold 数 | 1 | **5** |
| Perch teacher | per-step ONNX 推論 | **pre-computed cache (np.memmap)** |
| Pseudo gen | 単一推論 | **delta-shift TTA (3 view)** |
| 学習時間 (5-fold 総計) | (1 fold 3.5h × 5 = 17.5h) | **~14h (cache 込み)** |
| 出力 | exp017 single | exp020-weights-5fold (新 Kaggle Dataset) |

## NFNet rule (継承)
- LR=3e-4 据え置き ([[feedback_nfnet_lr_no_sqrt_scaling]])
- batch=192 (Blackwell VRAM 活用)
- N_TOTAL_EPOCHS: R1=25, **R2=15 (時短、best_ckpt で peak 保証)**
- **WARMUP_EPOCHS=2** (3→2)、**NUM_WORKERS=12** (8→12)

## Pseudo TTA (新規)
delta-shift × 3 view (0s, +0.5s, -0.5s) を平均、Gaussian smooth + Power Transform γ=1.2。
時間追加: pseudo gen が 3x になるが Perch cache の節約で総合 -50% 時間。

## 5-fold loop (新規)
```
for fold in [0, 1, 2, 3, 4]:
    if drive_already_has(fold): skip
    else:
        train_r1(fold)        # focal 0.85 + labeled_sc 0.15
        gen_pseudo_tta(fold)  # R1 pseudo + TTA → r1-pseudo
        train_r2(fold)        # focal 0.70 + labeled_sc 0.10 + pseudo_sc 0.20
        gen_pseudo_tta(fold)  # R2 pseudo + TTA → r2-pseudo (optional, R3 candidate)
        mirror_drive(fold)
```

## 想定 LB
- 単一 fold: 0.921 (exp017 と同等)
- 5-fold ensemble (infer): **0.928-0.935** (val→LB gap -0.004 が継続なら)
- blend に投入で **0.950+** 射程

## Drive 構成
- input: `/content/drive/MyDrive/kaggle/birdclef2026/`
- output: `output/exp020/fold{k}/r1/, r2/, r1-pseudo/, r2-pseudo/` (k=0-4)
- Perch cache: Kaggle Dataset `maekeso/birdclef2026-perch-emb-cache` から DL

## 最後の Cell
`runtime.unassign()` で自動切断 (unit 節約)
""", "hdr"))

# =============================================================================
# Cell 1: Setup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup — Drive mount, pip install, kaggle.json
# ============================================================
!pip install -q timm librosa soundfile scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

DRIVE_INPUT_DIR  = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp020"
DRIVE_EXP_DIR.mkdir(parents=True, exist_ok=True)
assert DRIVE_INPUT_DIR.exists(), f"Drive input folder missing: {DRIVE_INPUT_DIR}"
print(f"Drive input:  {DRIVE_INPUT_DIR}")
print(f"Drive exp020: {DRIVE_EXP_DIR}")

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
# Cell 2: Data DL + Perch cache DL
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 2: Data DL — Kaggle API (competition + Perch cache)
# ============================================================
import time, zipfile, subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm.auto import tqdm

api = KaggleApi(); api.authenticate()
print("kaggle authenticated")

T0_total = time.time()

# ===== Competition data =====
TA_DIR = LOCAL_DATA / "train_audio"
TS_DIR = LOCAL_DATA / "train_soundscapes"

need_dl = (
    not TA_DIR.exists() or sum(1 for _ in TA_DIR.rglob("*.ogg")) < 40000 or
    not TS_DIR.exists() or sum(1 for _ in TS_DIR.glob("*.ogg")) < 10000
)

if need_dl:
    print(f"\n[1/3] Downloading birdclef-2026 competition data (~25GB)...")
    t0 = time.time()
    api.competition_download_files("birdclef-2026", path=str(LOCAL_DATA),
                                    force=False, quiet=False)
    print(f"  DL done in {(time.time()-t0)/60:.1f} min")
    zips = list(LOCAL_DATA.glob("birdclef-2026*.zip"))
    assert zips, "No birdclef-2026 zip found"
    zip_path = zips[0]
    print(f"\n  Extracting {zip_path.name} ({zip_path.stat().st_size/1e9:.1f}GB)...")
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

# CSVs to /content/data/competition/
comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
    src_in_root = LOCAL_DATA / fn
    dst = comp / fn
    if src_in_root.exists() and not dst.exists():
        shutil.copy2(str(src_in_root), str(dst))

# ===== Perch cache =====
PERCH_CACHE_DIR = LOCAL_DATA / "perch-cache"
EMB_PATH = PERCH_CACHE_DIR / "emb.npy"
META_PATH = PERCH_CACHE_DIR / "meta.csv"

if not EMB_PATH.exists() or not META_PATH.exists():
    print(f"\n[2/3] Downloading Perch cache (~1.1GB) from Kaggle Dataset...")
    PERCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    api.dataset_download_files("maekeso/birdclef2026-perch-emb-cache",
                                path=str(PERCH_CACHE_DIR), unzip=True, quiet=False)
    print(f"  DL done in {(time.time()-t0)/60:.1f} min")
else:
    print("\nPerch cache already present locally")

assert EMB_PATH.exists(), f"Perch cache emb.npy missing: {EMB_PATH}"
assert META_PATH.exists(), f"Perch cache meta.csv missing: {META_PATH}"
print(f"  emb.npy: {EMB_PATH.stat().st_size/1e9:.2f} GB")
print(f"  meta.csv: {META_PATH.stat().st_size/1e6:.2f} MB")

print(f"\n=== Total prep time: {(time.time()-T0_total)/60:.1f} min ===")
r = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
print(r.stdout)
""", "dl_data"))

# =============================================================================
# Cell 3: Imports + Config
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
from scipy.ndimage import gaussian_filter1d
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
PERCH_CACHE_DIR = LOCAL_DATA / "perch-cache"
EMB_PATH = PERCH_CACHE_DIR / "emb.npy"
META_PATH = PERCH_CACHE_DIR / "meta.csv"

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
FOLDS = [0, 1, 2, 3, 4]   # ★ 5-fold
FORCE_FRESH_PER_FOLD = True   # 各 fold で新規初期化 (Babych 流)

# ★ NFNet rule: LR sqrt scaling 厳禁、batch 増でも LR 据え置き ([[feedback_nfnet_lr_no_sqrt_scaling]])
# ★ B 案: R2 短縮 + warmup 短縮 で時短 (best_ckpt 保存で peak 取り損ねなし)
N_TOTAL_EPOCHS_R1 = 25
N_TOTAL_EPOCHS_R2 = 15     # 20 → 15 (pseudo peak ep 4-8、15 ep で十分 buffer)
BATCH = 192
LR = 3e-4
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 2          # 3 → 2 (warmup 比率 8-12% に最適化)

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

# Source mix per round
SHARES_R1 = {"focal": 0.85, "labeled_sc": 0.15}
SHARES_R2 = {"focal": 0.70, "labeled_sc": 0.10, "pseudo_sc": 0.20}
SOURCE_WEIGHTS = {
    "focal":          1.0,
    "focal_missing":  0.0,
    "labeled_sc":     1.0,
    "pseudo_sc":      0.5,   # noisier than hard labels
}

NUM_WORKERS = 12           # ★ B 案: 8 → 12、data loading 並列度 up (+10-15% 速)
PERSISTENT_WORKERS = True

# ===== Pseudo TTA settings =====
TTA_SHIFTS_SAMPLES = [0, SR // 2, -SR // 2]   # 3 view: 0s, +0.5s, -0.5s
GAUSS_SIGMA  = 0.65
POWER_GAMMA  = 1.2
N_WINDOWS = 12   # 60s SS file → 12 × 5s chunks
CHUNK_N   = TRAIN_SAMPLES

# ===== Timer =====
SESSION_START = time.time()
MAX_RUNTIME_SEC = 22.0 * 3600   # 22h safety margin

print(f"Backbone: {BACKBONE}")
print(f"Folds: {FOLDS}")
print(f"Batch: {BATCH} | R1 epochs: {N_TOTAL_EPOCHS_R1} | R2 epochs: {N_TOTAL_EPOCHS_R2}")
print(f"LR: {LR} | WD: {WD} | warmup: {WARMUP_EPOCHS}ep")
print(f"R1 mix: {SHARES_R1}")
print(f"R2 mix: {SHARES_R2}")
print(f"TTA shifts: {[f'{s/SR:+.1f}s' for s in TTA_SHIFTS_SAMPLES]}")
print(f"Max runtime: {MAX_RUNTIME_SEC/3600:.1f}h")
""", "config"))

# =============================================================================
# Cell 4: Load CSVs + Perch cache lookup tables
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 4: Load CSVs + Perch cache lookup tables
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

print("Checking focal file existence...")
_t0 = time.time()
train_df["exists"] = train_df["filename"].map(lambda fn: (TA_DIR / fn).exists())
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"  {len(train_df)} focal files exist ({time.time()-_t0:.1f}s)")
train_df["original_idx"] = np.arange(len(train_df))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fold

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
    sc_meta = (sc_labels_raw[["filename", "start_sec"]].drop_duplicates().reset_index(drop=True))
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
    sc_files = sc_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_meta["fold"] = sc_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    non_s22_mask_sc = (sc_meta["site"].values != "S22")
    print(f"Labeled SS: {len(sc_meta)} windows, {int(Y_SC.sum())} positives")
else:
    print("LABELS_PATH missing")
    sc_meta = pd.DataFrame(columns=["filename", "start_sec", "site", "fold"])
    Y_SC = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    non_s22_mask_sc = np.zeros(0, dtype=bool)

# ===== Perch cache lookup tables =====
print(f"\nLoading Perch cache meta.csv...")
meta_df = pd.read_csv(META_PATH)
print(f"  {len(meta_df)} cached chunks")
print(f"  sources: {meta_df['source'].value_counts().to_dict()}")

# Focal: filename -> list of (chunk_idx, row_idx)
focal_meta = meta_df[meta_df["source"] == "focal"].reset_index(drop=True)
focal_chunk_lookup = {}
for fn, sub in focal_meta.groupby("filename"):
    focal_chunk_lookup[fn] = sub[["chunk_idx", "row_idx"]].values.astype(np.int32)
print(f"  focal: {len(focal_chunk_lookup)} files with cache")

# SS: (filename, chunk_idx) -> row_idx ; SS filename in cache may or may not have .ogg suffix
ss_meta = meta_df[meta_df["source"] == "ss"].reset_index(drop=True)
ss_lookup = {}
for fn, sub in ss_meta.groupby("filename"):
    # Strip .ogg if present, store both forms
    base = fn.rsplit(".ogg", 1)[0] if fn.endswith(".ogg") else fn
    for ci, ri in sub[["chunk_idx", "row_idx"]].values:
        ss_lookup[(base, int(ci))] = int(ri)
        ss_lookup[(fn, int(ci))]   = int(ri)
print(f"  ss: {len(set(ss_meta['filename']))} files, {len(ss_lookup)} (file,chunk) entries")

# emb.npy memmap
print(f"\nMemory-mapping emb.npy ({EMB_PATH.stat().st_size/1e9:.2f} GB)...")
EMB_MMAP = np.load(str(EMB_PATH), mmap_mode="r")
print(f"  shape: {EMB_MMAP.shape}, dtype: {EMB_MMAP.dtype}")
assert EMB_MMAP.shape[1] == PERCH_EMBED_DIM

print("\nOK CSVs + Perch cache loaded")
""", "load_data"))

# =============================================================================
# Cell 5: Model defs (no PerchTeacher class needed — cache lookup)
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 5: Model — Mel + SpecAugment + DistillHead + BirdSEDModel
#  NOTE: PerchTeacher class is dropped, training reads emb directly from cache.
# ============================================================
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

class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        gap = feature_map.mean(dim=[2, 3])
        return self.proj(gap)

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

print("OK model defs ready (no PerchTeacher; cache lookup integrated)")
""", "model"))

# =============================================================================
# Cell 6: Dataset defs with Perch cache lookup
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 6: Datasets — FocalDS, LabeledSCDS, PseudoScDS with Perch cache emb
# ============================================================
import soundfile as sf
import librosa
from functools import lru_cache

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

def _load_ogg_chunk(path, chunk_idx, n_samples_target=TRAIN_SAMPLES):
    '''Load 5s chunk at chunk_idx*5s (chunk-aligned, matches Perch cache).'''
    wav = _load_full_audio_cached(str(path))
    if wav is None:
        return None
    start = chunk_idx * n_samples_target
    end = start + n_samples_target
    if end <= len(wav):
        return wav[start:end].copy()
    out = np.zeros(n_samples_target, dtype=np.float32)
    avail = wav[start:start + n_samples_target] if start < len(wav) else np.array([])
    out[:len(avail)] = avail
    return out

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

    def _load_chunk_random(self, i):
        fn = self.filenames[i]
        path = TA_DIR / fn
        # pick a random chunk_idx from cache (fallback to chunk 0 if no cache)
        chunks_info = focal_chunk_lookup.get(fn)
        if chunks_info is None or len(chunks_info) == 0:
            chunk = _load_ogg_chunk(path, 0)
            row_idx = -1
        else:
            ci_idx = np.random.randint(len(chunks_info)) if self.aug else 0
            chunk_idx, row_idx = chunks_info[ci_idx]
            chunk = _load_ogg_chunk(path, int(chunk_idx))
        if chunk is None:
            return None, None, -1
        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        if self.primary[i] in self.l2i:
            lb[self.l2i[self.primary[i]]] = 1.0
        if self.secondary_lookup is not None and self.original_idx is not None:
            for s in self.secondary_lookup.get(int(self.original_idx[i]), []):
                if s in self.l2i: lb[self.l2i[s]] = 1.0
        return chunk, lb, int(row_idx)

    def __getitem__(self, i):
        ch1, lb1, ri1 = self._load_chunk_random(i)
        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                    torch.zeros(PERCH_EMBED_DIM), torch.ones(NUM_CLASSES),
                    torch.ones(NUM_CLASSES), "focal_missing")

        # Mixup at audio level (label + Perch emb interpolated similarly)
        if USE_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            for _ in range(3):
                j = np.random.randint(len(self.df))
                ch2, lb2, ri2 = self._load_chunk_random(j)
                if ch2 is not None: break
            if ch2 is not None:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                if self.aug: ch_mix = apply_aug(ch_mix)
                lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                # Perch emb mixup: linear interp (deterministic since both pre-computed)
                emb1 = EMB_MMAP[ri1].astype(np.float32) if ri1 >= 0 else np.zeros(PERCH_EMBED_DIM, dtype=np.float32)
                emb2 = EMB_MMAP[ri2].astype(np.float32) if ri2 >= 0 else np.zeros(PERCH_EMBED_DIM, dtype=np.float32)
                emb_mix = (lam * emb1 + (1 - lam) * emb2).astype(np.float32)
                return (torch.from_numpy(ch_mix).unsqueeze(0),
                        torch.from_numpy(lb.astype(np.float32)),
                        torch.from_numpy(emb_mix),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")

        if self.aug: ch1 = apply_aug(ch1)
        emb = EMB_MMAP[ri1].astype(np.float32) if ri1 >= 0 else np.zeros(PERCH_EMBED_DIM, dtype=np.float32)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1),
                torch.from_numpy(emb),
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
        fn_with_ext = fn if fn.endswith(".ogg") else fn + ".ogg"
        path = TS_DIR / fn_with_ext
        start_sec = int(self.start_secs[i])
        chunk_idx = start_sec // 5
        wav = _load_ogg_chunk(path, chunk_idx)
        if wav is None:
            wav = np.zeros(TRAIN_SAMPLES, dtype=np.float32)
        if self.aug:
            wav = apply_aug(wav)
        row_idx = ss_lookup.get((fn, chunk_idx), ss_lookup.get((fn_with_ext, chunk_idx), -1))
        emb = EMB_MMAP[row_idx].astype(np.float32) if row_idx >= 0 else np.zeros(PERCH_EMBED_DIM, dtype=np.float32)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.from_numpy(emb),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "labeled_sc")


class PseudoScDS(Dataset):
    '''Pseudo-labeled SS windows from prior round.'''
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
        fn_with_ext = fn if fn.endswith(".ogg") else fn + ".ogg"
        path = self.audio_dir / fn_with_ext
        start_sec = float(self.start_secs[i])
        chunk_idx = int(start_sec // 5)
        wav = _load_ogg_chunk(path, chunk_idx)
        if wav is None:
            wav = np.zeros(TRAIN_SAMPLES, dtype=np.float32)
        if self.aug:
            wav = apply_aug(wav)
        row_idx = ss_lookup.get((fn, chunk_idx), ss_lookup.get((fn_with_ext, chunk_idx), -1))
        emb = EMB_MMAP[row_idx].astype(np.float32) if row_idx >= 0 else np.zeros(PERCH_EMBED_DIM, dtype=np.float32)
        return (torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i]),
                torch.from_numpy(emb),
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
    return (torch.stack([b[0] for b in batch]),   # wav
            torch.stack([b[1] for b in batch]),   # label
            torch.stack([b[2] for b in batch]),   # perch_emb (NEW)
            torch.stack([b[3] for b in batch]),   # wt
            torch.stack([b[4] for b in batch]),   # mk
            [b[5] for b in batch])                # sr (source name)

def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)

print("OK datasets ready (Perch cache integrated)")
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
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22_mask)
    r["non_s22_macro"] = round(float(a) if not np.isnan(a) else 0.0, 4)
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
        fn_with_ext = fn if fn.endswith(".ogg") else fn + ".ogg"
        start_sec = int(row["start_sec"])
        chunk_idx = start_sec // 5
        wav = _load_ogg_chunk(TS_DIR / fn_with_ext, chunk_idx)
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
    preds_blend = []
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
            preds_blend.append(p_blend)
    return np.concatenate(preds_blend)

print("OK eval helpers ready")
""", "eval"))

# =============================================================================
# Cell 8: Pseudo generation with TTA
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 8: Pseudo generation with delta-shift TTA
# ============================================================
import glob as _glob

def _file_to_chunks(path):
    '''Load 60s audio, reshape into N_WINDOWS x CHUNK_N. Used for pseudo inference.'''
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    target = N_WINDOWS * CHUNK_N
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


def _predict_chunks(model, mel_tf, chunks_np):
    '''chunks_np: (N_WINDOWS, CHUNK_N) -> probs (N_WINDOWS, NUM_CLASSES).'''
    wav_t = torch.from_numpy(chunks_np).unsqueeze(1).to(device)
    mel = mel_tf(wav_t)
    for i in range(mel.size(0)):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    mel = mel.to(memory_format=torch.channels_last)
    with autocast():
        clip_logits, framewise = model(mel, return_framewise=True)
        frame_max = framewise.max(dim=1).values
        p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()
        p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()
    return 0.5 * p_clip + 0.5 * p_fmax


def generate_pseudo_with_tta(model, mel_tf, ts_files, tta_shifts=TTA_SHIFTS_SAMPLES,
                              gauss_sigma=GAUSS_SIGMA, power_gamma=POWER_GAMMA):
    '''Generate pseudo with delta-shift TTA. Returns DataFrame ready for CSV.'''
    all_filenames, all_start_secs, all_end_secs, all_probs = [], [], [], []
    t0 = time.time()
    N_FILES = len(ts_files)

    with torch.no_grad():
        for fi, fpath in enumerate(ts_files):
            stem = Path(fpath).stem
            try:
                wav_60s = _file_to_chunks(fpath)
            except Exception as e:
                print(f"  WARN {stem}: {e}")
                wav_60s = np.zeros(N_WINDOWS * CHUNK_N, dtype=np.float32)

            # TTA: shift wav by each delta, predict, average
            tta_preds = []
            for shift in tta_shifts:
                wav_shifted = np.roll(wav_60s, shift)
                chunks = wav_shifted.reshape(N_WINDOWS, CHUNK_N).astype(np.float32)
                probs = _predict_chunks(model, mel_tf, chunks)
                tta_preds.append(probs)
            probs_file = np.mean(tta_preds, axis=0)
            probs_file = gaussian_filter1d(probs_file, sigma=gauss_sigma, axis=0,
                                            mode="nearest").astype(np.float32)
            all_probs.append(probs_file)
            for wi in range(N_WINDOWS):
                all_filenames.append(stem)
                all_start_secs.append(wi * TRAIN_DURATION)
                all_end_secs.append((wi + 1) * TRAIN_DURATION)

            if (fi + 1) % 500 == 0 or fi == N_FILES - 1 or fi == 0:
                elapsed = time.time() - t0
                rate = (fi + 1) / max(elapsed, 1e-6)
                eta = (N_FILES - fi - 1) / max(rate, 1e-6)
                print(f"    pseudo+TTA [{fi+1:5d}/{N_FILES}] {elapsed:6.1f}s "
                      f"{rate:5.2f} f/s ETA {eta/60:5.1f}min", flush=True)

    prob_mat = np.concatenate(all_probs, axis=0).astype(np.float32)
    # Power Transform
    prob_mat = np.power(prob_mat, power_gamma).astype(np.float32)
    df = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)
    df.insert(0, "filename",  np.array(all_filenames))
    df.insert(1, "start_sec", np.array(all_start_secs, dtype=np.float32))
    df.insert(2, "end_sec",   np.array(all_end_secs,   dtype=np.float32))
    return df

print("OK pseudo+TTA helper ready")
""", "pseudo_tta"))

# =============================================================================
# Cell 9: build_active_datasets per fold per round
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 9: build_active_datasets(fold_k, round, pseudo_df=None)
# ============================================================
def build_active_datasets(fold_k, round_name, pseudo_meta_df=None, Y_pseudo=None):
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
    if round_name == "r2" and pseudo_meta_df is not None and Y_pseudo is not None:
        pds = PseudoScDS(pseudo_meta_df, Y_pseudo, TS_DIR, aug=True)
        items.append(("pseudo_sc", pds, len(pds)))
    return items

print("OK build_active_datasets ready")
""", "build_active"))

# =============================================================================
# Cell 10: train_one_round helper
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 10: train_one_round — encapsulates train loop for R1 or R2
# ============================================================
def train_one_round(fold_k, round_name, n_epochs, shares,
                     pseudo_meta_df=None, Y_pseudo=None,
                     out_dir=None):
    '''Train one round (R1 or R2) for one fold. Returns final (best_ns22, best_macro, model).'''
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n[Fold {fold_k} {round_name.upper()}] train ({n_epochs} ep, shares={shares})\n{'='*60}")

    active = build_active_datasets(fold_k, round_name, pseudo_meta_df, Y_pseudo)
    NAMES, DATASETS, SIZES = zip(*active)
    NAMES, DATASETS, SIZES = list(NAMES), list(DATASETS), list(SIZES)
    print(f"  Streams: {dict(zip(NAMES, SIZES))}")

    mds = ConcatDataset(DATASETS)
    n_steps_ep = max(100, int(sum(SIZES) / BATCH))
    print(f"  steps/epoch: {n_steps_ep}")

    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = GradScaler()
    warmup_steps = n_steps_ep * WARMUP_EPOCHS
    total_steps  = n_steps_ep * n_epochs
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=MIN_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    # Val prep
    if len(sc_meta) > 0:
        vm = sc_meta["fold"].values == fold_k
        val_sc_df = sc_meta[vm].reset_index(drop=True)
        Y_val = Y_SC[vm]
        ns22_val = non_s22_mask_sc[vm]
        val_wavs = _load_val_waveforms(val_sc_df)
        print(f"  Val: {len(val_wavs)} windows")
    else:
        val_wavs = []
        Y_val = np.zeros((0, NUM_CLASSES), dtype=np.float32)
        ns22_val = np.zeros(0, dtype=bool)

    mel_transform = MelSpecTransform().to(device)
    spec_augment = SpecAugment().to(device)

    best_ns22 = -1.0
    best_macro = -1.0
    history = []
    epoch_times = []

    for epoch in range(n_epochs):
        elapsed = time.time() - SESSION_START
        if epoch_times:
            est_next = max(epoch_times[-3:])
            if elapsed + est_next * 1.3 > MAX_RUNTIME_SEC:
                print(f"  [stop] time budget exhausted before ep {epoch+1}")
                break
        ep_start = time.time()
        model.train()

        smp = MixSamp(SIZES, NAMES, shares, BATCH, n_steps_ep, seed=42 + epoch)
        train_loader = DataLoader(
            mds, batch_sampler=smp, collate_fn=collate_m,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
            pin_memory=True,
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )

        el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
        for batch_idx, (wav, lb, perch_emb, wt, mk, sr) in enumerate(train_loader):
            wav = wav.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            perch_emb = perch_emb.to(device, non_blocking=True)
            wt = wt.to(device, non_blocking=True)
            mk = mk.to(device, non_blocking=True)
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
                if USE_PERCH_DISTILL:
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

            if batch_idx % 50 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"    ep{epoch+1:02d} batch {batch_idx:4d}/{n_steps_ep}  "
                      f"loss={loss.item():.4f} cls={cls_loss.item():.4f} "
                      f"dist={distill_loss.item():.4f} lr={cur_lr:.2e}", flush=True)

        train_loss_avg = el / max(nb_count, 1)
        cls_loss_avg = el_cls / max(nb_count, 1)
        dist_loss_avg = el_dist / max(nb_count, 1)

        if len(val_wavs) > 0:
            val_preds = _predict_from_waveforms(model, mel_transform, val_wavs)
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            val_ns22 = r["non_s22_macro"]
            val_macro = r["macro_auc_all"]
        else:
            val_ns22 = val_macro = float("nan")

        ep_elapsed = time.time() - ep_start
        epoch_times.append(ep_elapsed)
        ep = epoch + 1

        # Save ckpt
        state_to_save = {
            "epoch": ep, "fold": fold_k, "round": round_name,
            "n_total_epochs": n_epochs,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_ns22": best_ns22, "best_macro": best_macro,
        }
        torch.save(state_to_save, out_dir / "ckpt_latest.pth")
        is_best_ns22 = (not math.isnan(val_ns22)) and val_ns22 > best_ns22
        is_best_macro = (not math.isnan(val_macro)) and val_macro > best_macro
        if is_best_ns22:
            best_ns22 = val_ns22
            state_to_save["best_ns22"] = best_ns22
            torch.save(state_to_save, out_dir / "ckpt_best_ns22.pth")
        if is_best_macro:
            best_macro = val_macro
            state_to_save["best_macro"] = best_macro
            torch.save(state_to_save, out_dir / "ckpt_best_macro.pth")

        history.append({"epoch": ep, "train_loss": round(train_loss_avg, 5),
                        "cls_loss": round(cls_loss_avg, 5),
                        "dist_loss": round(dist_loss_avg, 5),
                        "val_ns22": val_ns22, "val_macro": val_macro,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_sec": round(ep_elapsed, 1)})
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        total_elapsed = time.time() - SESSION_START
        print(f"\n  === Ep {ep}/{n_epochs}: loss={train_loss_avg:.4f} "
              f"cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} "
              f"({ep_elapsed:.0f}s, session {total_elapsed/60:.1f}min) ===\n")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"  [{round_name.upper()} done] best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}")
    del optimizer, scheduler, scaler, mel_transform, spec_augment
    gc.collect(); torch.cuda.empty_cache()
    return best_ns22, best_macro, model

print("OK train_one_round ready")
""", "train_one_round"))

# =============================================================================
# Cell 11: Main 5-fold loop
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 11: Main 5-fold loop
# ============================================================
def mirror_dir_to_drive(local_dir: Path, drive_dir: Path):
    '''Copy ckpt + history files from local out dir to Drive.'''
    drive_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(local_dir.glob("*")):
        if not f.is_file(): continue
        if f.suffix not in {".pth", ".json", ".txt", ".csv"}: continue
        try:
            shutil.copy2(str(f), str(drive_dir / f.name))
            n += 1
        except Exception as e:
            print(f"    [WARN] mirror {f.name}: {e}")
    return n


# Track per-fold results
fold_results = {}

for FOLD_K in FOLDS:
    print(f"\n{'#'*70}\n# Fold {FOLD_K} / {N_FOLDS-1}\n{'#'*70}")
    DRIVE_FOLD_DIR = DRIVE_EXP_DIR / f"fold{FOLD_K}"
    DRIVE_R1_DIR   = DRIVE_FOLD_DIR / "r1"
    DRIVE_R2_DIR   = DRIVE_FOLD_DIR / "r2"
    DRIVE_R1_PSEUDO = DRIVE_FOLD_DIR / "r1-pseudo"
    DRIVE_R2_PSEUDO = DRIVE_FOLD_DIR / "r2-pseudo"
    for d in [DRIVE_R1_DIR, DRIVE_R2_DIR, DRIVE_R1_PSEUDO, DRIVE_R2_PSEUDO]:
        d.mkdir(parents=True, exist_ok=True)

    # ===== Resume: skip fold if final R2 ckpt exists =====
    R2_BEST = DRIVE_R2_DIR / "ckpt_best_ns22.pth"
    if R2_BEST.exists():
        print(f"  Fold {FOLD_K}: R2 best ckpt exists on Drive — skipping (resume)")
        try:
            st = torch.load(str(R2_BEST), map_location="cpu", weights_only=False)
            fold_results[FOLD_K] = {"r1": "skipped", "r2": st.get("best_ns22", -1)}
        except Exception:
            fold_results[FOLD_K] = {"r1": "skipped", "r2": "?"}
        continue

    # Local out dirs (per-fold)
    LOCAL_R1 = LOCAL_OUT / f"fold{FOLD_K}" / "r1"
    LOCAL_R2 = LOCAL_OUT / f"fold{FOLD_K}" / "r2"
    LOCAL_R1.mkdir(parents=True, exist_ok=True)
    LOCAL_R2.mkdir(parents=True, exist_ok=True)

    fold_t0 = time.time()

    # ===== R1 train =====
    R1_BEST = DRIVE_R1_DIR / "ckpt_best_ns22.pth"
    if R1_BEST.exists():
        print(f"  Fold {FOLD_K}: R1 ckpt exists on Drive, loading skip-train")
        st_r1 = torch.load(str(R1_BEST), map_location="cpu", weights_only=False)
        r1_best_ns22 = st_r1.get("best_ns22", -1)
        r1_best_macro = st_r1.get("best_macro", -1)
        # Copy local for pseudo gen
        for f in DRIVE_R1_DIR.glob("ckpt_*.pth"):
            shutil.copy2(str(f), str(LOCAL_R1 / f.name))
    else:
        r1_best_ns22, r1_best_macro, _ = train_one_round(
            fold_k=FOLD_K, round_name="r1", n_epochs=N_TOTAL_EPOCHS_R1,
            shares=SHARES_R1, out_dir=LOCAL_R1)
        # Mirror R1 ckpts to Drive
        mirror_dir_to_drive(LOCAL_R1, DRIVE_R1_DIR)

    # ===== R1 → Pseudo generation with TTA =====
    R1_PSEUDO_CSV = DRIVE_R1_PSEUDO / "pseudo_labels.csv"
    if R1_PSEUDO_CSV.exists():
        print(f"  Fold {FOLD_K}: R1 pseudo CSV exists, loading")
        pseudo_df_r1 = pd.read_csv(R1_PSEUDO_CSV)
    else:
        print(f"\n  --- Generating R1 pseudo (TTA, fold {FOLD_K}) ---")
        # Load best R1 ckpt
        ckpt_r1 = torch.load(str(LOCAL_R1 / "ckpt_best_ns22.pth"), map_location="cpu", weights_only=False)
        model_for_pseudo = BirdSEDModel().to(device)
        model_for_pseudo.load_state_dict(ckpt_r1["model_state"], strict=False)
        model_for_pseudo.eval()
        model_for_pseudo = model_for_pseudo.to(memory_format=torch.channels_last)
        mel_tf_pseudo = MelSpecTransform().to(device)
        ts_files = sorted(_glob.glob(str(TS_DIR / "*.ogg")))
        pseudo_df_r1 = generate_pseudo_with_tta(model_for_pseudo, mel_tf_pseudo, ts_files)
        pseudo_df_r1.to_csv(R1_PSEUDO_CSV, index=False)
        print(f"  R1 pseudo saved: {R1_PSEUDO_CSV} ({R1_PSEUDO_CSV.stat().st_size/1e6:.1f}MB)")
        del model_for_pseudo, mel_tf_pseudo
        gc.collect(); torch.cuda.empty_cache()

    # Prepare pseudo for R2
    Y_PSEUDO_R1 = pseudo_df_r1[PRIMARY_LABELS].values.astype(np.float32)
    pseudo_meta_r1 = pseudo_df_r1[["filename", "start_sec"]].copy().reset_index(drop=True)
    pseudo_meta_r1["filename"] = pseudo_meta_r1["filename"].astype(str)
    pseudo_meta_r1["start_sec"] = pseudo_meta_r1["start_sec"].astype(float)

    # ===== R2 train =====
    r2_best_ns22, r2_best_macro, _ = train_one_round(
        fold_k=FOLD_K, round_name="r2", n_epochs=N_TOTAL_EPOCHS_R2,
        shares=SHARES_R2,
        pseudo_meta_df=pseudo_meta_r1, Y_pseudo=Y_PSEUDO_R1,
        out_dir=LOCAL_R2)
    mirror_dir_to_drive(LOCAL_R2, DRIVE_R2_DIR)

    # ===== R2 → Pseudo (optional, R3 candidate, save for blend) =====
    R2_PSEUDO_CSV = DRIVE_R2_PSEUDO / "pseudo_labels.csv"
    if not R2_PSEUDO_CSV.exists():
        print(f"\n  --- Generating R2 pseudo (TTA, fold {FOLD_K}) ---")
        ckpt_r2 = torch.load(str(LOCAL_R2 / "ckpt_best_ns22.pth"), map_location="cpu", weights_only=False)
        model_for_pseudo = BirdSEDModel().to(device)
        model_for_pseudo.load_state_dict(ckpt_r2["model_state"], strict=False)
        model_for_pseudo.eval()
        model_for_pseudo = model_for_pseudo.to(memory_format=torch.channels_last)
        mel_tf_pseudo = MelSpecTransform().to(device)
        ts_files = sorted(_glob.glob(str(TS_DIR / "*.ogg")))
        pseudo_df_r2 = generate_pseudo_with_tta(model_for_pseudo, mel_tf_pseudo, ts_files)
        pseudo_df_r2.to_csv(R2_PSEUDO_CSV, index=False)
        print(f"  R2 pseudo saved: {R2_PSEUDO_CSV} ({R2_PSEUDO_CSV.stat().st_size/1e6:.1f}MB)")
        del model_for_pseudo, mel_tf_pseudo
        gc.collect(); torch.cuda.empty_cache()

    fold_elapsed = time.time() - fold_t0
    fold_results[FOLD_K] = {
        "r1_best_ns22": r1_best_ns22, "r1_best_macro": r1_best_macro,
        "r2_best_ns22": r2_best_ns22, "r2_best_macro": r2_best_macro,
        "elapsed_min": round(fold_elapsed / 60, 1),
    }

    print(f"\n  >>> Fold {FOLD_K} DONE in {fold_elapsed/60:.1f}min: "
          f"R1={r1_best_ns22:.4f} R2={r2_best_ns22:.4f} <<<\n")

    # Clean local /content for next fold (Drive has everything)
    try:
        shutil.rmtree(LOCAL_R1, ignore_errors=True)
        shutil.rmtree(LOCAL_R2, ignore_errors=True)
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()

print(f"\n{'='*60}\n5-fold loop complete\n{'='*60}")
for k, v in sorted(fold_results.items()):
    print(f"  Fold {k}: {v}")
""", "fold_loop"))

# =============================================================================
# Cell 12: Upload all folds to Kaggle Dataset
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 12: Upload all fold ckpts to Kaggle Dataset (exp020-weights-5fold)
# ============================================================
import tempfile

# ★ Re-establish Kaggle auth (long session で token 期限切れ対策)
KJ_CANDIDATES = [DRIVE_INPUT_DIR / "kaggle.json",
                  Path("/content/drive/MyDrive/kaggle.json")]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
if KJ is not None:
    KAGGLE_CFG = Path.home() / ".kaggle"
    KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
    os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
    creds = json.loads(KJ.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print(f"kaggle re-authenticated for upload")

USER  = "maekeso"
SLUG  = "birdclef2026-exp020-weights-5fold"
TITLE = "birdclef2026 exp020 weights 5fold"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_staged = 0
    for fold_k in FOLDS:
        DRIVE_R1_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r1"
        DRIVE_R2_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r2"
        for src_dir, prefix in [(DRIVE_R1_DIR, f"r1_fold{fold_k}_"),
                                  (DRIVE_R2_DIR, f"r2_fold{fold_k}_")]:
            for fn in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "history.json"]:
                src = src_dir / fn
                if src.exists():
                    dst = td / f"{prefix}{fn}"
                    shutil.copy2(str(src), str(dst))
                    n_staged += 1
                    print(f"  staged: {prefix}{fn}  ({src.stat().st_size/1e6:.1f}MB)")
    print(f"\nStaged {n_staged} files")

    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary_lines = [f"Fold {k}: r1={v.get('r1_best_ns22','?')} r2={v.get('r2_best_ns22','?')}"
                      for k, v in sorted(fold_results.items())]
    version_notes = "exp020 5-fold | " + " ; ".join(summary_lines)

    try:
        api.dataset_list_files(f"{USER}/{SLUG}")
        dataset_exists = True
        print(f"\nDataset exists → version_up")
    except Exception:
        dataset_exists = False
        print(f"\nDataset not found → create_new")

    t0 = time.time()
    uploaded_ok = False
    try:
        if dataset_exists:
            api.dataset_create_version(folder=str(td),
                                        version_notes=version_notes[:498],
                                        dir_mode="zip", quiet=False)
            print(f"\nOK new version uploaded ({time.time()-t0:.1f}s)")
            uploaded_ok = True
        else:
            api.dataset_create_new(folder=str(td), public=False,
                                    dir_mode="zip", quiet=False)
            print(f"\nOK new dataset created ({time.time()-t0:.1f}s)")
            uploaded_ok = True
    except Exception as e:
        print(f"\n[UPLOAD ERROR] {type(e).__name__}: {str(e)[:400]}")
        print(f"Files staged in {td}, will retry manually if needed")

    # ★ Verify upload (no silent fail like past exp014/017)
    if uploaded_ok:
        try:
            files_check = api.dataset_list_files(f"{USER}/{SLUG}").files
            print(f"\n✓ Verified: Dataset has {len(files_check)} files")
            for f in files_check[:5]:
                print(f"   {f.name}")
        except Exception as e:
            print(f"\n[VERIFY FAIL] {type(e).__name__}: {str(e)[:200]}")
            print(f"Dataset may not be fully indexed yet, check Kaggle UI manually")

    print(f"\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
""", "upload"))

# =============================================================================
# Cell 13: Summary
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 13: Session summary
# ============================================================
total_time = time.time() - SESSION_START
print(f"\n{'='*60}")
print(f"exp020 5-fold session summary")
print(f"{'='*60}")
print(f"  Backbone:       {BACKBONE}")
print(f"  Folds trained:  {[k for k in FOLDS if k in fold_results]}")
print(f"  Total time:     {total_time/60:.1f} min ({total_time/3600:.2f}h)")
print()
for k, v in sorted(fold_results.items()):
    print(f"  Fold {k}: {v}")
print(f"\n  Drive output:   {DRIVE_EXP_DIR}")
print(f"  Kaggle Dataset: https://www.kaggle.com/datasets/maekeso/birdclef2026-exp020-weights-5fold")
""", "summary"))

# =============================================================================
# Cell 14: Terminate runtime
# =============================================================================
cells.append(code_cell(r"""# ============================================================
# Cell 14: Terminate Colab runtime — save Pro+ compute units
# ============================================================
print("All exp020 phases complete. Terminating Colab runtime in 5s...")
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

out_path = HERE / "nb_train_5fold_tta.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
