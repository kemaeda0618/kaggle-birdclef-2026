"""Generate exp029: exp017 R2 single teacher → l1 student (single fold) NB.

Design (Option B: 単 teacher distill 検証):
- Student: eca_nfnet_l1 (新 backbone、capacity 増)
- Single fold (5-fold avg homogenization 回避)
- Teacher pseudo: pseudo_e17.csv (exp017 R2 single fold output、exp028 で生成済)
- Perch cache 利用 (focal augmentation 影響薄、速度優先)
- N_TOTAL_EPOCHS=**20** (Option C 採用、warmup 4/20 = 20% で適正)
- WARMUP=4 (l1 で +1、exp022 同設定)
- SHARES_R2: pseudo_sc=**0.25** (Option C、単 teacher distill 強化、focal 0.65 で補正)

期待 LB: 0.921-0.926 (exp017 R2 = 0.921 から +0.001-0.005)
- l1 backbone capacity 増の効果検証
- 効くなら Multi-teacher R3 (6 teacher) へ展開
- 効かないなら別 axis (BirdNET 等) に転換

Estimated: ~2.5-3h on Blackwell (l1 single fold、20 epoch)

Run: python _gen_nb_train_r3_l1_single.py
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

cells.append(md_cell(r"""# exp029 R3 (l1 single fold, exp017 R2 teacher)

**前提**: `nb_train_r1_5fold.ipynb` 完走済、Drive に R1 ckpt + R1 pseudo CSV あり。

## 構成
- Perch cache 使用 (`maekeso/birdclef2026-perch-emb-cache`)
- Teacher pseudo from Drive (`kaggle/birdclef2026/pseudo_e17.csv`、exp028 で生成済)
- **N_EPOCHS=15** (Plan I、peak ep 4-8 に buffer +7-11、best_ckpt で確実 peak 捕捉)、WARMUP=3、batch=192、LR=3e-4
- NUM_WORKERS=16 (data loading 並列度 +33%)
- 5-fold loop、Drive resume per fold
- R2 pseudo gen は skip
- 最終: Kaggle Dataset `birdclef2026-exp029-l1-single` に R3 ckpt upload

## 想定: ~5.5-6h on Blackwell (Plan I、20 ep 比 -30%)
""", "hdr"))

cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup
# ============================================================
!pip install -q timm librosa soundfile scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

DRIVE_INPUT_DIR  = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp029"
assert DRIVE_INPUT_DIR.exists()

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
    print(f"kaggle.json: {KJ}")

LOCAL_DATA = Path("/content/data")
LOCAL_OUT  = Path("/content/output")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_OUT.mkdir(parents=True, exist_ok=True)
""", "setup"))

cells.append(code_cell(r"""# ============================================================
# Cell 2: Data DL — competition + Perch cache
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
    print(f"\nDownloading birdclef-2026 (~25GB)...")
    t0 = time.time()
    api.competition_download_files("birdclef-2026", path=str(LOCAL_DATA),
                                    force=False, quiet=False)
    print(f"  DL done in {(time.time()-t0)/60:.1f} min")
    zips = list(LOCAL_DATA.glob("birdclef-2026*.zip"))
    zip_path = zips[0]
    t_extract = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        total_bytes = sum(i.file_size for i in infos)
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="extract", mininterval=1.0)
        for info in infos:
            zf.extract(info, LOCAL_DATA)
            pbar.update(info.file_size)
        pbar.close()
    print(f"  extracted in {(time.time()-t_extract)/60:.1f} min")
    zip_path.unlink()

n_ta = sum(1 for _ in TA_DIR.rglob("*.ogg")) if TA_DIR.exists() else 0
n_ts = sum(1 for _ in TS_DIR.glob("*.ogg")) if TS_DIR.exists() else 0
print(f"  train_audio: {n_ta}, train_soundscapes: {n_ts}")

comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
    src_in_root = LOCAL_DATA / fn
    dst = comp / fn
    if src_in_root.exists() and not dst.exists():
        shutil.copy2(str(src_in_root), str(dst))

PERCH_CACHE_DIR = LOCAL_DATA / "perch-cache"
EMB_PATH = PERCH_CACHE_DIR / "emb.npy"
META_PATH = PERCH_CACHE_DIR / "meta.csv"
if not EMB_PATH.exists() or not META_PATH.exists():
    print(f"\nDownloading Perch cache (~1.1GB)...")
    PERCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    api.dataset_download_files("maekeso/birdclef2026-perch-emb-cache",
                                path=str(PERCH_CACHE_DIR), unzip=True, quiet=False)
    print(f"  DL done in {(time.time()-t0)/60:.1f} min")
assert EMB_PATH.exists() and META_PATH.exists()
print(f"  emb.npy: {EMB_PATH.stat().st_size/1e9:.2f}GB")

# ★ exp029: teacher pseudo (pseudo_e17.csv) を Kaggle NB output から DL → Drive へ
TEACHER_PSEUDO_DRIVE = DRIVE_INPUT_DIR / "pseudo_e17.csv"
if not TEACHER_PSEUDO_DRIVE.exists():
    print(f"\nDownloading teacher pseudo (pseudo_e17.csv) from Kaggle NB output...")
    t0 = time.time()
    TEACHER_PSEUDO_TMP = LOCAL_DATA / "pseudo_e17_kaggle"
    TEACHER_PSEUDO_TMP.mkdir(parents=True, exist_ok=True)
    try:
        # Selective DL of pseudo_e17.csv only (avoid huge model output)
        api.kernels_output_download_file("maekeso/birdclef2026-exp028-pseudo-e17",
                                         file_name="pseudo_e17.csv",
                                         path=str(TEACHER_PSEUDO_TMP))
        src = TEACHER_PSEUDO_TMP / "pseudo_e17.csv"
        assert src.exists(), f"DL failed: {src}"
        shutil.copy2(str(src), str(TEACHER_PSEUDO_DRIVE))
        print(f"  DL done in {(time.time()-t0)/60:.1f} min, saved to {TEACHER_PSEUDO_DRIVE}")
    except Exception as e:
        # Fallback: download all output (heavier)
        print(f"  selective DL failed ({e}), trying full output download...")
        api.kernels_output("maekeso/birdclef2026-exp028-pseudo-e17",
                           path=str(TEACHER_PSEUDO_TMP), force=True)
        src = next(TEACHER_PSEUDO_TMP.rglob("pseudo_e17.csv"), None)
        assert src is not None, "pseudo_e17.csv not found in NB output"
        shutil.copy2(str(src), str(TEACHER_PSEUDO_DRIVE))
        print(f"  fallback DL done in {(time.time()-t0)/60:.1f} min")
print(f"  pseudo_e17.csv: {TEACHER_PSEUDO_DRIVE.stat().st_size/1e6:.1f}MB")

print(f"\n=== Total prep time: {(time.time()-T0_total)/60:.1f} min ===")
""", "dl_data"))

cells.append(code_cell(r"""# ============================================================
# Cell 3: Imports + Config
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

NUM_CLASSES = 234
SR = 32000
TRAIN_DURATION = 5
TRAIN_SAMPLES = SR * TRAIN_DURATION
VAL_SAMPLES = TRAIN_SAMPLES
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000

BACKBONE = "eca_nfnet_l1"   # ★ exp029: l0 → l1 (capacity 増、Option B 検証)
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

N_FOLDS = 5               # ★ exp029: 5-fold split で分割するが、train するのは fold 0 のみ
FOLDS = [0]               # single fold 学習 (homogenize 回避)、ただし split は 5-fold で行う

# ★ Plan I: 15 ep + NUM_WORKERS=16 で時短重視
# pseudo distill peak ep 4-8、15 ep で buffer +7-11 ep (十分)
# best_ckpt 自動保存で peak 確実捕捉、val 同等期待
# 5-fold 5.5-6h、CU 余裕で Plan B (multi-teacher) も実行可
N_TOTAL_EPOCHS = 20         # ★ exp029 Option C: 15 → 20、warmup 4/20 = 20% で適正
BATCH = 192
LR = 3e-4
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 4         # ★ exp029: l1 で +1 ep (exp022 同設定)

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

SHARES_R2 = {"focal": 0.65, "labeled_sc": 0.10, "pseudo_sc": 0.25}   # ★ exp029 Option C: pseudo 0.20 → 0.25、distill 強化
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "labeled_sc": 1.0, "pseudo_sc": 0.5}

NUM_WORKERS = 16        # Plan I: 12 → 16、data loading 並列度 +33%
PERSISTENT_WORKERS = True

SESSION_START = time.time()
MAX_RUNTIME_SEC = 22.0 * 3600

print(f"Backbone: {BACKBONE} | Folds: {FOLDS}")
print(f"R2 epochs: {N_TOTAL_EPOCHS} | warmup: {WARMUP_EPOCHS} | batch: {BATCH} | LR: {LR}")
print(f"Sources: {SHARES_R2}")
""", "config"))

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
train_df["exists"] = train_df["filename"].map(lambda fn: (TA_DIR / fn).exists())
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
train_df["original_idx"] = np.arange(len(train_df))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fold

focal_secondary_labels = {}
for idx, row in train_df.iterrows():
    sec = row.get("secondary_labels", "")
    if pd.isna(sec) or sec in ("", "[]"): continue
    try:
        sec_list = eval(sec) if isinstance(sec, str) else []
    except Exception: continue
    valid = [s for s in sec_list if s in LABEL2IDX]
    if valid:
        focal_secondary_labels[int(row["original_idx"])] = valid

counts = train_df["primary_label"].value_counts()
rare_species = counts[counts < MIN_SAMPLE].index.tolist()
extra_rows = []
for sp in rare_species:
    sp_rows = train_df[train_df["primary_label"] == sp]
    n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
    for _ in range(n_copies):
        extra_rows.append(sp_rows)
if extra_rows:
    train_df = pd.concat([train_df] + extra_rows, ignore_index=True)

if LABELS_PATH.exists():
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    if sc_labels_raw["start"].dtype == object:
        sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    else:
        sc_labels_raw["start_sec"] = sc_labels_raw["start"].astype(int)
    sc_meta = sc_labels_raw[["filename", "start_sec"]].drop_duplicates().reset_index(drop=True)
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
else:
    sc_meta = pd.DataFrame(columns=["filename", "start_sec", "site", "fold"])
    Y_SC = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    non_s22_mask_sc = np.zeros(0, dtype=bool)

# Perch cache lookup
print(f"\nLoading Perch cache meta.csv...")
meta_df = pd.read_csv(META_PATH)
focal_meta = meta_df[meta_df["source"] == "focal"].reset_index(drop=True)
focal_chunk_lookup = {}
for fn, sub in focal_meta.groupby("filename"):
    focal_chunk_lookup[fn] = sub[["chunk_idx", "row_idx"]].values.astype(np.int32)

ss_meta = meta_df[meta_df["source"] == "ss"].reset_index(drop=True)
ss_lookup = {}
for fn, sub in ss_meta.groupby("filename"):
    base = fn.rsplit(".ogg", 1)[0] if fn.endswith(".ogg") else fn
    for ci, ri in sub[["chunk_idx", "row_idx"]].values:
        ss_lookup[(base, int(ci))] = int(ri)
        ss_lookup[(fn, int(ci))]   = int(ri)
print(f"  focal: {len(focal_chunk_lookup)} files, ss: {len(ss_lookup)} entries")

EMB_MMAP = np.load(str(EMB_PATH), mmap_mode="r")
print(f"  emb shape: {EMB_MMAP.shape}")

print("OK CSVs + Perch cache loaded")
""", "load_data"))

cells.append(code_cell(r"""# ============================================================
# Cell 5: Model
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

print("OK model defs")
""", "model"))

cells.append(code_cell(r"""# ============================================================
# Cell 6: Datasets — cache lookup
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
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            torch.stack([b[4] for b in batch]),
            [b[5] for b in batch])

def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)

print("OK datasets")
""", "dataset"))

cells.append(code_cell(r"""# ============================================================
# Cell 7: Eval helpers
# ============================================================
def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col): continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError: continue
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

print("OK eval helpers")
""", "eval"))

cells.append(code_cell(r"""# ============================================================
# Cell 8: train_r2 function
# ============================================================
def train_r2(fold_k, pseudo_meta_df, Y_pseudo, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n[Fold {fold_k} R2] train ({N_TOTAL_EPOCHS} ep)\n{'='*60}")

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
    pds = PseudoScDS(pseudo_meta_df, Y_pseudo, TS_DIR, aug=True)
    items.append(("pseudo_sc", pds, len(pds)))

    NAMES, DATASETS, SIZES = zip(*items)
    NAMES, DATASETS, SIZES = list(NAMES), list(DATASETS), list(SIZES)
    print(f"  Streams: {dict(zip(NAMES, SIZES))}")

    mds = ConcatDataset(DATASETS)
    n_steps_ep = max(100, int(sum(SIZES) / BATCH))
    print(f"  steps/epoch: {n_steps_ep}")

    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = GradScaler()
    warmup_steps = n_steps_ep * WARMUP_EPOCHS
    total_steps  = n_steps_ep * N_TOTAL_EPOCHS
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=MIN_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    if len(sc_meta) > 0:
        vm = sc_meta["fold"].values == fold_k
        val_sc_df = sc_meta[vm].reset_index(drop=True)
        Y_val = Y_SC[vm]
        ns22_val = non_s22_mask_sc[vm]
        val_wavs = _load_val_waveforms(val_sc_df)
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

    for epoch in range(N_TOTAL_EPOCHS):
        elapsed = time.time() - SESSION_START
        if epoch_times:
            est_next = max(epoch_times[-3:])
            if elapsed + est_next * 1.3 > MAX_RUNTIME_SEC:
                print(f"  [stop] time budget"); break
        ep_start = time.time()
        model.train()

        smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch)
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
                clip_logits, framewise, distill_emb = model(
                    mel, return_framewise=True, return_distill=True)
                frame_max_logits = framewise.max(dim=1).values
                bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                bce = 0.5 * bce_clip + 0.5 * bce_frame
                ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                cls_loss = (ps * sw).mean()
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss

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

        state_to_save = {
            "epoch": ep, "fold": fold_k, "round": "r2",
            "n_total_epochs": N_TOTAL_EPOCHS,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_ns22": best_ns22, "best_macro": best_macro,
        }
        torch.save(state_to_save, out_dir / "ckpt_latest.pth")
        if (not math.isnan(val_ns22)) and val_ns22 > best_ns22:
            best_ns22 = val_ns22
            state_to_save["best_ns22"] = best_ns22
            torch.save(state_to_save, out_dir / "ckpt_best_ns22.pth")
        if (not math.isnan(val_macro)) and val_macro > best_macro:
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
        print(f"\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} "
              f"cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} "
              f"({ep_elapsed:.0f}s, session {total_elapsed/60:.1f}min) ===\n")
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  [R2 done] best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}")
    del optimizer, scheduler, scaler, mel_transform, spec_augment
    gc.collect(); torch.cuda.empty_cache()
    return best_ns22, best_macro

print("OK train_r2 ready")
""", "train_r2"))

cells.append(code_cell(r"""# ============================================================
# Cell 9: Main 5-fold loop
# ============================================================
def mirror_dir_to_drive(local_dir, drive_dir):
    drive_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(local_dir.glob("*")):
        if not f.is_file(): continue
        if f.suffix not in {".pth", ".json"}: continue
        try:
            shutil.copy2(str(f), str(drive_dir / f.name))
            n += 1
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")
    return n


fold_results = {}

for FOLD_K in FOLDS:
    print(f"\n{'#'*70}\n# exp029 R3 (l1 student, e17 teacher) Fold {FOLD_K} (of {N_FOLDS}-fold split, single fold training)\n{'#'*70}")
    DRIVE_FOLD_DIR  = DRIVE_EXP_DIR / f"fold{FOLD_K}"
    # ★ exp029: teacher pseudo は exp017 R2 (pseudo_e17.csv) を Drive 上に置く前提
    # pseudo_e17.csv は exp028 で生成済 → Kaggle Notebook output から download → Drive へ
    # Drive path: kaggle/birdclef2026/pseudo_e17.csv (固定)
    DRIVE_TEACHER_PSEUDO = DRIVE_INPUT_DIR / "pseudo_e17.csv"
    DRIVE_R2_DIR    = DRIVE_FOLD_DIR / "r3"
    DRIVE_R2_DIR.mkdir(parents=True, exist_ok=True)

    if not DRIVE_TEACHER_PSEUDO.exists():
        print(f"  [SKIP] teacher pseudo missing: {DRIVE_TEACHER_PSEUDO}")
        print(f"  Download from Kaggle: maekeso/birdclef2026-exp028-pseudo-e17 (NB output)")
        print(f"  Or use kaggle api: api.kernels_output('maekeso/birdclef2026-exp028-pseudo-e17', path='/tmp')")
        fold_results[FOLD_K] = {"status": "skipped (no teacher pseudo)"}
        continue

    R2_BEST = DRIVE_R2_DIR / "ckpt_best_ns22.pth"
    if R2_BEST.exists():
        print(f"  Fold {FOLD_K}: R3 ckpt already on Drive — skip")
        try:
            st = torch.load(str(R2_BEST), map_location="cpu", weights_only=False)
            fold_results[FOLD_K] = {"r2_best_ns22": st.get("best_ns22", -1),
                                      "r2_best_macro": st.get("best_macro", -1),
                                      "status": "skipped"}
        except Exception:
            fold_results[FOLD_K] = {"status": "skipped (load err)"}
        continue

    LOCAL_R2 = LOCAL_OUT / f"fold{FOLD_K}" / "r3"
    LOCAL_R2.mkdir(parents=True, exist_ok=True)
    fold_t0 = time.time()

    print(f"  Loading teacher pseudo (exp017 R2): {DRIVE_TEACHER_PSEUDO}")
    pseudo_df_r1 = pd.read_csv(DRIVE_TEACHER_PSEUDO)
    print(f"    {len(pseudo_df_r1)} rows")
    Y_PSEUDO = pseudo_df_r1[PRIMARY_LABELS].values.astype(np.float32)
    pseudo_meta = pseudo_df_r1[["filename", "start_sec"]].copy().reset_index(drop=True)
    pseudo_meta["filename"] = pseudo_meta["filename"].astype(str)
    pseudo_meta["start_sec"] = pseudo_meta["start_sec"].astype(float)

    r2_best_ns22, r2_best_macro = train_r2(
        fold_k=FOLD_K, pseudo_meta_df=pseudo_meta, Y_pseudo=Y_PSEUDO,
        out_dir=LOCAL_R2)
    mirror_dir_to_drive(LOCAL_R2, DRIVE_R2_DIR)

    fold_elapsed = time.time() - fold_t0
    fold_results[FOLD_K] = {
        "r2_best_ns22": r2_best_ns22,
        "r2_best_macro": r2_best_macro,
        "elapsed_min": round(fold_elapsed / 60, 1),
    }
    print(f"\n  >>> Fold {FOLD_K} R2 DONE in {fold_elapsed/60:.1f}min: R2={r2_best_ns22:.4f} <<<\n")

    try:
        shutil.rmtree(LOCAL_R2, ignore_errors=True)
    except Exception: pass
    gc.collect(); torch.cuda.empty_cache()

print(f"\n{'='*60}\n5-fold R2 complete\n{'='*60}")
for k, v in sorted(fold_results.items()):
    print(f"  Fold {k}: {v}")
""", "fold_loop"))

cells.append(code_cell(r"""# ============================================================
# Cell 10: Upload R1 + R2 ckpts to Kaggle Dataset
# ============================================================
import tempfile

KJ = next((p for p in [DRIVE_INPUT_DIR / "kaggle.json",
                        Path("/content/drive/MyDrive/kaggle.json")] if p.exists()), None)
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
print("kaggle re-auth OK")

USER  = "maekeso"
SLUG  = "birdclef2026-exp029-l1-single"
TITLE = "birdclef2026 exp029 l1 single (R3 from e17)"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_staged = 0
    for fold_k in FOLDS:
        # ★ exp029: R3 ckpt のみ upload (R1/R2 は別 exp で管理)
        DRIVE_R3_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r3"
        for fn in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "history.json"]:
            src = DRIVE_R3_DIR / fn
            if src.exists():
                dst = td / f"r3_fold{fold_k}_{fn}"
                shutil.copy2(str(src), str(dst))
                n_staged += 1
    print(f"Staged {n_staged} files")
    meta = {"title": TITLE, "id": f"{USER}/{SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = " ; ".join([f"f{k}_r3={v.get('r2_best_ns22','?')}" for k, v in sorted(fold_results.items())])
    version_notes = f"exp029 l1 single (R3 from e17) | {summary}"[:498]

    try:
        api.dataset_list_files(f"{USER}/{SLUG}")
        dataset_exists = True
    except Exception:
        dataset_exists = False

    uploaded_ok = False
    try:
        if dataset_exists:
            api.dataset_create_version(folder=str(td), version_notes=version_notes,
                                        dir_mode="zip", quiet=False)
            print("OK new version uploaded")
            uploaded_ok = True
        else:
            api.dataset_create_new(folder=str(td), public=False,
                                    dir_mode="zip", quiet=False)
            print("OK new dataset created")
            uploaded_ok = True
    except Exception as e:
        print(f"[UPLOAD ERROR] {type(e).__name__}: {str(e)[:400]}")

    if uploaded_ok:
        try:
            files_check = api.dataset_list_files(f"{USER}/{SLUG}").files
            print(f"✓ Verified: {len(files_check)} files")
        except Exception as e:
            print(f"[VERIFY FAIL] {str(e)[:200]}")
    print(f"\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
""", "upload"))

cells.append(code_cell(r"""# ============================================================
# Cell 11: Summary
# ============================================================
total_time = time.time() - SESSION_START
print(f"\n{'='*60}")
print(f"exp029 l1 single fold R3 summary")
print(f"{'='*60}")
print(f"  Backbone: {BACKBONE}")
print(f"  R2 epochs/fold: {N_TOTAL_EPOCHS}")
print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
for k, v in sorted(fold_results.items()):
    print(f"  Fold {k}: {v}")
print(f"\n  Kaggle Dataset: https://www.kaggle.com/datasets/maekeso/birdclef2026-exp029-l1-single")
""", "summary"))

cells.append(code_cell(r"""# ============================================================
# Cell 12: Terminate Colab runtime
# ============================================================
print("All R2 phases complete. Terminating Colab runtime in 5s...")
import time as _t
_t.sleep(5)
from google.colab import runtime
runtime.unassign()
""", "terminate"))

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

out_path = HERE / "nb_train_r3_l1_single.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
