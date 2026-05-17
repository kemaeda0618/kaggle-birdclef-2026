"""Generate exp022 R1 5-fold NB: eca_nfnet_l1, per-step Perch (no cache), TTA pseudo.

Design:
- exp020 (l0) 派生、backbone のみ l1 (~41M params、1.7x l0)
- R1 ONLY (R2 は別 NB で Perch cache を使う)
- per-step Perch ONNX 推論 (Focal の continuous random crop 維持)
- 5-fold loop with Drive resume per fold
- R1 pseudo generation with delta-shift TTA (5-view) per fold, saved to Drive
- WARMUP_EPOCHS=**4** (l1 で l0 比 +1 ep、early LR ramp 緩和)
- grad_clip=1.0 (timm 既定で済、明示維持)

Estimated time (Blackwell 96GB, 5-fold):
- Per fold: R1 ~75 min + pseudo+TTA ~15 min ≈ 90 min (l1 で l0 比 ~1.7x)
- 5-fold total: ~7.5h + DL 15 min ≈ ~8h

Run: python _gen_nb_train_r1_5fold.py  ->  writes nb_train_r1_5fold.ipynb
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

cells.append(md_cell(r"""# exp022 R1 5-fold (eca_nfnet_l1, no cache, per-step Perch, TTA pseudo)

**Goal:** exp020 (l0) を l1 (~41M params、1.7x l0) で 5-fold 化、R1 ckpt + R1 pseudo CSV を Drive に保存。
ensemble に投入して exp020 5-fold + Tucker + NB4 等の blend で +0.003-0.007 上乗せ狙い。
R2 学習は別 NB (`nb_train_r2_5fold.ipynb`、Perch cache 利用)。

## Key 設計 (vs exp020)
- **BACKBONE = `eca_nfnet_l1`** (l0 → l1)
- **WARMUP_EPOCHS = 4** (l0 の 3 から +1 ep、l1 大規模化に伴う発散リスク緩和)
- LR=3e-4、BATCH=192、N_TOTAL_EPOCHS=25 は l0 と同 (NFNet 無 sqrt scaling rule)
- Perch cache 不使用 (per-step ONNX、Focal の連続 random crop 維持)
- 5-fold loop、Drive resume per fold
- grad_clip=1.0 (l1 で gradient explode 保険、l0 と同値)

## TTA (R1 pseudo 生成時)
delta-shift **5 view** (0s, ±0.5s, ±1.0s) → Gaussian smooth (σ=0.65) → Power Transform (γ=1.2)
柔軟性確保のため **per-view raw CSV も保存** (後から重み付き平均 / view 数調整可)

## Drive 出力 (各 fold)
- `output/exp022/fold{k}/r1/ckpt_*.pth` (R1 ckpt)
- `output/exp022/fold{k}/r1-pseudo/pseudo_view{0..4}_raw.csv` (各 view raw probs)
- `output/exp022/fold{k}/r1-pseudo/pseudo_views_meta.json` (shift メタ)
- `output/exp022/fold{k}/r1-pseudo/pseudo_labels.csv` (5 view 平均+smooth+PT、R2 入力用)

## 想定: ~8h on Blackwell 96GB (R1 train ~6h + pseudo+TTA 5 view ~1.5h + DL + upload)
""", "hdr"))

cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup
# ============================================================
!pip install -q timm onnxruntime-gpu librosa soundfile scipy

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, json, shutil, time, subprocess
from pathlib import Path

DRIVE_INPUT_DIR  = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp022"
DRIVE_EXP_DIR.mkdir(parents=True, exist_ok=True)
assert DRIVE_INPUT_DIR.exists()
print(f"Drive input:  {DRIVE_INPUT_DIR}")
print(f"Drive exp022: {DRIVE_EXP_DIR}")

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
# Cell 2: Data DL — competition + Perch v2 ONNX (cache 不使用)
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
    assert zips
    zip_path = zips[0]
    print(f"\n  Extracting...")
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
else:
    print("Competition data already present locally")

n_ta = sum(1 for _ in TA_DIR.rglob("*.ogg")) if TA_DIR.exists() else 0
n_ts = sum(1 for _ in TS_DIR.glob("*.ogg")) if TS_DIR.exists() else 0
print(f"\n  train_audio: {n_ta}, train_soundscapes: {n_ts}")

comp = LOCAL_DATA / "competition"
comp.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
    src_in_root = LOCAL_DATA / fn
    dst = comp / fn
    if src_in_root.exists() and not dst.exists():
        shutil.copy2(str(src_in_root), str(dst))

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
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

BASE = LOCAL_DATA / "competition"
TA_DIR = LOCAL_DATA / "train_audio"
TS_DIR = LOCAL_DATA / "train_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
LABELS_PATH = BASE / "train_soundscapes_labels.csv"
PERCH_PATH = LOCAL_DATA / "perch-onnx" / "perch_v2_no_dft.onnx"

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

BACKBONE = "eca_nfnet_l1"   # ★ exp022: l0 → l1、~41M params、Tucker 公開比 1.7x
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536
ALPHA_DISTILL = 1.0

N_FOLDS = 5
FOLDS = [0, 1, 2, 3, 4]

# ★ NFNet rule: LR=3e-4 据え置き (l0 と同)、WARMUP=4 (l1 大規模化に伴い +1 ep)
N_TOTAL_EPOCHS = 25
BATCH = 192
LR = 3e-4
MIN_LR = 1e-6
WD = 1e-4
WARMUP_EPOCHS = 4           # ★ exp022: 3 → 4、l1 で early LR ramp 発散リスク緩和

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

SHARES_R1 = {"focal": 0.85, "labeled_sc": 0.15}
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "labeled_sc": 1.0}

NUM_WORKERS = 12
PERSISTENT_WORKERS = True

TTA_SHIFTS_SAMPLES = [0, SR // 2, -SR // 2, SR, -SR]   # 5 view: 0s, ±0.5s, ±1.0s (広範囲)
GAUSS_SIGMA  = 0.65
POWER_GAMMA  = 1.2
N_WINDOWS = 12
CHUNK_N   = TRAIN_SAMPLES

SESSION_START = time.time()
MAX_RUNTIME_SEC = 22.0 * 3600

print(f"Backbone: {BACKBONE} | Folds: {FOLDS}")
print(f"R1 epochs: {N_TOTAL_EPOCHS} | warmup: {WARMUP_EPOCHS} | batch: {BATCH} | LR: {LR}")
""", "config"))

cells.append(code_cell(r"""# ============================================================
# Cell 4: Load CSVs + fold split
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
    print(f"Labeled SS: {len(sc_meta)} windows")
else:
    sc_meta = pd.DataFrame(columns=["filename", "start_sec", "site", "fold"])
    Y_SC = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    non_s22_mask_sc = np.zeros(0, dtype=bool)

print("OK CSVs loaded")
""", "load_data"))

cells.append(code_cell(r"""# ============================================================
# Cell 5: Model — Mel + SpecAugment + PerchTeacher (per-step ONNX) + BirdSEDModel
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
        self._embed_out_name = None
        for o in self.session.get_outputs():
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self._embed_out_name = o.name
                break
        if self._embed_out_name is None:
            self._embed_out_name = self.session.get_outputs()[0].name
        print(f"PerchTeacher: embed_out='{self._embed_out_name}', providers={self.session.get_providers()}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run([self._embed_out_name], {self.input_name: wav_np})
        return torch.from_numpy(results[0]).float()


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

print("OK model + PerchTeacher (per-step) ready")
""", "model"))

cells.append(code_cell(r"""# ============================================================
# Cell 6: Datasets — FocalDS (continuous random crop) + LabeledSCDS
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
        avail = full[start:start + n_samples_target] if start < len(full) else np.array([])
        out[:len(avail)] = avail
        return out
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
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
    '''Continuous random crop (exp017 R1 v2 path).'''
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

print("OK datasets (FocalDS continuous crop)")
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

cells.append(code_cell(r"""# ============================================================
# Cell 8: Pseudo gen with TTA (3 view), saving each view + averaged
# 柔軟性最大化のため:
#   - raw per-view CSV (no smooth, no PT): pseudo_view{i}_raw.csv
#   - 既存互換の averaged+smoothed+PT CSV: pseudo_labels.csv (R2 入力用)
#   - shift meta: pseudo_views_meta.json
# ============================================================
import glob as _glob

def _file_to_wav60s(path):
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


def generate_pseudo_save_all_views(model, mel_tf, ts_files, save_dir,
                                     tta_shifts=TTA_SHIFTS_SAMPLES,
                                     gauss_sigma=GAUSS_SIGMA,
                                     power_gamma=POWER_GAMMA):
    '''TTA pseudo gen with per-view raw saving for max flexibility.

    Saves to save_dir:
      - pseudo_view{i}_raw.csv  (no smooth, no PT、各 TTA view 単体)
      - pseudo_views_meta.json  (view → shift mapping)
      - pseudo_labels.csv       (averaged + per-file Gauss smooth + Power Transform、R2 input 用)
    Returns: final pseudo DataFrame (pseudo_labels.csv content) for direct use.
    '''
    save_dir.mkdir(parents=True, exist_ok=True)
    view_probs = [[] for _ in tta_shifts]
    all_filenames, all_start_secs, all_end_secs = [], [], []
    t0 = time.time()
    N_FILES = len(ts_files)
    with torch.no_grad():
        for fi, fpath in enumerate(ts_files):
            stem = Path(fpath).stem
            try:
                wav_60s = _file_to_wav60s(fpath)
            except Exception as e:
                print(f"  WARN {stem}: {e}")
                wav_60s = np.zeros(N_WINDOWS * CHUNK_N, dtype=np.float32)
            for vi, shift in enumerate(tta_shifts):
                wav_shifted = np.roll(wav_60s, shift)
                chunks = wav_shifted.reshape(N_WINDOWS, CHUNK_N).astype(np.float32)
                probs = _predict_chunks(model, mel_tf, chunks)
                view_probs[vi].append(probs)
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

    filenames_arr = np.array(all_filenames)
    start_arr = np.array(all_start_secs, dtype=np.float32)
    end_arr = np.array(all_end_secs, dtype=np.float32)

    # ===== Save raw per-view CSVs (no smooth, no PT - max flexibility) =====
    raw_view_arrays = []
    view_meta_list = []
    for vi, shift in enumerate(tta_shifts):
        prob_mat = np.concatenate(view_probs[vi], axis=0).astype(np.float32)
        raw_view_arrays.append(prob_mat)
        shift_s = shift / SR
        view_meta_list.append({"view_id": vi, "shift_samples": int(shift),
                                 "shift_sec": float(shift_s)})
        view_csv = save_dir / f"pseudo_view{vi}_raw.csv"
        df_view = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)
        df_view.insert(0, "filename",  filenames_arr)
        df_view.insert(1, "start_sec", start_arr)
        df_view.insert(2, "end_sec",   end_arr)
        df_view.to_csv(view_csv, index=False)
        print(f"  saved raw view {vi} (shift={shift_s:+.2f}s): "
              f"{view_csv.name} ({view_csv.stat().st_size/1e6:.1f}MB)")

    # Save shift meta
    meta_dict = {
        "views": view_meta_list,
        "tta_count": len(tta_shifts),
        "n_windows": N_WINDOWS,
        "n_files": N_FILES,
        "primary_labels": PRIMARY_LABELS,
        "gauss_sigma_used_for_pseudo_labels": float(gauss_sigma),
        "power_gamma_used_for_pseudo_labels": float(power_gamma),
    }
    (save_dir / "pseudo_views_meta.json").write_text(
        json.dumps(meta_dict, indent=2, default=str), encoding="utf-8")
    print(f"  saved: pseudo_views_meta.json")

    # ===== Build R2-ready pseudo: mean → per-file Gauss smooth → Power Transform =====
    avg_probs = np.mean(raw_view_arrays, axis=0).astype(np.float32)
    n_files_total = len(avg_probs) // N_WINDOWS
    avg_reshaped = avg_probs.reshape(n_files_total, N_WINDOWS, NUM_CLASSES)
    smoothed = gaussian_filter1d(avg_reshaped, sigma=gauss_sigma, axis=1, mode="nearest")
    final_probs = smoothed.reshape(-1, NUM_CLASSES).astype(np.float32)
    final_probs = np.power(final_probs, power_gamma).astype(np.float32)

    final_df = pd.DataFrame(final_probs, columns=PRIMARY_LABELS)
    final_df.insert(0, "filename",  filenames_arr)
    final_df.insert(1, "start_sec", start_arr)
    final_df.insert(2, "end_sec",   end_arr)
    print(f"  built R2-ready averaged pseudo: shape={final_df.shape}, "
          f"sigma={gauss_sigma}, gamma={power_gamma}")
    return final_df


# Backward-compat wrapper: 直接 R2-ready df を返す呼び出し
def generate_pseudo_with_tta(model, mel_tf, ts_files, save_dir=None,
                              tta_shifts=TTA_SHIFTS_SAMPLES,
                              gauss_sigma=GAUSS_SIGMA, power_gamma=POWER_GAMMA):
    '''Wrapper: save_dir 必須 (旧 API 互換ではないので注意).'''
    assert save_dir is not None, "save_dir is required for view-saving pseudo gen"
    return generate_pseudo_save_all_views(model, mel_tf, ts_files, save_dir,
                                            tta_shifts, gauss_sigma, power_gamma)

print("OK pseudo+TTA helper ready (per-view raw CSVs + averaged pseudo_labels.csv)")
""", "pseudo_tta"))

cells.append(code_cell(r"""# ============================================================
# Cell 9: build_active_datasets_r1
# ============================================================
def build_active_datasets_r1(fold_k):
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
    return items

print("OK build_active_datasets_r1 ready")
""", "build_active"))

cells.append(code_cell(r"""# ============================================================
# Cell 10: train_r1(fold_k, out_dir) — per-step Perch ONNX
# ============================================================
def train_r1(fold_k, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n[Fold {fold_k} R1] train ({N_TOTAL_EPOCHS} ep)\n{'='*60}")

    active = build_active_datasets_r1(fold_k)
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
    perch_teacher = PerchTeacher(PERCH_PATH, "cuda" if torch.cuda.is_available() else "cpu")

    best_ns22 = -1.0
    best_macro = -1.0
    history = []
    epoch_times = []

    for epoch in range(N_TOTAL_EPOCHS):
        elapsed = time.time() - SESSION_START
        if epoch_times:
            est_next = max(epoch_times[-3:])
            if elapsed + est_next * 1.3 > MAX_RUNTIME_SEC:
                print(f"  [stop] time budget exhausted before ep {epoch+1}")
                break
        ep_start = time.time()
        model.train()

        smp = MixSamp(SIZES, NAMES, SHARES_R1, BATCH, n_steps_ep, seed=42 + epoch)
        train_loader = DataLoader(
            mds, batch_sampler=smp, collate_fn=collate_m,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
            pin_memory=True,
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )

        el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
        for batch_idx, (wav, lb, wt, mk, sr) in enumerate(train_loader):
            wav = wav.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
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
                # per-step Perch ONNX
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
            "epoch": ep, "fold": fold_k, "round": "r1",
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

    print(f"  [R1 done] best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}")
    del optimizer, scheduler, scaler, mel_transform, spec_augment, perch_teacher
    gc.collect(); torch.cuda.empty_cache()
    return best_ns22, best_macro, model

print("OK train_r1 ready")
""", "train_r1"))

cells.append(code_cell(r"""# ============================================================
# Cell 11: Main 5-fold loop (R1 train + R1 pseudo+TTA per fold)
# ============================================================
def mirror_dir_to_drive(local_dir, drive_dir):
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


fold_results = {}

for FOLD_K in FOLDS:
    print(f"\n{'#'*70}\n# R1 Fold {FOLD_K} / {N_FOLDS-1}\n{'#'*70}")
    DRIVE_FOLD_DIR  = DRIVE_EXP_DIR / f"fold{FOLD_K}"
    DRIVE_R1_DIR    = DRIVE_FOLD_DIR / "r1"
    DRIVE_R1_PSEUDO = DRIVE_FOLD_DIR / "r1-pseudo"
    DRIVE_R1_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_R1_PSEUDO.mkdir(parents=True, exist_ok=True)

    R1_BEST = DRIVE_R1_DIR / "ckpt_best_ns22.pth"
    R1_PSEUDO_CSV = DRIVE_R1_PSEUDO / "pseudo_labels.csv"

    if R1_BEST.exists() and R1_PSEUDO_CSV.exists():
        print(f"  Fold {FOLD_K}: R1 + R1-pseudo already on Drive — skip")
        try:
            st = torch.load(str(R1_BEST), map_location="cpu", weights_only=False)
            fold_results[FOLD_K] = {"r1_best_ns22": st.get("best_ns22", -1),
                                      "r1_best_macro": st.get("best_macro", -1),
                                      "status": "skipped"}
        except Exception:
            fold_results[FOLD_K] = {"status": "skipped (load err)"}
        continue

    LOCAL_R1 = LOCAL_OUT / f"fold{FOLD_K}" / "r1"
    LOCAL_R1.mkdir(parents=True, exist_ok=True)
    fold_t0 = time.time()

    if not R1_BEST.exists():
        r1_best_ns22, r1_best_macro, _ = train_r1(fold_k=FOLD_K, out_dir=LOCAL_R1)
        mirror_dir_to_drive(LOCAL_R1, DRIVE_R1_DIR)
    else:
        print(f"  R1 ckpt exists on Drive, copy local for pseudo gen")
        st_r1 = torch.load(str(R1_BEST), map_location="cpu", weights_only=False)
        r1_best_ns22 = st_r1.get("best_ns22", -1)
        r1_best_macro = st_r1.get("best_macro", -1)
        for f in DRIVE_R1_DIR.glob("ckpt_*.pth"):
            shutil.copy2(str(f), str(LOCAL_R1 / f.name))

    # R1 pseudo gen with TTA — saves per-view raw CSVs + averaged pseudo_labels.csv
    if not R1_PSEUDO_CSV.exists():
        print(f"\n  --- R1 pseudo (TTA + per-view save, fold {FOLD_K}) ---")
        ckpt_r1 = torch.load(str(LOCAL_R1 / "ckpt_best_ns22.pth"),
                              map_location="cpu", weights_only=False)
        model_r1 = BirdSEDModel().to(device)
        model_r1.load_state_dict(ckpt_r1["model_state"], strict=False)
        model_r1.eval()
        model_r1 = model_r1.to(memory_format=torch.channels_last)
        mel_tf_pseudo = MelSpecTransform().to(device)
        ts_files = sorted(_glob.glob(str(TS_DIR / "*.ogg")))
        # save_dir = DRIVE_R1_PSEUDO で per-view raw CSV + meta も保存される
        pseudo_df = generate_pseudo_save_all_views(
            model_r1, mel_tf_pseudo, ts_files, DRIVE_R1_PSEUDO)
        pseudo_df.to_csv(R1_PSEUDO_CSV, index=False)
        print(f"  R1 pseudo saved (averaged for R2): {R1_PSEUDO_CSV} "
              f"({R1_PSEUDO_CSV.stat().st_size/1e6:.1f}MB)")
        print(f"  Per-view raw CSVs also in: {DRIVE_R1_PSEUDO}")
        del model_r1, mel_tf_pseudo
        gc.collect(); torch.cuda.empty_cache()

    fold_elapsed = time.time() - fold_t0
    fold_results[FOLD_K] = {
        "r1_best_ns22": r1_best_ns22,
        "r1_best_macro": r1_best_macro,
        "elapsed_min": round(fold_elapsed / 60, 1),
    }
    print(f"\n  >>> Fold {FOLD_K} R1 DONE in {fold_elapsed/60:.1f}min: R1={r1_best_ns22:.4f} <<<\n")

    try:
        shutil.rmtree(LOCAL_R1, ignore_errors=True)
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()

print(f"\n{'='*60}\n5-fold R1 + pseudo complete\n{'='*60}")
for k, v in sorted(fold_results.items()):
    print(f"  Fold {k}: {v}")
""", "fold_loop"))

cells.append(code_cell(r"""# ============================================================
# Cell 12: Upload R1 ckpts to Kaggle Dataset (backup)
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
SLUG  = "birdclef2026-exp022-r1-5fold"
TITLE = "birdclef2026 exp022 R1 5fold weights"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_staged = 0
    for fold_k in FOLDS:
        DRIVE_R1_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r1"
        for fn in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "history.json"]:
            src = DRIVE_R1_DIR / fn
            if src.exists():
                dst_name = f"r1_fold{fold_k}_{fn}"
                shutil.copy2(str(src), str(td / dst_name))
                n_staged += 1
    print(f"Staged {n_staged} files")
    meta = {"title": TITLE, "id": f"{USER}/{SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = " ; ".join([f"f{k}={v.get('r1_best_ns22','?')}" for k, v in sorted(fold_results.items())])
    version_notes = f"exp022 R1 5fold (l1) | {summary}"[:498]

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
# Cell 13: Summary
# ============================================================
total_time = time.time() - SESSION_START
print(f"\n{'='*60}")
print(f"exp022 R1 5-fold summary (eca_nfnet_l1)")
print(f"{'='*60}")
print(f"  Backbone: {BACKBONE}")
print(f"  R1 epochs/fold: {N_TOTAL_EPOCHS}")
print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
for k, v in sorted(fold_results.items()):
    print(f"  Fold {k}: {v}")
print(f"\n  Drive: {DRIVE_EXP_DIR}")
print(f"  Next: run nb_train_r2_5fold.ipynb")
""", "summary"))

cells.append(code_cell(r"""# ============================================================
# Cell 14: Terminate Colab runtime
# ============================================================
print("All R1 phases complete. Terminating Colab runtime in 5s...")
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

out_path = HERE / "nb_train_r1_5fold.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
