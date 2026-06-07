"""Build exp080 Colab training NB.

Differences from Kaggle version:
  - Drive mount + Kaggle API DL at start
  - Paths: /content/data/ (local SSD) instead of /kaggle/input/
  - Larger batch (L4 has more VRAM)
  - num_workers=8 (Colab has more CPU)
  - Drive output mirror at end

Drive root: /content/drive/MyDrive/kaggle/birdclef2026/
Output:     /content/drive/MyDrive/kaggle/birdclef2026/output/exp080/
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_train_colab.ipynb")


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md_cell("""# exp080 Colab: 1-stage Combined Training (b0)

**Colab G4 (96GB VRAM) 想定**

Differences from Kaggle:
  - Drive mount (input/output)
  - Kaggle API DL for all datasets (local SSD)
  - **Batch 256** (96GB VRAM 余裕、step 数 1/8)
  - **num_workers=16**
  - **LR 8.5e-4** (sqrt(256/32) scale from base 3e-4)
  - 20 epoch (12.5h budget 内、~3-4h 完走見込)
"""))

# Cell 1: Mount + setup
cells.append(code_cell("""# Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# pip install (Colab has most preinstalled)
!pip install -q librosa timm onnxruntime

import sys, os
print(f"Python: {sys.version[:50]}")
"""))

# Cell 2: Kaggle API setup + data download
cells.append(code_cell("""# Setup Kaggle API + DL datasets
import os, json, shutil, zipfile
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026")
LOCAL_DATA = Path("/content/data")
LOCAL_OUT = Path("/content/output")
DRIVE_OUT = DRIVE_ROOT / "output" / "exp080"
LOCAL_DATA.mkdir(exist_ok=True, parents=True)
LOCAL_OUT.mkdir(exist_ok=True, parents=True)
DRIVE_OUT.mkdir(exist_ok=True, parents=True)

# Kaggle credentials
KAGGLE_JSON_DRIVE = DRIVE_ROOT / "kaggle.json"
if KAGGLE_JSON_DRIVE.exists():
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.copy(str(KAGGLE_JSON_DRIVE), os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    # Also extract KGAT token if present
    with open(KAGGLE_JSON_DRIVE) as f:
        kj = json.load(f)
    if "key" in kj and kj["key"].startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = kj["key"]
    print("kaggle.json loaded from Drive")
else:
    print(f"WARN: {KAGGLE_JSON_DRIVE} not found. Manual kaggle.json upload needed.")

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print(f"Kaggle API authenticated")
"""))

cells.append(code_cell("""# DL datasets to /content/data
import zipfile, time
from tqdm.auto import tqdm

def dl_competition(name, dest):
    if (dest / "train.csv").exists() and (dest / "train_audio").exists():
        print(f"  {name} already cached at {dest}")
        return
    dest.mkdir(exist_ok=True, parents=True)
    print(f"  DL competition {name} ...")
    t0 = time.time()
    api.competition_download_files(name, path=str(dest), quiet=False)
    # unzip
    zips = list(dest.glob("*.zip"))
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
        zp.unlink()
    print(f"  {name}: {(time.time()-t0)/60:.1f} min")

def dl_dataset(slug, dest):
    if dest.exists() and any(dest.iterdir()):
        print(f"  {slug} already cached at {dest}")
        return
    dest.mkdir(exist_ok=True, parents=True)
    print(f"  DL dataset {slug} ...")
    t0 = time.time()
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    print(f"  {slug}: {(time.time()-t0)/60:.1f} min")

def dl_kernel_output(slug, dest):
    if dest.exists() and any(dest.iterdir()):
        print(f"  {slug} already cached at {dest}")
        return
    dest.mkdir(exist_ok=True, parents=True)
    print(f"  DL kernel output {slug} ...")
    t0 = time.time()
    try:
        api.kernels_output(slug, path=str(dest), force=True, quiet=False)
    except UnicodeEncodeError:
        pass
    print(f"  {slug}: {(time.time()-t0)/60:.1f} min")

# 1. BC2026 competition
dl_competition("birdclef-2026", LOCAL_DATA / "birdclef-2026")

# 2. Tucker SED ONNX
dl_dataset("tuckerarrants/bc2026-distilled-sed-public", LOCAL_DATA / "tucker_sed")

# 3. Babych BC25 1st place
dl_dataset("nikitababich/birdclef2025-1st-place-ensemble", LOCAL_DATA / "babych_b25")

# 4. XC Part 3 (dataset)
dl_dataset("maekeso/birdclef2026-xc-api-dl-part3", LOCAL_DATA / "xc_part3")

# 5. exp080a (adaptive blend pseudo, kernel output)
dl_kernel_output("maekeso/birdclef2026-exp080a-adaptive-blend", LOCAL_DATA / "exp080a")

# 6. exp080b (XC pseudo, kernel output)
dl_kernel_output("maekeso/birdclef2026-exp080b-xc-pseudo-tucker", LOCAL_DATA / "exp080b")

# 7. XC Part 1 (kernel output)
dl_kernel_output("maekeso/birdclef2026-exp047-xc-api-dl-part1", LOCAL_DATA / "xc_part1")

# 8. XC Part 2 (kernel output)
dl_kernel_output("maekeso/birdclef2026-exp047-xc-api-dl-part2", LOCAL_DATA / "xc_part2")

print(f"\\nAll downloads done at {LOCAL_DATA}")
"""))

cells.append(code_cell("""# Imports
import os, time, json, gc, math, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import timm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tqdm.auto as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, torch {torch.__version__}, timm {timm.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
START = time.time()
"""))

# CFG (Colab spec, larger batch + more workers)
cells.append(code_cell("""# CFG
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC

# Mel (Tucker spec)
N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 20
F_MAX = 16000
TOP_DB = 80

# Train (Colab G4, 96GB VRAM) — v2: trivial collapse 対策
N_EPOCHS = 25
BATCH_SIZE = 256       # 96GB VRAM 余裕
LR = 1.4e-3            # ★ 8.5e-4→1.4e-3: Babych base 5e-4 × sqrt(256/32) = trivial 解 脱出強化
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2      # ★ 4→2: warmup 短縮、早く target LR
DROP_PATH = 0.10       # ★ 0.15→0.10: 初期 capacity 確保
LABEL_SMOOTHING = 0.0  # ★ 0.05→0.0: hard signal sharpen

NUM_WORKERS = 16
PERSISTENT_WORKERS = True

# Aug
MIXUP_ALPHA = 0.4
MIXUP_P = 0.3     # ★ 0.5→0.3: mixup 弱める、direct learning 強化
SPECAUG_FREQ = 10
SPECAUG_TIME = 10
BG_MIX_P = 0.7    # ★ 0.5→0.7: domain shift (focal→soundscape) 攻撃強化 (Tier 1)

# Val
VAL_FRACTION = 0.20
N_CLASSES = 234
LOG_STEP_INTERVAL = 100

SEED = 42

# Paths (Colab local SSD)
DATA_PATH = LOCAL_DATA / "birdclef-2026"
TRAIN_CSV = DATA_PATH / "train.csv"
TRAIN_AUDIO_DIR = DATA_PATH / "train_audio"
TRAIN_SC_DIR = DATA_PATH / "train_soundscapes"
TAXONOMY_CSV = DATA_PATH / "taxonomy.csv"
SAMPLE_SUB = DATA_PATH / "sample_submission.csv"

# Babych b0 ckpt
BABYCH_DIR = LOCAL_DATA / "babych_b25"
BABYCH_B0_CKPT = None
for f in BABYCH_DIR.rglob("tf_efficientnet_b0*.pt"):
    BABYCH_B0_CKPT = f; break
assert BABYCH_B0_CKPT is not None, f"Babych b0 ckpt not found in {BABYCH_DIR}"
print(f"Babych b0: {BABYCH_B0_CKPT}")

# Pseudo paths
SOUNDSCAPE_PSEUDO_DIR = LOCAL_DATA / "exp080a"
XC_PSEUDO_DIR = LOCAL_DATA / "exp080b"

# XC audio paths
XC_PART1 = LOCAL_DATA / "xc_part1"
XC_PART2 = LOCAL_DATA / "xc_part2"
XC_PART3 = LOCAL_DATA / "xc_part3"

OUT_DIR = LOCAL_OUT
DRIVE_OUT_DIR = DRIVE_OUT

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"\\nDATA_PATH: {DATA_PATH}")
print(f"OUT_DIR: {OUT_DIR}")
print(f"DRIVE_OUT: {DRIVE_OUT_DIR}")
"""))

# Cell 5: Load taxo + stratified split (same as Kaggle)
cells.append(code_cell("""# Load taxonomy + sample_submission
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES
LABEL2IDX = {label: i for i, label in enumerate(PRIMARY_LABELS)}
print(f"234 species: {len(PRIMARY_LABELS)}")

taxo = pd.read_csv(TAXONOMY_CSV)
label_to_taxon = dict(zip(taxo["primary_label"].astype(str), taxo["class_name"].astype(str)))
TAXON_MASKS = {
    t: np.array([i for i, lbl in enumerate(PRIMARY_LABELS) if label_to_taxon.get(lbl, "") == t])
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
}
print(f"Taxon: " + " ".join(f"{t}={len(m)}" for t, m in TAXON_MASKS.items()))

# Load train.csv (focal)
train_df = pd.read_csv(TRAIN_CSV)
train_df["primary_label"] = train_df["primary_label"].astype(str)
def _exists(fn): return (TRAIN_AUDIO_DIR / fn).exists()
train_df["exists"] = train_df["filename"].map(_exists)
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"train_df (focal, existing): {len(train_df)}")

# Stratified 80/20 split
from collections import Counter
label_counts = Counter(train_df["primary_label"])
single_label = [lbl for lbl, c in label_counts.items() if c < 2]
multi_label_df = train_df[~train_df["primary_label"].isin(single_label)].reset_index(drop=True)
single_label_df = train_df[train_df["primary_label"].isin(single_label)].reset_index(drop=True)

train_multi, val_multi = train_test_split(
    multi_label_df, test_size=VAL_FRACTION,
    stratify=multi_label_df["primary_label"], random_state=SEED,
)
train_focal_df = pd.concat([train_multi, single_label_df], ignore_index=True).reset_index(drop=True)
val_focal_df = val_multi.reset_index(drop=True)
print(f"  train_focal: {len(train_focal_df)}, val_focal: {len(val_focal_df)}")
print(f"  train sp: {train_focal_df['primary_label'].nunique()}, val sp: {val_focal_df['primary_label'].nunique()}")
"""))

# Cell 6: Load pseudo (same as Kaggle but local paths)
cells.append(code_cell("""# Load soundscape pseudo (exp080a)
ss_npz_path = next(SOUNDSCAPE_PSEUDO_DIR.rglob("pseudo_adaptive_234.npz"))
ss_npz = np.load(ss_npz_path, allow_pickle=True)
ss_probs = ss_npz["probs"].astype(np.float32)
ss_file_ids = ss_npz["file_ids"]
print(f"SS pseudo: {ss_probs.shape}, mean={ss_probs.mean():.5f}")

ss_id_to_path = {str(fid): TRAIN_SC_DIR / f"{fid}.ogg" for fid in ss_file_ids}
print(f"  SS exists: {sum(1 for p in ss_id_to_path.values() if p.exists())} / {len(ss_file_ids)}")

# Load XC pseudo (exp080b)
xc_npz_path = next(XC_PSEUDO_DIR.rglob("xc_pseudo_tucker.npz"))
xc_npz = np.load(xc_npz_path, allow_pickle=True)
xc_probs = xc_npz["probs"].astype(np.float32)
xc_file_ids = xc_npz["file_ids"]
xc_n_actual = xc_npz["n_actual_chunks"]
print(f"\\nXC pseudo: {xc_probs.shape}, mean={xc_probs.mean():.5f}, total valid chunks={int(xc_n_actual.sum())}")

# Build XC id → path map (rglob across 3 parts, tqdm 進捗)
print(f"  Building XC audio path index (rglob across Part 1+2+3)...")
xc_id_to_path_full = {}
for base in [XC_PART1, XC_PART2, XC_PART3]:
    if not base.exists():
        print(f"    {base.name}: skip (not mounted)")
        continue
    # First: count total mp3 for tqdm
    mp3_iter = base.rglob('*.mp3')
    base_files = list(tqdm.tqdm(mp3_iter, desc=f'  rglob {base.name}', unit='file'))
    print(f"    {base.name}: {len(base_files)} mp3 found")
    for fp in tqdm.tqdm(base_files, desc=f'  index {base.name}', unit='file'):
        xc_id_to_path_full[fp.stem] = fp
print(f"  Total XC mp3 indexed: {len(xc_id_to_path_full)}")

# Map xc_file_ids → path (subset of full)
xc_id_to_path = {}
for xc_id in tqdm.tqdm(xc_file_ids, desc='  match xc_id→path', unit='id'):
    s = str(xc_id)
    if s in xc_id_to_path_full:
        xc_id_to_path[s] = xc_id_to_path_full[s]
print(f"  XC matched: {len(xc_id_to_path)} / {len(xc_file_ids)}")
"""))

# Cell 7-9: model + datasets + train loop (mostly same as Kaggle)
cells.append(code_cell("""# Model (Babych SED b0)
def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class AttHead(nn.Module):
    def __init__(self, in_chans, p=0.5, num_class=234, hidden_dim=512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2), nn.Linear(in_chans, hidden_dim), nn.ReLU(), nn.Dropout(p),
        )
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        return {"framewise_logit": self.fix_scale(feat)}


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
        norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
        return (Xstd - norm_min) / (norm_max - norm_min + self.eps)


class SpecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(sample_rate=SR, normalized=True, n_fft=N_FFT,
                             hop_length=HOP_LENGTH, win_length=N_FFT,
                             f_max=F_MAX, n_mels=N_MELS, f_min=F_MIN),
            T.AmplitudeToDB(top_db=TOP_DB),
        )
        self.norm = NormalizeMelSpec()
    def forward(self, x):
        return self.norm(self.feature_extractor(x))


class CLEFClassifierSED(nn.Module):
    def __init__(self, num_classes=N_CLASSES, drop_path_rate=DROP_PATH):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0.ns_jft_in1k", pretrained=True, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)
        self.num_classes = num_classes

    def forward(self, wav, return_framewise=False):
        spec = self.mel_spectr_generator(wav)
        spec3 = torch.stack([spec, spec, spec], 1)
        feat = self.backbone(spec3)[-1]
        head_output = self.head(feat)
        framewise_logit = head_output["framewise_logit"]
        clip_logit = framewise_logit.max(dim=-1).values
        if return_framewise:
            return clip_logit, framewise_logit
        return clip_logit


def make_model_with_babych_init():
    model = CLEFClassifierSED()
    babych_state = torch.load(str(BABYCH_B0_CKPT), weights_only=True, map_location="cpu")
    backbone_state = {k: v for k, v in babych_state.items() if k.startswith("backbone.")}
    msg = model.load_state_dict(backbone_state, strict=False)
    print(f"  Babych load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    return model


_tmp = make_model_with_babych_init()
print(f"Model: {sum(p.numel() for p in _tmp.parameters())/1e6:.1f}M params")
del _tmp; gc.collect()
"""))

cells.append(code_cell("""# Datasets — same as Kaggle version
class FocalHardDS(Dataset):
    def __init__(self, df, train_audio_dir, label2idx, ss_paths_for_bg=None, train_mode=True):
        self.df = df.reset_index(drop=True)
        self.dir = Path(train_audio_dir)
        self.label2idx = label2idx
        self.train_mode = train_mode
        self.ss_paths = ss_paths_for_bg if (train_mode and ss_paths_for_bg) else None

    def __len__(self): return len(self.df)

    def load_audio(self, filename):
        try:
            y, _ = librosa.load(str(self.dir / filename), sr=SR, mono=True)
            return y.astype(np.float32)
        except Exception:
            return np.zeros(SR * 5, dtype=np.float32)

    def crop_5s(self, y):
        if len(y) < WINDOW_SAMPLES:
            pad = WINDOW_SAMPLES - len(y)
            left = np.random.randint(0, pad + 1) if self.train_mode else pad // 2
            y = np.pad(y, (left, pad - left))
        elif len(y) > WINDOW_SAMPLES:
            start = np.random.randint(0, len(y) - WINDOW_SAMPLES + 1) if self.train_mode else (len(y) - WINDOW_SAMPLES) // 2
            y = y[start: start + WINDOW_SAMPLES]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.load_audio(row["filename"])
        y = self.crop_5s(y)
        if self.train_mode and self.ss_paths and np.random.random() < BG_MIX_P:
            bg_path = self.ss_paths[np.random.randint(len(self.ss_paths))]
            try:
                bg, _ = librosa.load(str(bg_path), sr=SR, mono=True)
                bg = self.crop_5s(bg.astype(np.float32))
                y = 0.7 * y + 0.3 * bg
            except Exception:
                pass
        m = np.abs(y).max()
        if m > 0: y = y / m
        label = np.zeros(N_CLASSES, dtype=np.float32)
        if row["primary_label"] in self.label2idx:
            label[self.label2idx[row["primary_label"]]] = 1.0
        sec = str(row.get("secondary_labels", "")).strip()
        if sec and sec != "[]" and sec != "nan":
            for s in sec.replace("[", "").replace("]", "").replace("'", "").split(","):
                s = s.strip()
                if s in self.label2idx:
                    label[self.label2idx[s]] = 1.0
        if LABEL_SMOOTHING > 0:
            label = label * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / N_CLASSES
        return torch.from_numpy(y), torch.from_numpy(label)


class SoundscapePseudoDS(Dataset):
    def __init__(self, file_ids, id_to_path, pseudo_array):
        self.entries = []
        for i, fid in enumerate(file_ids):
            p = id_to_path.get(str(fid))
            if p and p.exists():
                self.entries.append((str(fid), p, i))
        self.pseudo = pseudo_array

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        fid, path, pseudo_idx = self.entries[idx]
        win = np.random.randint(12)
        try:
            y, _ = librosa.load(str(path), sr=SR, mono=True, offset=win * 5.0, duration=5.0)
        except Exception:
            y = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        else:
            y = y[:WINDOW_SAMPLES]
        m = np.abs(y).max()
        if m > 0: y = y / m
        target = self.pseudo[pseudo_idx, win].astype(np.float32)
        return torch.from_numpy(y.astype(np.float32)), torch.from_numpy(target)


class XCPseudoDS(Dataset):
    \"\"\"XC audio (MP3): full load + pad 60s + reshape for alignment.\"\"\"
    def __init__(self, file_ids, id_to_path, pseudo_array, n_actual_chunks):
        self.entries = []
        for i, fid in enumerate(file_ids):
            p = id_to_path.get(str(fid))
            n_act = int(n_actual_chunks[i])
            if p and p.exists() and n_act >= 1:
                self.entries.append((str(fid), p, i, n_act))
        self.pseudo = pseudo_array
        self.target_60s_samples = SR * 60

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        fid, path, pseudo_idx, n_act = self.entries[idx]
        try:
            y, _ = librosa.load(str(path), sr=SR, mono=True)
            y = y.astype(np.float32)
            if len(y) < self.target_60s_samples:
                y = np.pad(y, (0, self.target_60s_samples - len(y)))
            else:
                y = y[:self.target_60s_samples]
        except Exception:
            y = np.zeros(self.target_60s_samples, dtype=np.float32)
        chunks = y.reshape(12, WINDOW_SAMPLES)
        win = np.random.randint(n_act)
        chunk = chunks[win]
        m = np.abs(chunk).max()
        if m > 0: chunk = chunk / m
        target = self.pseudo[pseudo_idx, win].astype(np.float32)
        return torch.from_numpy(chunk.astype(np.float32)), torch.from_numpy(target)


ss_path_list = [p for p in ss_id_to_path.values() if p.exists()][:200]
train_focal_ds = FocalHardDS(train_focal_df, TRAIN_AUDIO_DIR, LABEL2IDX,
                              ss_paths_for_bg=ss_path_list, train_mode=True)
val_focal_ds = FocalHardDS(val_focal_df, TRAIN_AUDIO_DIR, LABEL2IDX, train_mode=False)
ss_ds = SoundscapePseudoDS(ss_file_ids, ss_id_to_path, ss_probs)
xc_ds = XCPseudoDS(xc_file_ids, xc_id_to_path, xc_probs, xc_n_actual)

print(f"train_focal_ds: {len(train_focal_ds)}, val_focal_ds: {len(val_focal_ds)}")
print(f"ss_ds: {len(ss_ds)}, xc_ds: {len(xc_ds)}")
train_combined_ds = ConcatDataset([train_focal_ds, ss_ds, xc_ds])
print(f"Combined: {len(train_combined_ds)}")
"""))

# Cell 10: train helpers + loop
cells.append(code_cell("""# Helpers
def mixup_audio(wav, label, alpha=MIXUP_ALPHA, p=MIXUP_P):
    if np.random.random() >= p:
        return wav, label
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(wav.size(0), device=wav.device)
    return lam * wav + (1 - lam) * wav[idx], lam * label + (1 - lam) * label[idx]


@torch.no_grad()
def eval_val_focal_macro(model, val_dl, device):
    model.eval()
    all_preds, all_labels = [], []
    for wav, label in val_dl:
        wav = wav.to(device, non_blocking=True)
        with autocast('cuda'):
            clip_logit = model(wav)
        all_preds.append(torch.sigmoid(clip_logit).float().cpu().numpy())
        all_labels.append(label.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    labels_bin = (labels > 0.5).astype(np.float32)
    aucs = []
    for sp in range(N_CLASSES):
        if labels_bin[:, sp].sum() > 0 and labels_bin[:, sp].sum() < len(labels_bin):
            try:
                aucs.append(roc_auc_score(labels_bin[:, sp], preds[:, sp]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def class_stats_str(aucs):
    if not aucs:
        return "n=0"
    vals = np.array(aucs)
    return (f"n={len(vals)} median={np.median(vals):.3f} p25={np.percentile(vals,25):.3f} "
            f"p75={np.percentile(vals,75):.3f} #>0.5={int((vals>0.5).sum())} #>0.7={int((vals>0.7).sum())} "
            f"#>0.9={int((vals>0.9).sum())} #perfect={int((vals>=1.0).sum())}")


@torch.no_grad()
def eval_val_full(model, val_dl, device):
    \"\"\"Returns val_focal_macro + per-taxon + class_stats (M7-style).\"\"\"
    model.eval()
    all_preds, all_labels = [], []
    for wav, label in val_dl:
        wav = wav.to(device, non_blocking=True)
        with autocast('cuda'):
            clip_logit = model(wav)
        all_preds.append(torch.sigmoid(clip_logit).float().cpu().numpy())
        all_labels.append(label.numpy())
    preds = np.concatenate(all_preds)
    labels_bin = (np.concatenate(all_labels) > 0.5).astype(np.float32)
    # per-class AUC
    per_sp_aucs = {}
    for sp in range(N_CLASSES):
        if labels_bin[:, sp].sum() > 0 and labels_bin[:, sp].sum() < len(labels_bin):
            try:
                per_sp_aucs[sp] = roc_auc_score(labels_bin[:, sp], preds[:, sp])
            except Exception: pass
    macro = float(np.mean(list(per_sp_aucs.values()))) if per_sp_aucs else float("nan")
    # Per-taxon
    taxon_aucs = {}
    for t, mask in TAXON_MASKS.items():
        tx_vals = [per_sp_aucs[sp] for sp in mask if sp in per_sp_aucs]
        taxon_aucs[t] = float(np.mean(tx_vals)) if tx_vals else float("nan")
    return macro, len(per_sp_aucs), taxon_aucs, list(per_sp_aucs.values())
"""))

cells.append(code_cell("""# Training loop (with M7-style logging restored)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

train_dl = DataLoader(train_combined_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                       persistent_workers=PERSISTENT_WORKERS, prefetch_factor=6)
val_dl = DataLoader(val_focal_ds, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=PERSISTENT_WORKERS)

model = make_model_with_babych_init().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
warmup_iters = WARMUP_EPOCHS * len(train_dl)
total_iters = N_EPOCHS * len(train_dl)
sched_warmup = LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_iters)
sched_cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=LR_MIN)
scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_iters])
scaler = GradScaler('cuda')

best_val = -1.0
history = []
total_steps = len(train_dl)
print(f"=== Training: {N_EPOCHS} epochs × {total_steps} steps, batch={BATCH_SIZE} ===")

for epoch in range(N_EPOCHS):
    t0_ep = time.time()
    model.train()
    tr_loss_sum, bce_sum, n_seen = 0.0, 0.0, 0
    for step, (wav, label) in enumerate(train_dl):
        wav = wav.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        wav, label = mixup_audio(wav, label)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            clip_logit, framewise_logit = model(wav, return_framewise=True)
            frame_max_logit = framewise_logit.max(dim=-1).values
            loss_clip = F.binary_cross_entropy_with_logits(clip_logit, label)
            loss_frame = F.binary_cross_entropy_with_logits(frame_max_logit, label)
            bce = 0.5 * loss_clip + 0.5 * loss_frame
            loss = bce
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tr_loss_sum += loss.item() * wav.size(0)
        bce_sum += bce.item() * wav.size(0)
        n_seen += wav.size(0)
        if step % LOG_STEP_INTERVAL == 0 or step == total_steps - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  [ep{epoch+1} step {step}/{total_steps}] loss={loss.item():.4f} bce={bce.item():.4f} lr={cur_lr:.2e}")

    tr_loss = tr_loss_sum / max(n_seen, 1)
    bce_avg = bce_sum / max(n_seen, 1)

    # Val (full M7-style)
    val_macro, n_valid_sp, taxon_aucs, all_aucs = eval_val_full(model, val_dl, DEVICE)
    ep_time = (time.time() - t0_ep) / 60
    total_time = (time.time() - START) / 60
    cur_lr = optimizer.param_groups[0]["lr"]
    is_best = val_macro > best_val
    best_tag = "BEST " if is_best else ""
    tax_line = "taxon: " + " ".join(f"{t}={taxon_aucs[t]:.3f}" if not math.isnan(taxon_aucs[t]) else f"{t}=nan"
                                     for t in ["Insecta", "Reptilia", "Amphibia", "Mammalia", "Aves"])
    cls_line = class_stats_str(all_aucs)

    print(f"=== Ep {epoch+1}/{N_EPOCHS}: loss={tr_loss:.4f} (bce={bce_avg:.4f}) "
          f"val_focal_macro={val_macro:.4f} ({n_valid_sp} sp) {best_tag}lr={cur_lr:.2e} "
          f"({ep_time:.1f}min, total {total_time:.1f}min) ===")
    print(f"    {tax_line}")
    print(f"    class: {cls_line}")

    if is_best:
        best_val = val_macro
        torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch, "val_focal_macro": val_macro},
                   OUT_DIR / "m_single_ckpt_best.pth")
        print(f"    BEST saved val={val_macro:.4f}")

    history.append({
        "ep": epoch, "tr_loss": tr_loss, "bce": bce_avg,
        "val_focal_macro": val_macro, "n_valid_sp": n_valid_sp,
        "taxon": taxon_aucs, "lr": cur_lr, "ep_time_min": ep_time, "best": is_best,
    })

    # Drive mirror per epoch (safety against Colab disconnect)
    try:
        shutil.copy(OUT_DIR / "m_single_ckpt_best.pth", DRIVE_OUT_DIR / "m_single_ckpt_best.pth")
        with open(DRIVE_OUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        print(f"  Drive mirror warn: {e}")

# Final
torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "epoch": N_EPOCHS - 1, "history": history},
           OUT_DIR / "m_single_ckpt_final.pth")
with open(OUT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2, default=str)
print(f"\\nTraining DONE: best val={best_val:.4f}, total {(time.time()-START)/60:.1f} min")
"""))

# Cell 12: ONNX export
cells.append(code_cell("""# ONNX export
print("=== ONNX export ===")
import shutil
best_ckpt = torch.load(OUT_DIR / "m_single_ckpt_best.pth", weights_only=False, map_location="cpu")
export_model = CLEFClassifierSED().cpu().eval()
export_model.load_state_dict(best_ckpt["model_state"])

dummy_wav = torch.randn(1, WINDOW_SAMPLES, dtype=torch.float32)
with torch.no_grad():
    out = export_model(dummy_wav)
    print(f"  Trace: {out.shape}")

onnx_path = OUT_DIR / "m_single_best.onnx"
try:
    torch.onnx.export(
        export_model, dummy_wav, str(onnx_path),
        input_names=["wav"], output_names=["clip_logits"],
        dynamic_axes={"wav": {0: "batch"}, "clip_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"  ONNX exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    print(f"  ONNX export FAILED: {e}")

try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"wav": dummy_wav.numpy()})[0]
    diff = np.abs(out.numpy() - onnx_out).max()
    print(f"  PyTorch vs ONNX diff: {diff:.6e}")
except Exception as e:
    print(f"  ONNX check failed: {e}")

# OpenVINO IR export (.xml + .bin, 3-5x CPU speedup vs ONNX)
print("\\n=== OpenVINO IR export ===")
!pip install -q openvino
try:
    import openvino as ov
    ov_model = ov.convert_model(str(onnx_path))
    ov_xml = OUT_DIR / "m_single_best.xml"
    ov.save_model(ov_model, str(ov_xml))
    ov_bin = OUT_DIR / "m_single_best.bin"
    print(f"  OV IR: {ov_xml.name} ({ov_xml.stat().st_size/1e6:.2f} MB), {ov_bin.name} ({ov_bin.stat().st_size/1e6:.2f} MB)")
    # Speed sanity check
    try:
        core = ov.Core()
        compiled = core.compile_model(ov_model, "CPU")
        ov_out = compiled([dummy_wav.numpy()])[compiled.output(0)]
        ov_diff = np.abs(out.numpy() - ov_out).max()
        print(f"  PyTorch vs OpenVINO diff: {ov_diff:.6e}")
    except Exception as e:
        print(f"  OV runtime check failed: {e}")
except Exception as e:
    print(f"  OpenVINO export FAILED: {e}")
    import traceback; traceback.print_exc()
"""))

# Cell 13: Drive mirror final + Kaggle dataset upload
cells.append(code_cell("""# Mirror to Drive + (optional) Kaggle Dataset upload
import shutil

# Mirror to Drive (.pth + .onnx + OpenVINO .xml/.bin + history)
for fname in [
    "m_single_ckpt_best.pth", "m_single_ckpt_final.pth",
    "m_single_best.onnx",
    "m_single_best.xml", "m_single_best.bin",
    "history.json",
]:
    src = OUT_DIR / fname
    if src.exists():
        dst = DRIVE_OUT_DIR / fname
        shutil.copy(str(src), str(dst))
        print(f"  Mirror: {fname} → {dst} ({dst.stat().st_size/1e6:.1f} MB)")

# Kaggle Dataset upload (for inference NB to consume)
KAGGLE_DS_SLUG = "birdclef2026-exp080-train-output"
DS_TITLE = "BirdCLEF2026 exp080 train output"

ds_dir = OUT_DIR / "kaggle_ds"
ds_dir.mkdir(exist_ok=True)
for fname in [
    "m_single_ckpt_best.pth", "m_single_best.onnx",
    "m_single_best.xml", "m_single_best.bin",
    "history.json",
]:
    src = OUT_DIR / fname
    if src.exists(): shutil.copy(str(src), str(ds_dir / fname))

meta = {"title": DS_TITLE, "id": f"maekeso/{KAGGLE_DS_SLUG}", "licenses": [{"name": "CC0-1.0"}]}
with open(ds_dir / "dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

try:
    api.dataset_create_new(folder=str(ds_dir), public=False, dir_mode="zip", quiet=False)
    print(f"  Created Kaggle Dataset: maekeso/{KAGGLE_DS_SLUG}")
except Exception:
    try:
        api.dataset_create_version(folder=str(ds_dir), version_notes="initial",
                                    dir_mode="zip", quiet=False)
        print(f"  Updated Kaggle Dataset: maekeso/{KAGGLE_DS_SLUG}")
    except Exception as e:
        print(f"  Upload failed (manual upload via Drive needed): {str(e)[:200]}")

print(f"\\n=== exp080 Colab DONE ===")
print(f"Best val_focal_macro: {best_val:.4f}")
print(f"Total time: {(time.time()-START)/60:.1f} min")
"""))

# Cell 14: Auto-disconnect (Pro+ unit saving)
cells.append(code_cell("""# Auto-disconnect (Colab Pro+ unit saving)
from google.colab import runtime
# Comment out if you want to inspect manually
# runtime.unassign()
print("Done. Uncomment runtime.unassign() to auto-disconnect.")
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Built: {OUT_PATH}")
print(f"  cells: {len(cells)}")
for c in cells:
    n_lines = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} L={n_lines}")
