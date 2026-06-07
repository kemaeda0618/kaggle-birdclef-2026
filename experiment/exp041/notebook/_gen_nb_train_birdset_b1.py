"""exp041: BirdSet EffNet-B1 (XCL pretrain) BC2026 fine-tune NB.

Train NB for Colab Pro A100. Loads dacquaviva/birdset-effnet-b1-xcl
(EfficientNet-B1 pretrained on Xeno-Canto Large via BirdSet) and fine-tunes
on BC2026 (labeled SS + train_audio).

Architecture: HuggingFace EfficientNetForImageClassification (transformers).
Pretrain chain: ImageNet → BirdSet XCL (9736 species) → BC2026 (234 species).

Mel-spec preprocessing matches BirdSet (n_fft=2048, hop=2048, n_mels=256,
power=2.0, top_db=80, normalize mean=-4.268 std=4.569 from ESC-50).
Input: (B, 1, 256, ~78) per 5s @ 32kHz.

Expected:
- standalone LB 0.93+ (Antoine reports EffNetV2-B0 + XC pretrain = 0.936)
- 4th stream as Perch-independent axis = high diversity contribution
- blend +0.002-0.005

Generates: experiment/exp041/notebook/nb_train_birdset_b1.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

# ── Header ──
cells.append(md_cell("hdr", r"""# exp041 — BirdSet EffNet-B1 (XCL pretrain) BC2026 fine-tune (Colab A100)

**Antoine 路線** (discussion 700763): EffNet + Xeno-Canto pretrain + BC2026 fine-tune で **Perch 完全独立軸** の 4th stream を作る。

## Architecture
| 項目 | 値 |
|---|---|
| Backbone | EfficientNet-B1 (`google/efficientnet-b1`) |
| Pretrain chain | ImageNet → BirdSet XCL (9736 species、Xeno-Canto Large) → BC2026 (234) |
| Pretrained weights | `dacquaviva/birdset-effnet-b1-xcl` (Kaggle Dataset、76 MB) |
| Framework | HuggingFace transformers (`EfficientNetForImageClassification`) |
| 訓練データ | BC2026 train_audio (35k focal) + labeled SS (66 files) |

## Mel-spec (BirdSet 仕様、my pipeline と異なる)
| 項目 | 値 | vs my exp017/020 |
|---|---|---|
| sample_rate | 32000 | 同じ |
| n_fft | 2048 | 同じ |
| hop_length | **2048** (no overlap) | **512 → 2048 (4x coarser time)** |
| n_mels | 256 | 同じ |
| power | 2.0 | 同じ |
| top_db | 80 | 同じ |
| normalize | mean=-4.268, std=4.569 (ESC-50) | per-window mean/std |
| 入力 shape (5s) | (1, 256, ~78) | (1, 256, 313) |

## Expected
- standalone LB **0.93+** (Antoine 0.936 with B0 + XC pretrain、B1 で同等以上)
- 4th stream = Perch 完全独立軸 → diversity contribution +0.002-0.005 blend lift

## Run
1. Colab Pro A100 で Run All
2. Drive に ckpt 保存
3. 最終 cell で Kaggle Dataset へ auto-upload
4. Inference NB は `experiment/exp041/notebook/nb_infer_birdset_b1.ipynb`
"""))

# ── Install ──
cells.append(code_cell("install", r"""# Install dependencies
import subprocess, sys

PIPS = [
    "transformers>=4.30",
    "safetensors",
    "kaggle",  # for Dataset upload
]
for p in PIPS:
    print(f"installing {p}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=True)
print("✅ deps installed")
"""))

# ── Imports ──
cells.append(code_cell("imports", r"""import os, sys, time, json, math, glob, re, gc, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchaudio
import soundfile as sf
import librosa

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
"""))

# ── Mount Drive + paths ──
# 過去 exp020 R1 と同じ convention:
#   Drive = project root (kaggle.json + output)
#   Local /content/data = competition data (Kaggle API で毎セッション download)
cells.append(code_cell("paths", r"""# Mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Drive: project root
DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_OUTPUT_DIR = DRIVE_INPUT_DIR / "output" / "exp041"
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert DRIVE_INPUT_DIR.exists(), f"Drive root not found: {DRIVE_INPUT_DIR}"
print(f"Drive root:   {DRIVE_INPUT_DIR}")
print(f"Drive output: {DRIVE_OUTPUT_DIR}")

# kaggle.json auth (multi-path search)
def _find_kaggle_json():
    candidates = [
        Path("/root/.kaggle/kaggle.json"),
        DRIVE_INPUT_DIR / "kaggle.json",
        Path("/content/kaggle.json"),
        Path("/content/drive/MyDrive/kaggle.json"),
        Path("/content/drive/MyDrive/kaggle/kaggle.json"),
        Path("/content/drive/MyDrive/.kaggle/kaggle.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        for hit in drive_root.rglob("kaggle.json"):
            return hit
    return None

import shutil
_cred_dst = Path("/root/.kaggle/kaggle.json")
_cred_dst.parent.mkdir(parents=True, exist_ok=True)
if not _cred_dst.exists():
    _cred_src = _find_kaggle_json()
    assert _cred_src is not None, "kaggle.json not found in any common location"
    print(f"kaggle.json: {_cred_src}")
    shutil.copy(_cred_src, _cred_dst)
    os.chmod(_cred_dst, 0o600)
_creds = json.loads(_cred_dst.read_text())
if _creds.get("key", "").startswith("KGAT_"):
    os.environ["KAGGLE_API_TOKEN"] = _creds["key"]

# Local /content for competition data + BirdSet weights
LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
BC_DIR = LOCAL_DATA   # competition files extract directly here (taxonomy.csv 等)
TRAIN_AUDIO_DIR = BC_DIR / "train_audio"
TRAIN_SC_DIR = BC_DIR / "train_soundscapes"

# BirdSet pretrained model — local (auto-download from Kaggle Dataset)
BIRDSET_DIR = LOCAL_DATA / "birdset-effnet-b1-xcl"
"""))

# ── Download competition data ──
cells.append(code_cell("download_data", r"""# Download competition data (~25 GB) to /content/data — fresh each session
import subprocess, zipfile, time
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm.auto import tqdm

api = KaggleApi(); api.authenticate()
print("kaggle authenticated")

# Skip if already extracted (resume)
TAXO_PATH = BC_DIR / "taxonomy.csv"
TRAIN_CSV = BC_DIR / "train.csv"
SAMPLE_SUB_PATH = BC_DIR / "sample_submission.csv"
SC_LABELS_CSV = BC_DIR / "train_soundscapes_labels.csv"

need_dl = (
    not TAXO_PATH.exists() or
    not TRAIN_AUDIO_DIR.exists() or sum(1 for _ in TRAIN_AUDIO_DIR.rglob("*.ogg")) < 30000 or
    not TRAIN_SC_DIR.exists()  or sum(1 for _ in TRAIN_SC_DIR.glob("*.ogg")) < 10000
)
if need_dl:
    print("\nDownloading birdclef-2026 (~25 GB)...")
    t0 = time.time()
    api.competition_download_files("birdclef-2026", path=str(BC_DIR), force=False, quiet=False)
    print(f"  DL done in {(time.time()-t0)/60:.1f} min")
    zips = list(BC_DIR.glob("birdclef-2026*.zip"))
    assert zips, "zip not found after download"
    zip_path = zips[0]
    print("\n  Extracting...")
    t_extract = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        total_bytes = sum(i.file_size for i in infos)
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="extract", mininterval=1.0)
        for info in infos:
            zf.extract(info, BC_DIR)
            pbar.update(info.file_size)
        pbar.close()
    print(f"  extracted in {(time.time()-t_extract)/60:.1f} min")
    zip_path.unlink()
else:
    print("Competition data already present locally")

n_ta = sum(1 for _ in TRAIN_AUDIO_DIR.rglob("*.ogg"))
n_ts = sum(1 for _ in TRAIN_SC_DIR.glob("*.ogg"))
print(f"\n  taxonomy: {TAXO_PATH.exists()}, train.csv: {TRAIN_CSV.exists()}, sample_sub: {SAMPLE_SUB_PATH.exists()}")
print(f"  train_audio: {n_ta}, train_soundscapes: {n_ts}")
"""))

# ── Download BirdSet weights ──
cells.append(code_cell("download_birdset", r"""# Download BirdSet EffNet-B1 XCL weights (~80 MB)
if (BIRDSET_DIR / "model.safetensors").exists() and (BIRDSET_DIR / "config.json").exists():
    print(f"BirdSet weights already present at {BIRDSET_DIR}")
else:
    print(f"Downloading dacquaviva/birdset-effnet-b1-xcl to {BIRDSET_DIR}...")
    BIRDSET_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "dacquaviva/birdset-effnet-b1-xcl",
        "-p", str(BIRDSET_DIR),
        "--unzip",
    ], check=True)

assert (BIRDSET_DIR / "model.safetensors").exists(), f"missing: {BIRDSET_DIR}/model.safetensors"
for f in sorted(BIRDSET_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
"""))

# ── Config ──
cells.append(code_cell("config", r"""# CONFIG
NUM_CLASSES = 234
SR = 32000
DURATION_SEC = 5
SAMPLES_PER_CHUNK = SR * DURATION_SEC      # 160000
N_WINDOWS = 12

# Mel-spec (BirdSet 仕様)
N_FFT = 2048
HOP_LENGTH = 2048     # ★ no overlap、BirdSet 仕様
N_MELS = 256
FMIN = 0
FMAX = None           # default (16000 = nyquist/2)
POWER = 2.0
TOP_DB = 80.0
NORM_MEAN = -4.268    # ESC-50 mean (BirdSet preprocessor)
NORM_STD = 4.569      # ESC-50 std

# Training (G4 96GB / Colab Pro+ 最適化)
N_EPOCHS = 15           # 20 → 15 (XCL pretrained で overfit リスク低減)
BATCH_SIZE = 192        # 過去 exp020 R1 と統一、G4 96GB で余裕
LR = 3e-4               # 1e-3 → 3e-4 (fine-tune 安全圏 + batch 4x の sqrt scale)
WEIGHT_DECAY = 1e-4
HEAD_LR_MULT = 5.0      # 3.0 → 5.0 (head 完全 fresh、より aggressive)
USE_MIXUP = True
MIXUP_ALPHA = 0.4
USE_SPEC_AUGMENT = True # Babych BC25 1位 SpecAugment +0.001-0.002
SPEC_FREQ_MASK = 20     # mel bins
SPEC_TIME_MASK = 10     # time frames (~640ms @ hop=2048)
SPEC_NUM_FREQ = 2
SPEC_NUM_TIME = 2
PCT_START = 0.05        # OneCycleLR warmup 5% (head が fresh なので必須)

SEED = 42

# Save
CKPT_NS22_PATH = DRIVE_OUTPUT_DIR / "ckpt_best_ns22.pth"
CKPT_LATEST_PATH = DRIVE_OUTPUT_DIR / "ckpt_latest.pth"
HISTORY_PATH = DRIVE_OUTPUT_DIR / "history.json"

print(f"Mel input shape per 5s: (1, {N_MELS}, ~{SAMPLES_PER_CHUNK//HOP_LENGTH})")
print(f"N_EPOCHS={N_EPOCHS}  BATCH={BATCH_SIZE}  LR={LR} (head x{HEAD_LR_MULT})")
print(f"OneCycleLR warmup={int(PCT_START*100)}%、Mixup α={MIXUP_ALPHA}、SpecAugment={USE_SPEC_AUGMENT}")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything(SEED)
"""))

# ── Load BC2026 metadata ──
cells.append(code_cell("metadata", r"""# Load BC2026 metadata
taxonomy = pd.read_csv(TAXO_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

train_df = pd.read_csv(TRAIN_CSV)
print(f"train_audio: {len(train_df)} rows")
print(f"  primary_label distinct: {train_df['primary_label'].nunique()}")

# Build samples list: train_audio focal + labeled SS
samples = []
for _, row in train_df.iterrows():
    fp = TRAIN_AUDIO_DIR / row["filename"]
    if fp.exists():
        labels = np.zeros(NUM_CLASSES, dtype=np.float32)
        if row["primary_label"] in label_to_idx:
            labels[label_to_idx[row["primary_label"]]] = 1.0
        # Secondary labels (optional, may help)
        if pd.notna(row.get("secondary_labels")) and isinstance(row["secondary_labels"], str):
            try:
                sec = eval(row["secondary_labels"])
                for s in (sec or []):
                    if s in label_to_idx:
                        labels[label_to_idx[s]] = 0.5
            except Exception:
                pass
        samples.append({"path": str(fp), "labels": labels, "src": "train_audio"})

# Labeled SS samples
if SC_LABELS_CSV.exists():
    sc_labels = pd.read_csv(SC_LABELS_CSV)
    sc_label_map = {}
    for _, r in sc_labels.iterrows():
        fn = r["filename"]
        end_sec = int(pd.Timedelta(r["end"]).total_seconds())
        rid = f"{Path(fn).stem}_{end_sec}"
        if rid not in sc_label_map:
            sc_label_map[rid] = np.zeros(NUM_CLASSES, dtype=np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in label_to_idx:
                sc_label_map[rid][label_to_idx[s]] = 1.0
    # Aggregate per-file SS samples (one 60s file = 12 × 5s windows)
    sc_files = sc_labels["filename"].unique().tolist()
    for fn in sc_files:
        fp = TRAIN_SC_DIR / fn
        if fp.exists():
            # Per-window samples
            for wi in range(N_WINDOWS):
                end_sec = (wi + 1) * DURATION_SEC
                rid = f"{Path(fn).stem}_{end_sec}"
                if rid in sc_label_map:
                    samples.append({
                        "path": str(fp),
                        "window_idx": wi,
                        "labels": sc_label_map[rid],
                        "src": "labeled_ss",
                    })

print(f"Total samples: {len(samples)}")
print(f"  train_audio: {sum(1 for s in samples if s['src']=='train_audio')}")
print(f"  labeled_ss:  {sum(1 for s in samples if s['src']=='labeled_ss')}")
"""))

# ── Dataset / DataLoader ──
cells.append(code_cell("dataset", r"""# Dataset
def load_audio_5s(path, sr=SR, window_idx=None):
    # Load 5s waveform; if window_idx is given, extract that 5s window from 60s file.
    try:
        wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    except Exception:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        wav = wav.astype(np.float32)

    if window_idx is not None:
        # Labeled SS: extract specific 5s window from 60s file
        start = window_idx * SAMPLES_PER_CHUNK
        end = start + SAMPLES_PER_CHUNK
        if end > len(wav):
            wav = np.pad(wav, (0, end - len(wav)))
        return wav[start:end].astype(np.float32)

    # train_audio: random crop 5s
    target_len = SAMPLES_PER_CHUNK
    if len(wav) <= target_len:
        return np.pad(wav, (0, target_len - len(wav))).astype(np.float32)
    start = np.random.randint(0, len(wav) - target_len + 1)
    return wav[start:start + target_len].astype(np.float32)


class MelSpecTransform(nn.Module):
    # BirdSet-style mel-spec: n_fft=2048, hop=2048 (no overlap), n_mels=256, normalize via ESC-50 stats.
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=POWER,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
    def forward(self, wav):
        # wav: (B, T) or (T,)
        if wav.ndim == 1: wav = wav.unsqueeze(0)
        mel = self.mel_spec(wav)              # (B, 256, T_mel)
        mel = self.db_transform(mel)
        mel = (mel - NORM_MEAN) / NORM_STD    # BirdSet normalize
        return mel  # (B, 256, T_mel)


class BC2026Dataset(Dataset):
    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        wav = load_audio_5s(s["path"], sr=SR, window_idx=s.get("window_idx"))
        if self.augment and np.random.rand() < 0.5:
            # Random gain (-3 to +3 dB)
            gain = 10 ** (np.random.uniform(-3, 3) / 20)
            wav = wav * gain
        return torch.from_numpy(wav).float(), torch.from_numpy(s["labels"]).float()


class SpecAugment(nn.Module):
    # Babych BC25 1位 流: freq + time masking で train-only augmentation
    def __init__(self, freq_mask, time_mask, num_freq, num_time):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask)
        self.num_freq, self.num_time = num_freq, num_time
    def forward(self, mel):
        for _ in range(self.num_freq): mel = self.freq_mask(mel)
        for _ in range(self.num_time): mel = self.time_mask(mel)
        return mel


print(f"Dataset class + SpecAugment defined")
"""))

# ── Model ──
cells.append(code_cell("model", r"""# Model: BirdSet EffNet-B1 + 234-class head
from transformers import EfficientNetForImageClassification, EfficientNetConfig

print(f"Loading BirdSet pretrained from {BIRDSET_DIR}...")
config = EfficientNetConfig.from_pretrained(BIRDSET_DIR)
config.num_labels = NUM_CLASSES
config.num_channels = 1  # mel-spec single channel

model = EfficientNetForImageClassification.from_pretrained(
    BIRDSET_DIR,
    config=config,
    ignore_mismatched_sizes=True,  # head 9736 → 234 で size mismatch OK
).to(device)

print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  config.num_labels = {model.config.num_labels}")
print(f"  classifier: {model.classifier}")

# Separate param groups for differential LR (head 3x LR)
backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]
print(f"  backbone params: {sum(p.numel() for p in backbone_params)/1e6:.1f}M")
print(f"  head params:     {sum(p.numel() for p in head_params)/1e3:.1f}K")
"""))

# ── Training loop ──
cells.append(code_cell("train", r"""# Training loop
mel_transform = MelSpecTransform().to(device)
spec_aug = SpecAugment(SPEC_FREQ_MASK, SPEC_TIME_MASK, SPEC_NUM_FREQ, SPEC_NUM_TIME).to(device) if USE_SPEC_AUGMENT else None

# Split: 95% train, 5% val
np.random.seed(SEED)
perm = np.random.permutation(len(samples))
val_size = max(64, int(len(samples) * 0.05))
val_idx = perm[:val_size]
tr_idx = perm[val_size:]
tr_samples = [samples[i] for i in tr_idx]
val_samples = [samples[i] for i in val_idx]
print(f"Train: {len(tr_samples)}, Val: {len(val_samples)}")

tr_ds = BC2026Dataset(tr_samples, augment=True)
val_ds = BC2026Dataset(val_samples, augment=False)
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

# Optimizer with differential LR
optimizer = AdamW([
    {"params": backbone_params, "lr": LR},
    {"params": head_params, "lr": LR * HEAD_LR_MULT},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[LR, LR * HEAD_LR_MULT],
    total_steps=N_EPOCHS * len(tr_loader),
    pct_start=PCT_START,
    anneal_strategy="cos",
)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

# Resume if ckpt exists
start_epoch = 0
best_val_auc = 0.0
history = []
if CKPT_LATEST_PATH.exists():
    print(f"Resuming from {CKPT_LATEST_PATH}")
    try:
        state = torch.load(str(CKPT_LATEST_PATH), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(CKPT_LATEST_PATH), map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = state.get("epoch", 0)
    best_val_auc = state.get("best_val_auc", 0.0)
    history = state.get("history", [])
    print(f"  resumed at epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

def compute_macro_auc(probs, labels, mask=None):
    # Macro ROC AUC across classes with >=1 positive sample
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(probs.shape[1]):
        if mask is not None and not mask[c]: continue
        if labels[:, c].sum() == 0 or labels[:, c].sum() == len(labels): continue
        try:
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


t0_total = time.time()
for epoch in range(start_epoch, N_EPOCHS):
    t_ep = time.time()
    model.train()
    tr_loss = 0.0
    n_batches = 0
    for batch_idx, (wav, lb) in enumerate(tr_loader):
        wav, lb = wav.to(device, non_blocking=True), lb.to(device, non_blocking=True)
        # Mel-spec
        with torch.no_grad():
            mel = mel_transform(wav)             # (B, 256, T_mel)
            mel = mel.unsqueeze(1)               # (B, 1, 256, T_mel) for EffNet
        # SpecAugment (train only)
        if spec_aug is not None:
            mel = spec_aug(mel)
        # Mixup
        if USE_MIXUP and np.random.rand() < 0.5:
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            idx = torch.randperm(mel.size(0))
            mel = lam * mel + (1 - lam) * mel[idx]
            lb = lam * lb + (1 - lam) * lb[idx]
        # Forward
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = model(pixel_values=mel).logits   # (B, 234)
            loss = F.binary_cross_entropy_with_logits(out, lb)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        tr_loss += loss.item(); n_batches += 1
        if (batch_idx + 1) % 100 == 0:
            print(f"  ep{epoch+1} [{batch_idx+1}/{len(tr_loader)}] loss={loss.item():.4f}")
    tr_loss /= max(n_batches, 1)

    # Val
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for wav, lb in val_loader:
            wav = wav.to(device)
            mel = mel_transform(wav).unsqueeze(1)
            out = model(pixel_values=mel).logits
            val_probs.append(torch.sigmoid(out).cpu().numpy())
            val_labels.append(lb.numpy())
    val_probs = np.concatenate(val_probs); val_labels = np.concatenate(val_labels)
    val_auc = compute_macro_auc(val_probs, val_labels)

    ep_time = time.time() - t_ep
    history.append({"epoch": epoch + 1, "tr_loss": tr_loss, "val_auc": val_auc, "time_min": ep_time / 60})
    print(f"Epoch {epoch+1}/{N_EPOCHS}  tr_loss={tr_loss:.4f}  val_auc={val_auc:.4f}  {ep_time/60:.1f}min")

    # Save
    state = {
        "epoch": epoch + 1, "best_val_auc": max(best_val_auc, val_auc),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "config": {"NUM_CLASSES": NUM_CLASSES, "SR": SR, "N_MELS": N_MELS,
                   "HOP_LENGTH": HOP_LENGTH, "N_FFT": N_FFT, "NORM_MEAN": NORM_MEAN, "NORM_STD": NORM_STD},
    }
    torch.save(state, str(CKPT_LATEST_PATH))
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(state, str(CKPT_NS22_PATH))
        print(f"  ★ new best val_auc={best_val_auc:.4f} saved")
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

print(f"\\n==== Training done in {(time.time()-t0_total)/60:.1f} min ====")
print(f"Best val_auc: {best_val_auc:.4f}")
"""))

# ── Upload to Kaggle Dataset ──
cells.append(code_cell("upload", r"""# Upload to Kaggle Dataset
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate (KGAT token from kaggle.json key field)
import json as _json, os as _os
KAGGLE_CRED = Path("/root/.kaggle/kaggle.json")
if not KAGGLE_CRED.exists():
    KAGGLE_CRED.parent.mkdir(parents=True, exist_ok=True)
    # User must have uploaded kaggle.json to /content first
    if Path("/content/kaggle.json").exists():
        shutil.copy("/content/kaggle.json", KAGGLE_CRED)
        _os.chmod(KAGGLE_CRED, 0o600)
_creds = _json.loads(KAGGLE_CRED.read_text())
if _creds.get("key", "").startswith("KGAT_"):
    _os.environ["KAGGLE_API_TOKEN"] = _creds["key"]

api = KaggleApi(); api.authenticate()

USER = "maekeso"
DATASET_SLUG = f"{USER}/birdclef2026-exp041-birdset-b1-weights"
UPLOAD_DIR = Path("/content/exp041_upload")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Copy ckpts + BirdSet config.json (for inference NB rebuild) to upload dir
for f in [CKPT_NS22_PATH, CKPT_LATEST_PATH, HISTORY_PATH]:
    if f.exists():
        shutil.copy(f, UPLOAD_DIR / f.name)
# Copy BirdSet config.json (needed at inference to rebuild EfficientNetConfig)
_birdset_cfg = BIRDSET_DIR / "config.json"
if _birdset_cfg.exists():
    shutil.copy(_birdset_cfg, UPLOAD_DIR / "birdset_config.json")
    print(f"  copied birdset_config.json")

# Create or update Dataset
metadata = {
    "title": "birdclef2026 exp041 BirdSet EffNet-B1 weights",
    "id": DATASET_SLUG,
    "licenses": [{"name": "CC0-1.0"}],
}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

try:
    # Try create first
    result = api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, quiet=False)
    print(f"NEW dataset created: {result}")
except Exception as e:
    if "already exists" in str(e) or "403" in str(e):
        print(f"Dataset exists, uploading new version...")
        result = api.dataset_create_version(
            folder=str(UPLOAD_DIR),
            version_notes=f"exp041 BirdSet B1 fine-tune val_auc={best_val_auc:.4f}",
            quiet=False, dir_mode="zip",
        )
        print(f"Version uploaded: {result}")
    else:
        print(f"upload err: {e}")
"""))

# ── Auto-disconnect (save compute credits) ──
cells.append(code_cell("disconnect", r"""# Auto-disconnect Colab runtime
print("Training complete. Disconnecting runtime in 30 sec...")
import time
time.sleep(30)
from google.colab import runtime
runtime.unassign()
"""))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"name": "exp041_train_birdset_b1.ipynb", "provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_train_birdset_b1.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path.name}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
