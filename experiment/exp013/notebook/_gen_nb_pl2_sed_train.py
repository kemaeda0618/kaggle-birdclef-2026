"""Generate exp013 NB2: SED student 1 fold training on Colab Pro A100.

Babych R1 noisy student の学生学習部。

入力:
- BC2026 competition data (train_audio + train_soundscapes + train.csv + taxonomy.csv)
  → kaggle competitions download で取得
- pseudo_labels.csv (NB1 出力)
  → kaggle datasets download maekeso/birdclef2026-exp013-pseudo-r1

学習:
- EffNet-B0 SED (Tucker と同アーキ、ImageNet 初期化、scratch 学習)
- データ: hard (train_audio, primary+secondary multi-hot) + soft (train_soundscapes pseudo)
- Loss: BCE 1 つで両方処理
- 30 epoch, 1 fold (= 全データ train、val なし、固定 epoch)
- Aug: SpecAug、focal mixup
- Perch distillation は使わない (R1 最小構成)

出力:
- /content/drive/MyDrive/birdclef2026/exp013/student_sed_fold0.onnx
- /content/drive/MyDrive/birdclef2026/exp013/student_sed_fold0.pth
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src,
    }


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": src,
    }


cells = []

cells.append(md_cell("hdr", """# exp013 NB2: SED Student 1 fold training (Babych R1)

Colab Pro A100 上で、NB1 が生成した pseudo_labels.csv を soft target として学生 SED を学習。

## 入力
| Source | 内容 | ラベル |
|---|---|---|
| BC2026 train_audio | ~46k focal recordings | hard (primary + secondary multi-hot) |
| BC2026 train_soundscapes | 10,658 files × 12 windows | soft (NB1 pseudo CSV) |

## 構成
- Backbone: EfficientNet-B0 (timm, ImageNet pretrained)
- SED head: per-frame attention pooling + clip + frame heads
- Mel: n_mels=256, n_fft=2048, hop=512 (Tucker spec)
- Training: 30 epoch, batch=64, AdamW lr=5e-4, cosine schedule
- Loss: BCE (clip + frame_max 平均)
- Aug: SpecAug + focal-focal mixup

## 出力 (Drive 経由)
- `student_sed_fold0.onnx` — NB3 が読み込む推論用
- `student_sed_fold0.pth` — checkpoint (再学習用)

## 想定時間 (Colab Pro A100)
- データ DL (BC2026 25GB): ~15-20 min
- pseudo CSV DL (393 MB): ~1 min
- 学習 30 epoch: ~60-90 min
- ONNX export + Drive 保存: ~2 min
- **合計: ~1.5-2h**
"""))

# --------------------------------------------------------------------------- #
# Cell 1: Setup (Drive mount, kaggle.json)
# --------------------------------------------------------------------------- #
cells.append(code_cell("setup", '''# === Cell 1: Mount Drive + setup auth ===
from google.colab import drive
drive.mount("/content/drive")

import os, json
from pathlib import Path

# kaggle.json を Drive から /root/.kaggle/ に配置
KAGGLE_JSON_DRIVE = Path("/content/drive/MyDrive/kaggle.json")
assert KAGGLE_JSON_DRIVE.exists(), \\
    f"Place kaggle.json at {KAGGLE_JSON_DRIVE} first (with KGAT_ token in 'key')"
os.makedirs("/root/.kaggle", exist_ok=True)
import shutil
shutil.copy(str(KAGGLE_JSON_DRIVE), "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)
# Kaggle SDK 認証 (KGAT_ トークン経由)
_kgat = json.loads(KAGGLE_JSON_DRIVE.read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat

# Drive 上の出力ディレクトリ
OUTPUT_DRIVE = Path("/content/drive/MyDrive/birdclef2026/exp013")
OUTPUT_DRIVE.mkdir(parents=True, exist_ok=True)
print(f"Output Drive dir: {OUTPUT_DRIVE}")

# データ DL 先 (Colab ローカル)
DATA_DIR = Path("/content/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
'''))

# --------------------------------------------------------------------------- #
# Cell 2: Install + Download data
# --------------------------------------------------------------------------- #
cells.append(code_cell("install-dl", '''# === Cell 2: Install deps + download Kaggle data ===
!pip install -q kaggle timm torchaudio onnx onnxscript

import time
T0 = time.time()
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# 1. BC2026 competition data (25GB)
COMP_DIR = DATA_DIR / "birdclef-2026"
if not (COMP_DIR / "taxonomy.csv").exists():
    print("Downloading BC2026 competition data (25GB, ~15-20 min)...")
    api.competition_download_files("birdclef-2026", path=str(DATA_DIR), quiet=False)
    # Unzip
    !cd /content/data && unzip -q birdclef-2026.zip -d birdclef-2026
    !rm /content/data/birdclef-2026.zip
    print(f"  done in {(time.time()-T0)/60:.1f} min")
else:
    print(f"BC2026 already at {COMP_DIR}")

# 2. pseudo_labels.csv (NB1 出力、393 MB)
PSEUDO_DIR = DATA_DIR / "pseudo-r1"
if not (PSEUDO_DIR / "pseudo_labels.csv").exists():
    print("Downloading pseudo_labels.csv (393 MB, ~1 min)...")
    api.dataset_download_files("maekeso/birdclef2026-exp013-pseudo-r1",
                                path=str(PSEUDO_DIR), unzip=True, quiet=False)
    print(f"  done in {(time.time()-T0)/60:.1f} min")
else:
    print(f"Pseudo labels already at {PSEUDO_DIR}")

print(f"\\nDownload phase total: {(time.time()-T0)/60:.1f} min")
print(f"Disk usage:")
!du -sh /content/data/*
'''))

# --------------------------------------------------------------------------- #
# Cell 3: Imports + Config
# --------------------------------------------------------------------------- #
cells.append(code_cell("imports-cfg", '''# === Cell 3: Imports + Config ===
import os, sys, time, gc, random, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import timm
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

# Paths
COMP_DIR    = DATA_DIR / "birdclef-2026"
PSEUDO_CSV  = DATA_DIR / "pseudo-r1" / "pseudo_labels.csv"
TRAIN_AUDIO = COMP_DIR / "train_audio"
TRAIN_SC    = COMP_DIR / "train_soundscapes"

# Audio / mel
SR          = 32000
WINDOW_SEC  = 5.0
WINDOW_SAMP = int(WINDOW_SEC * SR)
N_MELS      = 256
N_FFT       = 2048
HOP         = 512
FMIN        = 20
FMAX        = 16000

# Model
BACKBONE    = "tf_efficientnet_b0.ns_jft_in1k"   # Tucker と同じ

# Training
EPOCHS       = 30
BATCH        = 64
LR           = 5e-4
MIN_LR       = 1e-5
WD           = 1e-4
WARMUP_EPOCH = 2
NUM_WORKERS  = 8

# Source mixing (per epoch、シャッフル後の比率)
HARD_SHARE   = 0.6   # train_audio (hard) の割合
SOFT_SHARE   = 0.4   # train_soundscapes pseudo (soft) の割合

# Mixup (focal vs focal のみ、シンプル化)
MIXUP_PROB   = 0.5
MIXUP_ALPHA  = 0.4

# SpecAug
FREQ_MASK_PARAM = 16
TIME_MASK_PARAM = 16
NUM_FREQ_MASKS  = 2
NUM_TIME_MASKS  = 2

# Augment
AUG_GAIN_DB_RANGE = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)
AUG_PROB     = 0.5

NUM_CLASSES  = 234

print(f"Backbone: {BACKBONE}")
print(f"Epochs: {EPOCHS}, batch: {BATCH}, lr: {LR}")
print(f"Source mix: hard={HARD_SHARE}, soft={SOFT_SHARE}")
'''))

# --------------------------------------------------------------------------- #
# Cell 4: Load taxonomy + meta
# --------------------------------------------------------------------------- #
cells.append(code_cell("load-meta", '''# === Cell 4: Load taxonomy + train_audio meta + pseudo CSV ===

# Taxonomy: 234 species (sample_submission のカラム順 = primary_label の順)
sample_sub = pd.read_csv(COMP_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"Classes: {NUM_CLASSES}")

# train_audio meta (hard label)
train_df = pd.read_csv(COMP_DIR / "train.csv")
train_df = train_df[train_df["primary_label"].astype(str).isin(LABEL2IDX)].reset_index(drop=True)
train_df["filepath"] = train_df["filename"].apply(lambda f: str(TRAIN_AUDIO / f))
train_df["primary_idx"] = train_df["primary_label"].astype(str).map(LABEL2IDX)
print(f"train_audio rows: {len(train_df)}")

def parse_secondary(s):
    """secondary_labels (string of list) → list of label strings."""
    if pd.isna(s) or not isinstance(s, str):
        return []
    s = s.strip().strip("[]").strip()
    if not s:
        return []
    return [x.strip().strip("'\\"") for x in s.split(",") if x.strip()]

train_df["secondary_list"] = train_df["secondary_labels"].apply(parse_secondary)

# pseudo CSV (soft labels on train_soundscapes)
print("Loading pseudo CSV (393 MB)...")
t0 = time.time()
pseudo_df = pd.read_csv(PSEUDO_CSV)
print(f"  loaded {len(pseudo_df)} rows in {time.time()-t0:.1f}s")
print(f"  cols: {pseudo_df.columns[:6].tolist()} ... ({pseudo_df.shape[1]} total)")

# Pseudo label cols は PRIMARY_LABELS と同じ順
pseudo_label_cols = pseudo_df.columns[3:].tolist()
assert pseudo_label_cols == PRIMARY_LABELS, \\
    f"label order mismatch: {pseudo_label_cols[:5]} vs {PRIMARY_LABELS[:5]}"

# 各 pseudo row の filepath を construct
pseudo_df["filepath"] = pseudo_df["filename"].apply(
    lambda f: str(TRAIN_SC / (f if f.endswith(".ogg") else f + ".ogg"))
)
print(f"Sample pseudo path: {pseudo_df['filepath'].iloc[0]}")
assert Path(pseudo_df["filepath"].iloc[0]).exists(), "Pseudo file not found"
'''))

# --------------------------------------------------------------------------- #
# Cell 5: Dataset classes
# --------------------------------------------------------------------------- #
cells.append(code_cell("dataset", '''# === Cell 5: Dataset classes ===

def read_audio_window(filepath, start_sec=None, n_samples=WINDOW_SAMP):
    """Read .ogg, return mono float32 of length n_samples.
    If start_sec is None, randomly sample. Else use that exact start."""
    info = sf.info(filepath)
    total_samples = int(info.duration * info.samplerate)
    sr = info.samplerate

    if start_sec is None:
        # Random crop from focal recording
        if total_samples > n_samples:
            start = random.randint(0, total_samples - n_samples)
        else:
            start = 0
    else:
        start = int(start_sec * sr)

    y, _ = sf.read(filepath, start=start, frames=n_samples, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        # Rare in this competition but defensive
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    return y.astype(np.float32)


class HardDataset(Dataset):
    """train_audio: hard multi-hot from primary + secondary."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            wav = read_audio_window(row["filepath"], start_sec=None)
        except Exception:
            wav = np.zeros(WINDOW_SAMP, dtype=np.float32)

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        label[row["primary_idx"]] = 1.0
        for sl in row["secondary_list"]:
            if sl in LABEL2IDX:
                label[LABEL2IDX[sl]] = 1.0

        return torch.from_numpy(wav), torch.from_numpy(label)


class SoftDataset(Dataset):
    """train_soundscapes: soft labels from pseudo CSV."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.label_mat = df[PRIMARY_LABELS].values.astype(np.float32)
        self.start_secs = df["start_sec"].values
        self.filepaths = df["filepath"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            wav = read_audio_window(self.filepaths[idx],
                                    start_sec=self.start_secs[idx],
                                    n_samples=WINDOW_SAMP)
        except Exception:
            wav = np.zeros(WINDOW_SAMP, dtype=np.float32)
        label = self.label_mat[idx]
        return torch.from_numpy(wav), torch.from_numpy(label)


class MixedDataset(Dataset):
    """Hard + Soft をシャッフルして 1 つの dataset に。
    Hard と Soft の比率は HARD_SHARE / SOFT_SHARE で決まる。
    epoch ごとに sampling し直すために、長さを 2 つの max に揃える。"""
    def __init__(self, hard_df, soft_df):
        self.hard = HardDataset(hard_df)
        self.soft = SoftDataset(soft_df)
        # 1 epoch あたりのサンプル数を決める
        self.n_total = len(self.hard) + len(self.soft)
        self.hard_n = int(self.n_total * HARD_SHARE)
        self.soft_n = self.n_total - self.hard_n

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        if idx < self.hard_n:
            return self.hard[idx % len(self.hard)]
        else:
            return self.soft[(idx - self.hard_n) % len(self.soft)]


train_ds = MixedDataset(train_df, pseudo_df)
print(f"Train dataset: {len(train_ds)} samples (hard {train_ds.hard_n} + soft {train_ds.soft_n})")
'''))

# --------------------------------------------------------------------------- #
# Cell 6: Model
# --------------------------------------------------------------------------- #
cells.append(code_cell("model", '''# === Cell 6: Model (EffNet-B0 SED) ===

class MelSpecTransform(nn.Module):
    """Waveform -> mel spectrogram (dB) -> per-sample standardize.
    Tucker の推論 NB と同じ標準化 (mean=0, std=1 per spectrogram)。
    これを忘れると EffNet-B0 で活性が爆発し loss=NaN になる。"""
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, wav):
        # wav: (B, T)
        s = self.db(self.mel(wav))                                    # (B, n_mels, T)
        m = s.mean(dim=(1, 2), keepdim=True)
        v = s.std(dim=(1, 2), keepdim=True)
        s = (s - m) / (v + 1e-6)
        return s


class SpecAug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fm = nn.ModuleList([torchaudio.transforms.FrequencyMasking(FREQ_MASK_PARAM)
                                 for _ in range(NUM_FREQ_MASKS)])
        self.tm = nn.ModuleList([torchaudio.transforms.TimeMasking(TIME_MASK_PARAM)
                                 for _ in range(NUM_TIME_MASKS)])

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        for m in self.fm: x = m(x)
        for m in self.tm: x = m(x)
        return x


class AttBlock(nn.Module):
    """Attention pooling block for SED."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.att = nn.Conv1d(in_features, out_features, 1)
        self.cla = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x):
        # x: (B, C, T)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)   # (B, n_class, T)
        cla = self.cla(x)                                            # (B, n_class, T)
        x_clip = (norm_att * cla).sum(dim=2)                         # (B, n_class)
        return x_clip, cla


class SEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True,
                                           in_chans=1, num_classes=0,
                                           global_pool="")
        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, N_MELS, 313)
            feat = self.backbone(dummy)
            assert feat.dim() == 4, f"backbone output dim={feat.dim()}"
            self.feat_dim = feat.shape[1]
            self.feat_T   = feat.shape[3]
            print(f"Backbone feat: dim={self.feat_dim}, T={self.feat_T}")
        # Frequency pool: mean over freq dim
        self.fc1 = nn.Linear(self.feat_dim, 512)
        self.bn  = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.att_block = AttBlock(512, num_classes)

    def forward(self, mel):
        # mel: (B, 1, n_mels, T)
        h = self.backbone(mel)              # (B, C, F, T_b)
        h = h.mean(dim=2)                    # (B, C, T_b)  freq mean
        h = h.permute(0, 2, 1)               # (B, T_b, C)
        h = self.fc1(h)                      # (B, T_b, 512)
        h = self.bn(h.permute(0, 2, 1))      # (B, 512, T_b)
        h = F.relu(h)
        h = self.dropout(h)
        x_clip, cla = self.att_block(h)     # (B, n_class), (B, n_class, T_b)
        return x_clip, cla


# Sanity check
torch.cuda.empty_cache()
model = SEDModel().to(device)
mel_xform = MelSpecTransform().to(device)
spec_aug = SpecAug().to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

with torch.no_grad():
    dummy = torch.randn(2, WINDOW_SAMP, device=device)
    mel = mel_xform(dummy).unsqueeze(1)
    print(f"Mel shape: {mel.shape}")
    clip, cla = model(mel)
    print(f"Clip: {clip.shape}, frame: {cla.shape}")
'''))

# --------------------------------------------------------------------------- #
# Cell 7: Training
# --------------------------------------------------------------------------- #
cells.append(code_cell("train", '''# === Cell 7: Training loop ===

def random_gain_noise(wav):
    """Apply random gain + Gaussian noise."""
    if random.random() < AUG_PROB:
        gain_db = random.uniform(*AUG_GAIN_DB_RANGE)
        wav = wav * (10 ** (gain_db / 20))
    if random.random() < AUG_PROB:
        snr_db = random.uniform(*AUG_NOISE_SNR_DB_RANGE)
        sig_pow = (wav ** 2).mean(dim=-1, keepdim=True)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        noise = torch.randn_like(wav) * noise_pow.sqrt()
        wav = wav + noise
    return wav


def mixup_batch(wav, label, alpha=MIXUP_ALPHA):
    """Standard mixup. wav: (B, T), label: (B, C)."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(wav.size(0), device=wav.device)
    wav_mix = lam * wav + (1 - lam) * wav[perm]
    label_mix = lam * label + (1 - lam) * label[perm]
    return wav_mix, label_mix


train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           drop_last=True, persistent_workers=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
n_steps = len(train_loader) * EPOCHS
warmup_steps = len(train_loader) * WARMUP_EPOCH
def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return MIN_LR / LR + (1 - MIN_LR / LR) * 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = GradScaler()
bce = nn.BCEWithLogitsLoss()

print(f"Total steps: {n_steps} (warmup {warmup_steps}), {len(train_loader)} steps/epoch")

T_train = time.time()
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_n = 0
    t0 = time.time()
    for i, (wav, label) in enumerate(train_loader):
        wav   = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        wav = random_gain_noise(wav)
        if random.random() < MIXUP_PROB:
            wav, label = mixup_batch(wav, label)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            mel = mel_xform(wav).unsqueeze(1)
            mel = spec_aug(mel)
            clip, cla = model(mel)
            loss_clip = bce(clip, label)
            # frame-max BCE (SED 学習で重要な信号)
            frame_max = cla.max(dim=2).values
            loss_frame = bce(frame_max, label)
            loss = 0.5 * loss_clip + 0.5 * loss_frame

        # NaN guard: skip the step but advance scheduler
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        scaler.scale(loss).backward()
        # Gradient clipping for stability with autocast + mel
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item() * wav.size(0)
        epoch_n += wav.size(0)

    avg_loss = epoch_loss / epoch_n
    elapsed = time.time() - t0
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={avg_loss:.4f}, lr={cur_lr:.2e}, "
          f"{elapsed:.0f}s ({len(train_loader)/elapsed:.1f} steps/s)")

    # Save ckpt every 5 epochs (for safety)
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        ckpt_path = OUTPUT_DRIVE / f"student_sed_fold0_ep{epoch+1:02d}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch+1, "loss": avg_loss},
                   str(ckpt_path))
        print(f"    saved ckpt: {ckpt_path}")

print(f"\\nTraining done: {(time.time()-T_train)/60:.1f} min")
final_ckpt = OUTPUT_DRIVE / "student_sed_fold0_final.pth"
torch.save({"model": model.state_dict(), "epoch": EPOCHS, "loss": avg_loss},
           str(final_ckpt))
print(f"Final ckpt: {final_ckpt}")
'''))

# --------------------------------------------------------------------------- #
# Cell 8: ONNX export
# --------------------------------------------------------------------------- #
cells.append(code_cell("export", '''# === Cell 8: ONNX export (Tucker と同 IO で blend に流用可) ===

class SEDInferModel(nn.Module):
    """Wrap the model with mel computation for ONNX (input = mel directly,
    matching Tucker's ONNX which takes mel as input)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        # mel: (B, 1, N_MELS, T) — already log-mel + standardized
        clip, cla = self.model(mel)
        # Tucker's framewise output: (B, T_out, N_CLASSES) — transpose
        cla_t = cla.permute(0, 2, 1)   # (B, T_out, n_class)
        return clip, cla_t


model.eval()
infer = SEDInferModel(model).to(device).eval()

# Tucker と同じ入力 shape: (B, 1, 256, 313) for 5-sec window @ 32 kHz
T_mel_5sec = 313
dummy = torch.randn(1, 1, N_MELS, T_mel_5sec, device=device)
with torch.no_grad():
    out_clip, out_frame = infer(dummy)
    print(f"Test export: clip={out_clip.shape}, frame={out_frame.shape}")

onnx_path = OUTPUT_DRIVE / "student_sed_fold0.onnx"
torch.onnx.export(
    infer, dummy, str(onnx_path),
    input_names=["mel"], output_names=["clip_logits", "framewise"],
    dynamic_axes={
        "mel":         {0: "batch"},
        "clip_logits": {0: "batch"},
        "framewise":   {0: "batch"},
    },
    opset_version=17,
)
print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")

# Verify with onnxruntime
!pip install -q onnxruntime
import onnxruntime as ort
so = ort.SessionOptions()
sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                             providers=["CPUExecutionProvider"])
mel_np = dummy.cpu().numpy()
outs = sess.run(None, {"mel": mel_np})
print(f"ORT verify: clip={outs[0].shape}, frame={outs[1].shape}")
print(f"All outputs in OUTPUT_DRIVE:")
for f in sorted(OUTPUT_DRIVE.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB")
'''))

# --------------------------------------------------------------------------- #
# Cell 9: Optional - upload ONNX as Kaggle Dataset
# --------------------------------------------------------------------------- #
cells.append(code_cell("upload-kg", '''# === Cell 9 (任意): ONNX を Kaggle Dataset として upload ===
# 失敗してもいい。最悪手動で Drive から DL → Kaggle に upload して OK。
import tempfile

UPLOAD_SLUG = "birdclef2026-exp013-r1-student-sed"
UPLOAD_TITLE = "BirdCLEF2026 exp013 R1 Student SED"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(str(onnx_path), str(td / "student_sed_fold0.onnx"))
    # External data ファイル (.onnx.data) があれば一緒に upload
    # (PyTorch の onnx.export で大きいモデルは external data 形式になる)
    onnx_data = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if onnx_data.exists():
        shutil.copy(str(onnx_data), str(td / onnx_data.name))
        print(f"  + external data: {onnx_data.name} ({onnx_data.stat().st_size/1024/1024:.1f} MB)")
    meta = {
        "title": UPLOAD_TITLE,
        "id": f"maekeso/{UPLOAD_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        api.dataset_create_new(str(td), public=False, dir_mode="zip")
        print(f"Created: maekeso/{UPLOAD_SLUG}")
    except Exception as e:
        print(f"create_new failed (maybe exists): {e}")
        print("Trying version update...")
        try:
            api.dataset_create_version(str(td), version_notes="R1 1-fold student",
                                        dir_mode="zip")
            print(f"Updated: maekeso/{UPLOAD_SLUG}")
        except Exception as e2:
            print(f"update also failed: {e2}")
            print("Manual upload required: download ONNX from Drive, upload to Kaggle.")
'''))

# --------------------------------------------------------------------------- #
# Notebook assembly
# --------------------------------------------------------------------------- #
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"name": "exp013_nb_pl2_sed_train.ipynb", "provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_pl2_sed_train.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
