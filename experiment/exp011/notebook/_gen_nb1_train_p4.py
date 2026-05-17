"""Generate exp011 Phase 4 training notebook (= Phase 2 + raw waveform mixup).

Phase 4 scope (1 logical change from Phase 2):
- mel cache 直読 -> **ogg 直読 + GPU MelSpectrogram**
- spec mixup (alpha=0.5, fixed lambda) -> **raw waveform mixup (alpha=0.5, Beta sampling)**

Preserved from Phase 2 (Phase 1 v2 と同様):
- BCE clipwise + framewise max (NOT Soft CE — see feedback_soft_ce_needs_raw_mixup.md)
- target_size (256, 256) (Phase 5 で 256x384 化予定)
- chunk_duration 20s
- Backbone tf_efficientnetv2_b0 + GeMFreqPool + AttentionSEDHead
- AdamW lr=5e-4, CosineAnnealing, 20 epochs

Val 戦略 (Phase 3 教訓 feedback_small_val_holdout.md):
- Val-B = train_audio author 10% hold-out (~3500 windows) -> **primary**, early stop, best.pth に保存
- Val-A = labeled SS 16 files hold-out (~192 windows)     -> secondary, best_val_a.pth に保存

GPU: T4x2 (実質単 GPU). num_workers=4 with smoke test (legacy step1 で動作実績).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

# Header
cells.append(md_cell("hdr", """# exp011 Phase 4: Phase 2 + raw waveform mixup

1 Phase = 1 変更ルール (`feedback_isolate_changes.md`) に従い、Phase 2 v2 (LB 0.854) から **mixup を spec mixup -> raw waveform mixup** に変更。これに伴い入力経路も mel cache 直読 -> ogg 直読 + GPU MelSpectrogram になる (raw mixup 実現に必要な付随変更)。

## 差分 (Phase 2 v2 -> Phase 4)
| 項目 | Phase 2 v2 | Phase 4 |
|---|---|---|
| 入力 | mel cache (uint8 npy) 直読 | **ogg 直読 + GPU MelSpectrogram** |
| Mixup | spec mixup (fixed lambda=0.5) | **raw waveform mixup (Beta(0.5, 0.5) lambda)** |
| Loss | BCE clip+frame | BCE clip+frame (**変更なし**) |
| target_size | (256, 256) | (256, 256) (Phase 5 で変更予定) |
| chunk_duration | 20s | 20s |
| 早期停止基準 | Val-A best (best.pth) + Val-B best (best_val_b.pth) | **Val-B primary (best.pth)** + Val-A secondary (best_val_a.pth) |

## 副次効果
学習時の量子化誤差 ([-80, 20] dB -> uint8 -> 復元) が消え、推論側のリアル mel パイプラインと完全一致する。

## Goal
LB **0.86-0.88** (+0.005~0.025 vs Phase 2 v2 の 0.854)。Phase 5 (256x384) の伸び代を残す。

## 注意
旧 Phase 2 (Step 1) は同要素を含む 4 要素同時投入で LB 0.819 (-0.035)。Phase 4 は他 3 要素 (Soft CE, target_size 384, num_workers 4) を抜いて raw mixup の純粋寄与を見る。
"""))

# Imports
cells.append(code_cell("imports", r"""!pip install -q timm torchaudio scikit-learn

import os, gc, ast, glob, time, random, warnings, json, hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision
import timm

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPUs: {torch.cuda.device_count()}")
WALL_START = time.time()"""))

# Config
cells.append(code_cell("config", r"""# ==============================================================
# CONFIG (Phase 4 = Phase 2 + raw mixup)
# ==============================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # Audio / Mel (matches Phase 2 spec exactly)
    sr: int = 32_000
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 16_000
    top_db: float = 80.0
    db_min: float = -80.0
    db_max: float = 20.0

    # Chunk: 20 sec window (Phase 2 と同じ)
    chunk_duration: float = 20.0
    target_size: tuple = (256, 256)  # 据え置き (Phase 5 で 256x384 化)

    # Model
    backbone: str = "tf_efficientnetv2_b0"
    pretrained: bool = True
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0

    # Training
    epochs: int = 20
    batch_size: int = 32     # Phase 2 と同じ
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    num_workers: int = 4     # ogg loading I/O bound (legacy step1 で動作実績)

    # Augmentation (raw mixup へ切替)
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.5  # Beta(alpha, alpha) lambda
    freq_mask_param: int = 30
    time_mask_param: int = 40

    # Loss weights (BCE clip + frame)
    clip_loss_weight: float = 0.5
    frame_loss_weight: float = 0.5

    # Data
    seed: int = 42
    val_b_ratio: float = 0.10  # train_audio author hold-out (= primary)
    val_a_n_files: int = 16    # labeled SS 66 files のうち hold-out 数 (50 を訓練に残す)
    use_secondary_labels: bool = True
    include_soundscape_labels: bool = True

    # Paths
    data_root: str = "/kaggle/input/competitions/birdclef-2026"
    output_dir: str = "/kaggle/working"

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_duration * self.sr)  # 640000 for 20s

    @property
    def chunk_frames(self) -> int:
        return int(self.chunk_duration * self.sr / self.hop_length) + 1  # 1251

    @property
    def db_range(self) -> float:
        return self.db_max - self.db_min


cfg = Config()
set_seed(cfg.seed)
print(f"Chunk: {cfg.chunk_duration}s = {cfg.chunk_samples} samples = {cfg.chunk_frames} mel frames")
print(f"Target size (H, W): {cfg.target_size}")
print(f"Backbone: {cfg.backbone}, Epochs: {cfg.epochs}, Batch: {cfg.batch_size}, LR: {cfg.lr}")
print(f"Val-A (labeled SS hold-out): {cfg.val_a_n_files} files")
print(f"Val-B (train_audio author hold-out): {cfg.val_b_ratio*100:.0f}% (= primary)")"""))

# Paths
cells.append(code_cell("paths", r"""# ==============================================================
# PATHS & SPECIES (autodetect Kaggle mount points)
# ==============================================================
DATA_ROOT = None
for cand in [Path("/kaggle/input/competitions/birdclef-2026"),
             Path("/kaggle/input/birdclef-2026")]:
    if cand.exists():
        DATA_ROOT = cand; break
assert DATA_ROOT is not None, "birdclef-2026 not mounted"
cfg.data_root = str(DATA_ROOT)
print(f"DATA_ROOT: {DATA_ROOT}")

TRAIN_CSV = DATA_ROOT / "train.csv"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"
SC_LABELS_CSV = DATA_ROOT / "train_soundscapes_labels.csv"
TRAIN_AUDIO_DIR = DATA_ROOT / "train_audio"
TRAIN_SC_DIR = DATA_ROOT / "train_soundscapes"

WEIGHT_DIR = Path(cfg.output_dir) / "weights"
LOG_DIR = Path(cfg.output_dir) / "logs"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

assert TRAIN_AUDIO_DIR.exists(), f"train_audio not found at {TRAIN_AUDIO_DIR}"
assert TRAIN_SC_DIR.exists(), f"train_soundscapes not found at {TRAIN_SC_DIR}"
print(f"train_audio: {TRAIN_AUDIO_DIR}")
print(f"train_soundscapes: {TRAIN_SC_DIR}")

sub_df = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
SPECIES = list(sub_df.columns[1:])
SPECIES_TO_IDX = {sp: i for i, sp in enumerate(SPECIES)}
NUM_CLASSES = len(SPECIES)
print(f"Species: {NUM_CLASSES}")"""))

# Audio utils
cells.append(code_cell("audio-utils", r"""# ==============================================================
# AUDIO LOADING (ogg direct, on-the-fly resample, mono)
# ==============================================================
def load_audio_full(path):
    # Returns 1D numpy float32 at cfg.sr
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != cfg.sr:
        audio = torchaudio.functional.resample(audio, sr, cfg.sr)
    return audio.squeeze(0).numpy().astype(np.float32)


def take_chunk_random(audio, n_samples):
    if len(audio) >= n_samples:
        start = np.random.randint(0, len(audio) - n_samples + 1)
        return audio[start:start + n_samples]
    out = np.zeros(n_samples, dtype=np.float32)
    out[:len(audio)] = audio
    return out


def take_chunk_at(audio, start_sample, n_samples):
    # Anchored crop, pad zeros if out of range. start_sample may be negative.
    out = np.zeros(n_samples, dtype=np.float32)
    s = max(0, start_sample)
    e = min(len(audio), start_sample + n_samples)
    if s < e:
        out_off = s - start_sample
        out[out_off:out_off + (e - s)] = audio[s:e]
    return out


def take_chunk_center(audio, n_samples):
    if len(audio) >= n_samples:
        start = (len(audio) - n_samples) // 2
        return audio[start:start + n_samples]
    out = np.zeros(n_samples, dtype=np.float32)
    pad = (n_samples - len(audio)) // 2
    out[pad:pad + len(audio)] = audio
    return out


print("Audio utils ready")"""))

# Mel transform on GPU
cells.append(code_cell("mel-transform", r"""# ==============================================================
# MEL TRANSFORM (GPU): raw waveform -> mel_db -> resize + per-sample norm + 3ch
# Phase 2 inference NB と同じパイプライン (clamp [-80, 20] -> resize -> minmax -> repeat)
# ==============================================================
class MelTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax, power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.resize = torchvision.transforms.Resize(cfg.target_size, antialias=True)
        self.db_min = cfg.db_min
        self.db_max = cfg.db_max

    def forward(self, audio):
        # audio: (B, n_samples)
        with torch.amp.autocast("cuda", enabled=False):
            audio = audio.float()
            mel = self.mel_spec(audio)              # (B, n_mels, T)
            mel_db = self.amp_to_db(mel)
            mel_db = mel_db.clamp(min=self.db_min, max=self.db_max)
            x = self.resize(mel_db.unsqueeze(1)).squeeze(1)  # (B, H, W)
            B = x.shape[0]
            flat = x.reshape(B, -1)
            mn = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            mx = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
            x = (x - mn) / (mx - mn + 1e-7)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return x


class SpecAugmentations(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)

    def forward(self, x):
        return self.time_mask(self.freq_mask(x))


print("MelTransform + SpecAug ready")"""))

# Raw waveform mixup
cells.append(code_cell("raw-mixup", r"""# ==============================================================
# RAW WAVEFORM MIXUP (Beta(alpha, alpha) lambda, label = max OR)
# ==============================================================
class RawMixUp:
    def __init__(self, prob=0.5, alpha=0.5):
        self.prob = prob; self.alpha = alpha

    def __call__(self, audio, labels):
        # audio: (B, n_samples), labels: (B, n_classes)
        if torch.rand(1).item() > self.prob:
            return audio, labels
        idx = torch.randperm(audio.size(0), device=audio.device)
        lam = float(np.random.beta(self.alpha, self.alpha))
        audio_mix = lam * audio + (1.0 - lam) * audio[idx]
        label_mix = torch.max(labels, labels[idx])
        return audio_mix, label_mix


print("RawMixUp ready")"""))

# Model (same as Phase 2)
cells.append(code_cell("model", r"""# ==============================================================
# MODEL: timm v2B0 + GeMFreqPool + AttentionSEDHead (Phase 2 と同じ)
# ==============================================================
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init)); self.eps = eps

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)
        return x


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        clipwise_logit = (att * cls).sum(dim=-1)
        return {
            "clipwise_logit": clipwise_logit,
            "clipwise_prob": torch.sigmoid(clipwise_logit),
            "segmentwise_logit": cls.permute(0, 2, 1),
        }


class SEDModel(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        use_pre = pretrained if pretrained is not None else cfg.pretrained
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=use_pre, in_chans=cfg.in_channels,
            features_only=False, global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(feat_dim, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.gem_pool(feat)
        return self.head(pooled)


_m = SEDModel(cfg, pretrained=False)
print(f"Backbone features: {_m.backbone.num_features}")
print(f"Total params: {sum(p.numel() for p in _m.parameters())/1e6:.2f}M")
del _m"""))

# Loss (BCE clip+frame, same as Phase 2 - NOT Soft CE)
cells.append(code_cell("loss", r"""# ==============================================================
# LOSS: ClipFrameCELoss (BCE on clipwise + framewise max)
# Phase 2 と同じ。Soft CE は Phase 3 v3 LB 0.824 で棄却済 (feedback_soft_ce_needs_raw_mixup.md)
# ==============================================================
class ClipFrameCELoss(nn.Module):
    def __init__(self, clip_weight=0.5, frame_weight=0.5):
        super().__init__()
        self.cw = clip_weight; self.fw = frame_weight

    def forward(self, outputs, targets):
        clip_logit = outputs["clipwise_logit"]
        loss_clip = F.binary_cross_entropy_with_logits(clip_logit, targets)
        frame_max_logit = outputs["segmentwise_logit"].max(dim=1)[0]
        loss_frame = F.binary_cross_entropy_with_logits(frame_max_logit, targets)
        return self.cw * loss_clip + self.fw * loss_frame"""))

# Datasets (raw audio)
cells.append(code_cell("dataset", r"""# ==============================================================
# DATASETS (TrainAudio random 20s, SoundscapeSegment anchored 20s)
# ==============================================================
def _parse_secondary_labels(s):
    if pd.isna(s) or s in ("[]", ""):
        return []
    try:
        parsed = ast.literal_eval(s)
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _parse_time_to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t)
    if ":" in s:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(s)


class TrainAudioDataset(Dataset):
    # train_audio rows. mode='train': random 20s crop, mode='val': center 20s.
    def __init__(self, df, species_to_idx, cfg, mode="train"):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.num_classes = len(species_to_idx)
        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = TRAIN_AUDIO_DIR / row["filename"]
        try:
            audio = load_audio_full(path)
        except Exception:
            audio = np.zeros(self.cfg.chunk_samples, dtype=np.float32)
        if self.mode == "train":
            chunk = take_chunk_random(audio, self.cfg.chunk_samples)
        else:
            chunk = take_chunk_center(audio, self.cfg.chunk_samples)
        label = np.zeros(self.num_classes, dtype=np.float32)
        sp = str(row["primary_label"])
        if sp in self.species_to_idx:
            label[self.species_to_idx[sp]] = 1.0
        if self.cfg.use_secondary_labels:
            for sec in _parse_secondary_labels(row.get("secondary_labels", "[]")):
                if sec in self.species_to_idx:
                    label[self.species_to_idx[sec]] = 1.0
        return torch.from_numpy(chunk).float(), torch.from_numpy(label).float()


class SoundscapeSegmentDataset(Dataset):
    # labeled SS segments. 20s window ending at start_sec+5 (matches inference end_time).
    def __init__(self, segments, cfg):
        self.segments = segments
        self.cfg = cfg

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        path = TRAIN_SC_DIR / seg["filename"]
        try:
            audio = load_audio_full(path)
        except Exception:
            audio = np.zeros(self.cfg.chunk_samples, dtype=np.float32)
        end_sample = int((seg["start_sec"] + 5.0) * self.cfg.sr)
        start_sample = end_sample - self.cfg.chunk_samples
        chunk = take_chunk_at(audio, start_sample, self.cfg.chunk_samples)
        return torch.from_numpy(chunk).float(), torch.from_numpy(seg["label"]).float()


def prepare_sc_segments(sc_labels_df, species_to_idx, allowed_files=None):
    out = []
    nC = len(species_to_idx)
    for _, row in sc_labels_df.iterrows():
        if allowed_files is not None and row["filename"] not in allowed_files:
            continue
        label = np.zeros(nC, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                label[species_to_idx[sp]] = 1.0
        out.append({
            "filename": row["filename"],
            "start_sec": _parse_time_to_seconds(row["start"]),
            "label": label,
        })
    return out


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.cum = np.cumsum(self.lens)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        ds_idx = int(np.searchsorted(self.cum, idx, side="right"))
        local = idx - (self.cum[ds_idx - 1] if ds_idx > 0 else 0)
        return self.datasets[ds_idx][local]


print("Datasets ready")"""))

# Train utilities
cells.append(code_cell("train-utils", r"""# ==============================================================
# TRAIN / VAL UTILITIES (raw audio path: mixup -> mel transform -> spec aug)
# ==============================================================
def train_one_epoch(model, loader, optimizer, scheduler,
                    mel_transform, spec_aug, mixup, loss_fn, scaler, cfg, epoch):
    model.train()
    total = 0.0; n = 0
    for batch_idx, (audio, labels) in enumerate(tqdm(loader, desc="  train", leave=False)):
        audio = audio.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # Raw waveform mixup (Phase 4 唯一の差分)
        audio, labels = mixup(audio, labels)

        # Mel transform on GPU (FP32) + SpecAug
        mel = mel_transform(audio)
        mel = spec_aug(mel)

        with torch.amp.autocast("cuda"):
            outputs = model(mel)
            loss = loss_fn(outputs, labels)
        loss = loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / len(loader))
        total += loss.item() * cfg.grad_accum_steps; n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, mel_transform):
    model.eval()
    pp, tt = [], []
    for audio, labels in tqdm(loader, desc="  valid", leave=False):
        audio = audio.to(DEVICE, non_blocking=True)
        mel = mel_transform(audio)
        with torch.amp.autocast("cuda"):
            out = model(mel)
        pp.append(out["clipwise_prob"].float().cpu().numpy())
        tt.append(labels.numpy())
    if not pp:
        return np.zeros((0,)), np.zeros((0,))
    return np.concatenate(pp), np.concatenate(tt)


def compute_metrics(preds, targets):
    if preds.size == 0 or targets.size == 0:
        return {"macro_auc": 0.0, "num_classes_evaluated": 0}
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except ValueError:
                pass
    return {"macro_auc": float(np.mean(aucs)) if aucs else 0.0,
            "num_classes_evaluated": len(aucs)}


def split_train_audio_by_author(train_df, val_ratio, seed):
    if "author" in train_df.columns and train_df["author"].notna().all():
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["author"]))
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(train_df))
        n_val = max(1, int(len(train_df) * val_ratio))
        va_idx = idx[:n_val]; tr_idx = idx[n_val:]
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)


def split_sc_files(sc_labels_df, n_holdout, seed):
    # Deterministic hash-based selection of hold-out files
    files = sorted(sc_labels_df["filename"].unique().tolist())
    def _h(s):
        return int(hashlib.md5(f"{seed}-{s}".encode()).hexdigest(), 16)
    files_sorted = sorted(files, key=_h)
    holdout = set(files_sorted[:n_holdout])
    train = set(files_sorted[n_holdout:])
    return train, holdout"""))

# Training loop with dual val (Val-B primary)
cells.append(code_cell("train", r"""# ==============================================================
# PHASE 4 TRAINING (single fold + dual val, Val-B primary early stop)
# ==============================================================
train_df_full = pd.read_csv(TRAIN_CSV)
print(f"Train recordings: {len(train_df_full)}")

# Soundscape labeled segments + dual val split
sc_labels = pd.read_csv(SC_LABELS_CSV)
train_sc_files, val_a_files = split_sc_files(sc_labels, cfg.val_a_n_files, cfg.seed)
print(f"SC files: {len(train_sc_files)} train + {len(val_a_files)} Val-A")

sc_segs_train = prepare_sc_segments(sc_labels, SPECIES_TO_IDX, allowed_files=train_sc_files)
sc_segs_val_a = prepare_sc_segments(sc_labels, SPECIES_TO_IDX, allowed_files=val_a_files)
print(f"SC segments: {len(sc_segs_train)} train + {len(sc_segs_val_a)} Val-A")

# train_audio author split (Val-B = primary)
tr_df, vb_df = split_train_audio_by_author(train_df_full, cfg.val_b_ratio, cfg.seed)
print(f"train_audio: {len(tr_df)} train + {len(vb_df)} Val-B (author hold-out, primary)")

# Datasets (raw audio path)
train_ds_audio = TrainAudioDataset(tr_df, SPECIES_TO_IDX, cfg, mode="train")
train_ds_sc = SoundscapeSegmentDataset(sc_segs_train, cfg)
train_ds = ConcatDataset(train_ds_audio, train_ds_sc)

val_a_ds = SoundscapeSegmentDataset(sc_segs_val_a, cfg)
val_b_ds = TrainAudioDataset(vb_df, SPECIES_TO_IDX, cfg, mode="val")
print(f"Datasets | train={len(train_ds)} val_a={len(val_a_ds)} val_b={len(val_b_ds)}")

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=(cfg.num_workers > 0))
val_a_loader = DataLoader(val_a_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=(cfg.num_workers > 0))
val_b_loader = DataLoader(val_b_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=(cfg.num_workers > 0))

# Model on single GPU (Phase 1 lesson)
model = SEDModel(cfg).to(DEVICE)
print(f"Model on single GPU")

mel_transform = MelTransform(cfg).to(DEVICE)
spec_aug = SpecAugmentations(cfg).to(DEVICE)
mixup = RawMixUp(prob=cfg.mixup_prob, alpha=cfg.mixup_alpha)
loss_fn = ClipFrameCELoss(cfg.clip_loss_weight, cfg.frame_loss_weight).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
scaler = torch.amp.GradScaler("cuda")

# Phase 4: Val-B が primary -> best.pth. Val-A は secondary -> best_val_a.pth
WEIGHT_PATH = str(WEIGHT_DIR / "best.pth")               # primary = best by Val-B
WEIGHT_PATH_VAL_A = str(WEIGHT_DIR / "best_val_a.pth")   # secondary = best by Val-A
best_a, best_b = 0.0, 0.0
best_a_epoch, best_b_epoch = -1, -1
log_rows = []

# DEBUG: smoke test before main loop
print(f"\n[DEBUG] dataset[0] load test...")
_t0 = time.time()
_s = train_ds[0]
print(f"[DEBUG] train_ds[0] OK in {time.time()-_t0:.2f}s, audio.shape={tuple(_s[0].shape)}, label.sum={float(_s[1].sum()):.1f}")

print(f"[DEBUG] First batch fetch test...")
_t0 = time.time()
_iter = iter(train_loader)
_b = next(_iter)
print(f"[DEBUG] First batch OK in {time.time()-_t0:.2f}s, audio.shape={tuple(_b[0].shape)}")

print(f"[DEBUG] Forward+backward test (with mixup + mel transform)...")
_t0 = time.time()
_audio = _b[0].to(DEVICE, non_blocking=True)
_lab = _b[1].to(DEVICE, non_blocking=True)
_audio, _lab = mixup(_audio, _lab)
_mel = mel_transform(_audio)
_mel = spec_aug(_mel)
with torch.amp.autocast("cuda"):
    _out = model(_mel)
    _loss = loss_fn(_out, _lab)
scaler.scale(_loss).backward()
optimizer.zero_grad()
print(f"[DEBUG] Forward+backward OK in {time.time()-_t0:.2f}s, mel.shape={tuple(_mel.shape)}, loss={_loss.item():.4f}")
del _iter, _b, _audio, _lab, _mel, _out, _loss
torch.cuda.empty_cache()

print(f'\n{"=" * 60}\nPHASE 4 | epochs={cfg.epochs} | chunk={cfg.chunk_duration}s | raw mixup\n{"=" * 60}')
for epoch in range(cfg.epochs):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                              mel_transform, spec_aug, mixup, loss_fn,
                              scaler, cfg, epoch)
    pa, ta = validate(model, val_a_loader, mel_transform)
    pb, tb = validate(model, val_b_loader, mel_transform)
    ma = compute_metrics(pa, ta)
    mb = compute_metrics(pb, tb)
    elapsed = time.time() - t0

    is_best_a = ma["macro_auc"] > best_a
    is_best_b = mb["macro_auc"] > best_b
    flag = ""
    if is_best_b: flag += "  <- best Val-B (primary)"
    if is_best_a: flag += "  <- best Val-A"
    print(f'  Ep {epoch+1:02d}/{cfg.epochs}'
          f' | Loss={tr_loss:.4f}'
          f' | A={ma["macro_auc"]:.4f} ({ma["num_classes_evaluated"]} cls)'
          f' | B={mb["macro_auc"]:.4f} ({mb["num_classes_evaluated"]} cls)'
          f' | {elapsed:.0f}s{flag}')

    log_rows.append(dict(epoch=epoch+1, tr_loss=tr_loss,
                         va_a_auc=ma["macro_auc"], va_a_cls=ma["num_classes_evaluated"],
                         va_b_auc=mb["macro_auc"], va_b_cls=mb["num_classes_evaluated"],
                         time=elapsed))

    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    if is_best_b:
        best_b = mb["macro_auc"]; best_b_epoch = epoch
        torch.save({"epoch": epoch, "model_state_dict": state,
                    "metrics_val_a": ma, "metrics_val_b": mb,
                    "cfg": cfg.__dict__, "species": SPECIES}, WEIGHT_PATH)
    if is_best_a:
        best_a = ma["macro_auc"]; best_a_epoch = epoch
        torch.save({"epoch": epoch, "model_state_dict": state,
                    "metrics_val_a": ma, "metrics_val_b": mb,
                    "cfg": cfg.__dict__, "species": SPECIES}, WEIGHT_PATH_VAL_A)

print(f'\n{"=" * 60}')
print(f'Best Val-B (primary): {best_b:.4f} @ Ep {best_b_epoch + 1}')
print(f'Best Val-A: {best_a:.4f} @ Ep {best_a_epoch + 1}')
print(f'{"=" * 60}')
pd.DataFrame(log_rows).to_csv(str(LOG_DIR / "train.csv"), index=False)"""))

# Summary
cells.append(code_cell("summary", r"""# ==============================================================
# SUMMARY
# ==============================================================
elapsed_total = time.time() - WALL_START
print(f"Wall time: {elapsed_total/60:.1f} min")
print(f"Best Val-B (primary): {best_b:.4f} @ Ep {best_b_epoch + 1}")
print(f"Best Val-A:           {best_a:.4f} @ Ep {best_a_epoch + 1}")

ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
print(f"Saved primary checkpoint: {WEIGHT_PATH}")
print(f"  epoch: {ckpt['epoch'] + 1}")
print(f"  metrics_val_a: {ckpt['metrics_val_a']}")
print(f"  metrics_val_b: {ckpt['metrics_val_b']}")
print(f"  cfg.chunk_duration: {ckpt['cfg']['chunk_duration']}")
print(f"  cfg.backbone: {ckpt['cfg']['backbone']}")
print(f"  output: /kaggle/working/weights/best.pth (Val-B primary)")
print(f"          /kaggle/working/weights/best_val_a.pth (Val-A secondary)")"""))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb1_train_p4.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
