"""Generate exp009 training notebook."""
import json

def code_cell(cell_id, source):
    lines = source.split("\n")
    # Each line gets \n except the last
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src
    }

def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": src
    }

cells = []

# ── Header ──
cells.append(md_cell("hdr", """# exp009: exp006 5-Fold + Mel Cache

exp006 (EfficientNet-B0 SED + Pseudo Label) をベースに:
- **Mel Cache** で学習を高速化 (28min/epoch → ~45s/epoch)
- **5-Fold CV** で汎化性能を向上
- Phase 1: 5-fold train_audio 学習
- Phase 2: 5-fold ensemble で疑似ラベル生成
- Phase 3: 疑似ラベルで 5-fold 再学習"""))

# ── Imports ──
cells.append(code_cell("imports", """!pip install -q timm torchaudio scikit-learn

import os, gc, ast, glob, time, random, warnings, json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision
import timm

from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPUs: {torch.cuda.device_count()}")"""))

# ── Config ──
cells.append(code_cell("config", r"""# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    sr: int = 32_000
    chunk_duration: float = 5.0
    target_size: tuple = (256, 256)

    # Mel (matches mel cache parameters)
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 16_000
    top_db: float = 80.0

    # Mel cache dequantization
    db_min: float = -80.0
    db_max: float = 20.0

    # Model
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    pretrained: bool = True
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0

    # Phase 1 Training
    epochs_p1: int = 15
    batch_size: int = 32
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    scheduler_T_0: int = 5
    grad_accum_steps: int = 1
    num_workers: int = 4

    # Phase 3 Training
    epochs_p3: int = 10
    lr_p3: float = 3e-4
    early_stopping_p3: int = 5
    samples_per_epoch_p3: int = 60000

    # MixUp (on mel spectrograms)
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.5

    # Loss
    clip_loss_weight: float = 0.5
    frame_loss_weight: float = 0.5

    # Augmentations
    freq_mask_param: int = 30
    time_mask_param: int = 30

    # Data
    seed: int = 42
    n_folds: int = 5
    use_secondary_labels: bool = True
    include_soundscape_labels: bool = True

    # Phase 2
    pseudo_batch_size: int = 128

    # Paths
    data_root: str = "/kaggle/input/competitions/birdclef-2026"
    mel_cache_train_audio: str = ""
    mel_cache_train_sc: str = ""
    output_dir: str = "/kaggle/working"

    @property
    def chunk_frames(self) -> int:
        return int(self.chunk_duration * self.sr / self.hop_length) + 1  # 313

    @property
    def db_range(self) -> float:
        return self.db_max - self.db_min

cfg = Config()
set_seed(cfg.seed)
print(f"Chunk: {cfg.chunk_duration}s = {cfg.chunk_frames} frames")
print(f"Dequantize: uint8 / 255 * {cfg.db_range} + {cfg.db_min}")"""))

# ── Paths & Species ──
cells.append(code_cell("paths", r"""# ══════════════════════════════════════════════════════════════
# PATHS & SPECIES
# ══════════════════════════════════════════════════════════════
DATA_ROOT = Path(cfg.data_root)
TRAIN_CSV = DATA_ROOT / "train.csv"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"
SC_LABELS_CSV = DATA_ROOT / "train_soundscapes_labels.csv"

WEIGHT_DIR = Path(cfg.output_dir) / "weights"
LOG_DIR = Path(cfg.output_dir) / "logs"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detect mel cache paths (2 separate notebooks)
def find_mel_dir(nb_slug, subdir):
    base = Path(f"/kaggle/input/{nb_slug}")
    candidates = [
        base / "mel_cache" / subdir,
        base / subdir,
        base / "mel_cache",
        base,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            # Check if this dir contains .npy files or subdirs with .npy
            if any(c.rglob("*.npy")):
                return c
    # Fallback: search
    for p in base.rglob("*.npy"):
        return p.parent
    return None

MEL_TRAIN_DIR = find_mel_dir("birdclef2026-mel-cache-train-audio-256", "train_audio")
MEL_SC_DIR = find_mel_dir("birdclef2026-mel-cache-train-sc-256", "train_soundscapes")

assert MEL_TRAIN_DIR is not None, "Could not find train_audio mel cache"
assert MEL_SC_DIR is not None, "Could not find train_soundscapes mel cache"

cfg.mel_cache_train_audio = str(MEL_TRAIN_DIR)
cfg.mel_cache_train_sc = str(MEL_SC_DIR)

print(f"Mel train_audio dir: {MEL_TRAIN_DIR}")
print(f"Mel soundscapes dir: {MEL_SC_DIR}")

# Count files
n_train_mels = len(list(MEL_TRAIN_DIR.rglob("*.npy")))
n_sc_mels = len(list(MEL_SC_DIR.rglob("*.npy")))
print(f"  train_audio mels: {n_train_mels:,}")
print(f"  soundscape mels:  {n_sc_mels:,}")

# Verify config
for nb_slug in ["birdclef2026-mel-cache-train-audio-256", "birdclef2026-mel-cache-train-sc-256"]:
    base = Path(f"/kaggle/input/{nb_slug}")
    for cfg_path in [base / "mel_cache" / "config.json", base / "config.json"]:
        if cfg_path.exists():
            with open(cfg_path) as f:
                mel_cfg = json.load(f)
            print(f"Config ({nb_slug}): n_mels={mel_cfg.get('n_mels')}")
            assert mel_cfg["n_mels"] == cfg.n_mels, f"n_mels mismatch"
            break

sub_df = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
SPECIES = list(sub_df.columns[1:])
SPECIES_TO_IDX = {sp: i for i, sp in enumerate(SPECIES)}
print(f"Species: {len(SPECIES)}")"""))

# ── Mel Cache Utilities ──
cells.append(code_cell("mel-utils", r"""# ══════════════════════════════════════════════════════════════
# MEL CACHE UTILITIES
# ══════════════════════════════════════════════════════════════
def crop_mel(mel, target_frames, mode="train"):
    # Crop or pad mel spectrogram to target_frames.
    T = mel.shape[1]
    if T >= target_frames:
        if mode == "train":
            start = np.random.randint(0, T - target_frames + 1)
        else:
            start = 0
        return mel[:, start:start + target_frames]
    else:
        padded = np.zeros((mel.shape[0], target_frames), dtype=mel.dtype)
        if mode == "train":
            start = np.random.randint(0, target_frames - T + 1)
        else:
            start = 0
        padded[:, start:start + T] = mel
        return padded

def dequantize_mel(mel_uint8, db_min=-80.0, db_range=100.0):
    # uint8 -> float32 dB: mel_db = uint8 / 255 * db_range + db_min
    return mel_uint8.astype(np.float32) / 255.0 * db_range + db_min"""))

# ── Transform & Augmentations ──
cells.append(code_cell("transform", r"""# ══════════════════════════════════════════════════════════════
# TRANSFORM & AUGMENTATIONS (on mel spectrograms)
# ══════════════════════════════════════════════════════════════
class MelCacheTransform(nn.Module):
    # Resize + per-sample normalize + 3ch repeat.
    def __init__(self, cfg):
        super().__init__()
        self.resize = torchvision.transforms.Resize(cfg.target_size, antialias=True)

    @torch.no_grad()
    def forward(self, mel_db):
        # (B, n_mels, T) -> (B, 3, H, W)
        with torch.amp.autocast("cuda", enabled=False):
            mel_db = mel_db.float()
            mel = self.resize(mel_db.unsqueeze(1))  # (B, 1, H, W)
            mel = mel.squeeze(1)                     # (B, H, W)
            B = mel.shape[0]
            mel_flat = mel.reshape(B, -1)
            mel_min = mel_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel_max = mel_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
            mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return mel

class SpecAugmentations(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)

    def forward(self, mel):
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        return mel

class MelMixUp:
    # MixUp on mel spectrograms.
    def __init__(self, prob=0.5, alpha=0.5):
        self.prob = prob
        self.alpha = alpha

    def __call__(self, mels, labels):
        if torch.rand(1).item() > self.prob:
            return mels, labels
        indices = torch.randperm(mels.size(0), device=mels.device)
        mixed = self.alpha * mels + (1.0 - self.alpha) * mels[indices]
        mixed_labels = torch.max(labels, labels[indices])
        return mixed, mixed_labels"""))

# ── Model ──
cells.append(code_cell("model", r"""# ══════════════════════════════════════════════════════════════
# MODEL (same as exp006)
# ══════════════════════════════════════════════════════════════
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=self.eps).pow(p)
            x = x.mean(dim=2)
            x = x.pow(1.0 / p)
        return x

class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        clipwise_logit = (att * cls).sum(dim=-1)
        clipwise_prob = torch.sigmoid(clipwise_logit)
        segmentwise_logit = cls.permute(0, 2, 1)
        return {
            "clipwise_logit": clipwise_logit,
            "clipwise_prob": clipwise_prob,
            "segmentwise_logit": segmentwise_logit,
        }

class SEDModel(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        use_pretrained = pretrained if pretrained is not None else cfg.pretrained
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=use_pretrained,
            in_chans=cfg.in_channels, features_only=False,
            global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(feat_dim, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.gem_pool(features)
        return self.head(pooled)

_m = SEDModel(cfg, pretrained=False)
print(f"Backbone features: {_m.backbone.num_features}")
print(f"Total params: {sum(p.numel() for p in _m.parameters())/1e6:.1f}M")
del _m"""))

# ── Loss ──
cells.append(code_cell("loss", r"""# ══════════════════════════════════════════════════════════════
# LOSS (same as exp006)
# ══════════════════════════════════════════════════════════════
class ClipFrameCELoss(nn.Module):
    def __init__(self, clip_weight=0.5, frame_weight=0.5):
        super().__init__()
        self.clip_weight = clip_weight
        self.frame_weight = frame_weight

    def forward(self, outputs, targets):
        clip_logit = outputs["clipwise_logit"]
        loss_clip = F.binary_cross_entropy_with_logits(clip_logit, targets)
        seg_logit = outputs["segmentwise_logit"]
        frame_max_logit = seg_logit.max(dim=1)[0]
        loss_frame = F.binary_cross_entropy_with_logits(frame_max_logit, targets)
        return self.clip_weight * loss_clip + self.frame_weight * loss_frame"""))

# ── Dataset ──
cells.append(code_cell("dataset", r"""# ══════════════════════════════════════════════════════════════
# DATASET (mel cache version)
# ══════════════════════════════════════════════════════════════
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
    t = str(t)
    if ":" in t:
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(t)

def prepare_soundscape_segments(sc_labels_df, species_to_idx, frame_rate):
    segments = []
    num_classes = len(species_to_idx)
    for _, row in sc_labels_df.iterrows():
        label = np.zeros(num_classes, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                label[species_to_idx[sp]] = 1.0
        start_sec = _parse_time_to_seconds(row["start"])
        segments.append({
            "filename": row["filename"],
            "stem": Path(row["filename"]).stem,
            "start_frame": int(start_sec * frame_rate),
            "label": label,
        })
    return segments

class MelCacheDataset(Dataset):
    # Load pre-computed mel spectrograms from cache.

    def __init__(self, train_df, species_to_idx, cfg,
                 soundscape_segments=None, mode="train"):
        self.cfg = cfg
        self.species_to_idx = species_to_idx
        self.num_classes = len(species_to_idx)
        self.mode = mode
        self.mel_train_dir = Path(cfg.mel_cache_train_audio)
        self.mel_sc_dir = Path(cfg.mel_cache_train_sc)
        self.train_items = train_df.reset_index(drop=True)
        self.n_train = len(self.train_items)
        self.sc_segments = soundscape_segments or []
        self.n_sc = len(self.sc_segments)
        self.target_frames = cfg.chunk_frames

    def __len__(self):
        return self.n_train + self.n_sc

    def __getitem__(self, idx):
        if idx < self.n_train:
            mel, label = self._load_train_mel(idx)
        else:
            mel, label = self._load_sc_mel(idx - self.n_train)
        return torch.from_numpy(mel).float(), torch.from_numpy(label).float()

    def _load_train_mel(self, idx):
        row = self.train_items.iloc[idx]
        filename = row["filename"]
        mel_path = self.mel_train_dir / filename.replace(".ogg", ".npy")
        try:
            mel_uint8 = np.load(str(mel_path))
        except Exception:
            mel_uint8 = np.zeros((self.cfg.n_mels, self.target_frames), dtype=np.uint8)
        mel_uint8 = crop_mel(mel_uint8, self.target_frames, self.mode)
        mel_db = dequantize_mel(mel_uint8, self.cfg.db_min, self.cfg.db_range)
        label = np.zeros(self.num_classes, dtype=np.float32)
        sp = str(row["primary_label"])
        if sp in self.species_to_idx:
            label[self.species_to_idx[sp]] = 1.0
        if self.cfg.use_secondary_labels:
            for sec_sp in _parse_secondary_labels(row.get("secondary_labels", "[]")):
                if sec_sp in self.species_to_idx:
                    label[self.species_to_idx[sec_sp]] = 1.0
        return mel_db, label

    def _load_sc_mel(self, seg_idx):
        seg = self.sc_segments[seg_idx]
        mel_path = self.mel_sc_dir / (seg["stem"] + ".npy")
        try:
            mel_uint8 = np.load(str(mel_path))
        except Exception:
            mel_uint8 = np.zeros((self.cfg.n_mels, self.target_frames), dtype=np.uint8)
        start = seg["start_frame"]
        end = start + self.target_frames
        if end <= mel_uint8.shape[1]:
            mel_uint8 = mel_uint8[:, start:end]
        else:
            chunk = mel_uint8[:, start:]
            padded = np.zeros((self.cfg.n_mels, self.target_frames), dtype=np.uint8)
            padded[:, :min(chunk.shape[1], self.target_frames)] = chunk[:, :self.target_frames]
            mel_uint8 = padded
        mel_db = dequantize_mel(mel_uint8, self.cfg.db_min, self.cfg.db_range)
        return mel_db, seg["label"]

class SoundscapePseudoMelDataset(Dataset):
    # Soundscape segments with soft pseudo labels from mel cache.

    def __init__(self, pseudo_df, species_list, cfg):
        self.pseudo_df = pseudo_df.reset_index(drop=True)
        self.label_values = pseudo_df[species_list].values.astype(np.float32)
        self.cfg = cfg
        self.mel_sc_dir = Path(cfg.mel_cache_train_sc)
        self.target_frames = cfg.chunk_frames
        self.frame_rate = cfg.sr / cfg.hop_length

    def __len__(self):
        return len(self.pseudo_df)

    def __getitem__(self, idx):
        row = self.pseudo_df.iloc[idx]
        stem = Path(row["filename"]).stem
        mel_path = self.mel_sc_dir / (stem + ".npy")
        try:
            mel_uint8 = np.load(str(mel_path))
        except Exception:
            mel_uint8 = np.zeros((self.cfg.n_mels, self.target_frames), dtype=np.uint8)
        start_frame = int(float(row["start_sec"]) * self.frame_rate)
        end_frame = start_frame + self.target_frames
        if end_frame <= mel_uint8.shape[1]:
            mel_uint8 = mel_uint8[:, start_frame:end_frame]
        else:
            chunk = mel_uint8[:, start_frame:]
            padded = np.zeros((self.cfg.n_mels, self.target_frames), dtype=np.uint8)
            n = min(chunk.shape[1], self.target_frames)
            padded[:, :n] = chunk[:, :n]
            mel_uint8 = padded
        mel_db = dequantize_mel(mel_uint8, self.cfg.db_min, self.cfg.db_range)
        label = self.label_values[idx]
        return torch.from_numpy(mel_db).float(), torch.from_numpy(label).float()

print("Dataset classes defined")"""))

# ── Training Utilities ──
cells.append(code_cell("train-utils", r"""# ══════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, mel_transform,
                    spec_aug, mixup, loss_fn, device, cfg, epoch, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (mel_db, labels) in enumerate(tqdm(loader, desc=f"  train", leave=False)):
        mel_db = mel_db.to(device)
        labels = labels.to(device)

        # MixUp on mel (before resize/normalize)
        mel_db, labels = mixup(mel_db, labels)

        # Transform: resize + normalize + 3ch
        mel = mel_transform(mel_db)

        # SpecAugment
        if spec_aug.training:
            mel = spec_aug(mel)

        with torch.amp.autocast("cuda"):
            outputs = model(mel)
            loss = loss_fn(outputs, labels)

        loss = loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / len(loader))

        total_loss += loss.item() * cfg.grad_accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate(model, loader, mel_transform, device):
    model.eval()
    all_preds, all_targets = [], []
    for mel_db, labels in tqdm(loader, desc="  valid", leave=False):
        mel_db = mel_db.to(device)
        mel = mel_transform(mel_db)
        with torch.amp.autocast("cuda"):
            outputs = model(mel)
        preds = outputs["clipwise_prob"].float().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

def compute_metrics(preds, targets):
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except ValueError:
                pass
    return {
        "macro_auc": np.mean(aucs) if aucs else 0.0,
        "num_classes_evaluated": len(aucs),
    }

def create_folds(train_df, cfg):
    df = train_df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(df, df["primary_label"], groups=df["author"])
    ):
        df.loc[df.index[fold_idx], "fold"] = fold_idx
    # Fix: assign all val indices
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(df, df["primary_label"], groups=df["author"])
    ):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx
    return df"""))

# Wait, the create_folds function has a bug - it iterates twice. Let me fix it.
# Actually let me rewrite cell "train-utils" with the correct create_folds.

cells[-1] = code_cell("train-utils", r"""# ══════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, mel_transform,
                    spec_aug, mixup, loss_fn, device, cfg, epoch, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (mel_db, labels) in enumerate(tqdm(loader, desc=f"  train", leave=False)):
        mel_db = mel_db.to(device)
        labels = labels.to(device)

        # MixUp on mel (before resize/normalize)
        mel_db, labels = mixup(mel_db, labels)

        # Transform: resize + normalize + 3ch
        mel = mel_transform(mel_db)

        # SpecAugment
        if spec_aug.training:
            mel = spec_aug(mel)

        with torch.amp.autocast("cuda"):
            outputs = model(mel)
            loss = loss_fn(outputs, labels)

        loss = loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / len(loader))

        total_loss += loss.item() * cfg.grad_accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate(model, loader, mel_transform, device):
    model.eval()
    all_preds, all_targets = [], []
    for mel_db, labels in tqdm(loader, desc="  valid", leave=False):
        mel_db = mel_db.to(device)
        mel = mel_transform(mel_db)
        with torch.amp.autocast("cuda"):
            outputs = model(mel)
        preds = outputs["clipwise_prob"].float().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

def compute_metrics(preds, targets):
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except ValueError:
                pass
    return {
        "macro_auc": np.mean(aucs) if aucs else 0.0,
        "num_classes_evaluated": len(aucs),
    }

def create_folds(train_df, cfg):
    df = train_df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(df, df["primary_label"], groups=df["author"])
    ):
        df.loc[df.index[val_idx], "fold"] = fold_idx
    return df""")

# ── Phase 1 header ──
cells.append(md_cell("phase1-hdr", "## Phase 1: 5-Fold Training on train_audio"))

# ── Phase 1: 5-fold training ──
cells.append(code_cell("phase1", r"""# ══════════════════════════════════════════════════════════════
# PHASE 1: 5-FOLD TRAINING
# ══════════════════════════════════════════════════════════════
train_df = pd.read_csv(TRAIN_CSV)
print(f"Train recordings: {len(train_df)}")

# Soundscape GT labels
sc_segments = None
if cfg.include_soundscape_labels:
    sc_labels = pd.read_csv(SC_LABELS_CSV)
    frame_rate = cfg.sr / cfg.hop_length
    sc_segments = prepare_soundscape_segments(sc_labels, SPECIES_TO_IDX, frame_rate)
    print(f"Soundscape GT segments: {len(sc_segments)}")

# Create folds
train_df = create_folds(train_df, cfg)
print(f"Fold distribution:\n{train_df['fold'].value_counts().sort_index()}")

# 5-fold training
all_fold_aucs = []
total_t0 = time.time()

for fold in range(cfg.n_folds):
    print(f'\n{"=" * 60}')
    print(f"PHASE 1 | Fold {fold}/{cfg.n_folds - 1} | {cfg.epochs_p1} epochs")
    print(f'{"=" * 60}')

    fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
    print(f"  train={len(fold_train_df)}, val={len(fold_val_df)}")

    # Datasets
    train_ds = MelCacheDataset(fold_train_df, SPECIES_TO_IDX, cfg,
                               soundscape_segments=sc_segments, mode="train")
    val_ds = MelCacheDataset(fold_val_df, SPECIES_TO_IDX, cfg,
                             soundscape_segments=None, mode="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=True)

    # Model & training setup
    model = SEDModel(cfg).to(DEVICE)
    mel_transform = MelCacheTransform(cfg).to(DEVICE)
    spec_aug = SpecAugmentations(cfg).to(DEVICE)
    mixup = MelMixUp(prob=cfg.mixup_prob, alpha=cfg.mixup_alpha)
    loss_fn = ClipFrameCELoss(cfg.clip_loss_weight, cfg.frame_loss_weight).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.scheduler_T_0, eta_min=cfg.lr_min)
    scaler = torch.amp.GradScaler("cuda")

    weight_path = str(WEIGHT_DIR / f"phase1_best_fold{fold}.pth")
    best_auc, best_epoch = 0.0, -1
    log_rows = []

    for epoch in range(cfg.epochs_p1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                  mel_transform, spec_aug, mixup, loss_fn,
                                  DEVICE, cfg, epoch, scaler)
        preds, targets = validate(model, val_loader, mel_transform, DEVICE)
        metrics = compute_metrics(preds, targets)
        elapsed = time.time() - t0

        is_best = metrics["macro_auc"] > best_auc
        print(f'  Ep {epoch+1:02d}/{cfg.epochs_p1}'
              f' | Loss={tr_loss:.4f}'
              f' | AUC={metrics["macro_auc"]:.4f} ({metrics["num_classes_evaluated"]} cls)'
              f' | {elapsed:.0f}s{"  <- best" if is_best else ""}')

        log_rows.append(dict(fold=fold, epoch=epoch+1, tr_loss=tr_loss,
                             va_auc=metrics["macro_auc"], time=elapsed))

        if is_best:
            best_auc = metrics["macro_auc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "fold": fold,
                "model_state_dict": model.state_dict(),
                "metrics": metrics, "cfg": cfg.__dict__,
            }, weight_path)

    all_fold_aucs.append(best_auc)
    print(f"  Fold {fold} Best AUC: {best_auc:.4f} @ Epoch {best_epoch + 1}")

    pd.DataFrame(log_rows).to_csv(str(LOG_DIR / f"phase1_fold{fold}.csv"), index=False)

    del model, optimizer, scheduler, scaler, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

total_elapsed = time.time() - total_t0
print(f'\n{"=" * 60}')
print(f"Phase 1 Complete | {total_elapsed/60:.1f} min")
print(f"Mean AUC: {np.mean(all_fold_aucs):.4f}")
for f, auc in enumerate(all_fold_aucs):
    print(f"  Fold {f}: {auc:.4f}")
print(f'{"=" * 60}')"""))

# ── Phase 2 header ──
cells.append(md_cell("phase2-hdr", "## Phase 2: Pseudo Label Generation (5-fold ensemble)"))

# ── Phase 2 ──
cells.append(code_cell("phase2", r"""# ══════════════════════════════════════════════════════════════
# PHASE 2: PSEUDO LABEL GENERATION
# ══════════════════════════════════════════════════════════════
soundscape_mel_dir = Path(cfg.mel_cache_train_sc)
soundscape_mels = sorted(soundscape_mel_dir.glob("*.npy"))
print(f"Soundscape mel files: {len(soundscape_mels)}")

mel_transform = MelCacheTransform(cfg).to(DEVICE)
CHUNK_FRAMES = cfg.chunk_frames

all_pseudo_rows = []
t0 = time.time()

for fold in range(cfg.n_folds):
    weight_path = str(WEIGHT_DIR / f"phase1_best_fold{fold}.pth")
    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model = SEDModel(cfg, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Fold {fold}: loaded (AUC={ckpt['metrics']['macro_auc']:.4f})")

    fold_rows = []
    for mel_file in tqdm(soundscape_mels, desc=f"  Fold {fold} pseudo", leave=False):
        fname = mel_file.stem
        mel_uint8 = np.load(str(mel_file))
        T_full = mel_uint8.shape[1]
        n_chunks = max(1, T_full // CHUNK_FRAMES)

        chunks = []
        for i in range(n_chunks):
            start = i * CHUNK_FRAMES
            end = start + CHUNK_FRAMES
            if end <= T_full:
                chunk = mel_uint8[:, start:end]
            else:
                chunk = np.zeros((cfg.n_mels, CHUNK_FRAMES), dtype=np.uint8)
                chunk[:, :T_full - start] = mel_uint8[:, start:]
            chunks.append(dequantize_mel(chunk, cfg.db_min, cfg.db_range))

        chunks_tensor = torch.from_numpy(np.stack(chunks)).float().to(DEVICE)

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(chunks_tensor), cfg.pseudo_batch_size):
                batch = chunks_tensor[i:i + cfg.pseudo_batch_size]
                mel = mel_transform(batch)
                with torch.amp.autocast("cuda"):
                    outputs = model(mel)
                all_probs.append(outputs["clipwise_prob"].float().cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        for i in range(n_chunks):
            end_sec = (i + 1) * int(cfg.chunk_duration)
            row = {
                "filename": fname + ".ogg",
                "start_sec": i * int(cfg.chunk_duration),
                "end_sec": end_sec,
                "fold": fold,
            }
            for si, sp in enumerate(SPECIES):
                row[sp] = float(probs[i, si])
            fold_rows.append(row)

    all_pseudo_rows.extend(fold_rows)
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Average predictions across folds
pseudo_all_df = pd.DataFrame(all_pseudo_rows)
group_cols = ["filename", "start_sec", "end_sec"]
pseudo_df = pseudo_all_df.groupby(group_cols)[SPECIES].mean().reset_index()

elapsed = time.time() - t0
print(f"\nPhase 2 done: {len(pseudo_df)} segments in {elapsed:.0f}s")

# Stats
probs_matrix = pseudo_df[SPECIES].values
print(f"Mean: {probs_matrix.mean():.6f}, Max: {probs_matrix.max():.4f}")
for th in [0.5, 0.3, 0.1]:
    n_pos = (probs_matrix > th).sum()
    print(f"  > {th}: {n_pos:,} ({n_pos / probs_matrix.size * 100:.2f}%)")"""))

# ── Phase 3 header ──
cells.append(md_cell("phase3-hdr", "## Phase 3: 5-Fold Retrain with Pseudo Labels"))

# ── Phase 3 ──
cells.append(code_cell("phase3", r"""# ══════════════════════════════════════════════════════════════
# PHASE 3: 5-FOLD RETRAIN WITH PSEUDO LABELS
# ══════════════════════════════════════════════════════════════
all_fold_aucs_p3 = []
total_t0 = time.time()

for fold in range(cfg.n_folds):
    print(f'\n{"=" * 60}')
    print(f"PHASE 3 | Fold {fold}/{cfg.n_folds - 1} | {cfg.epochs_p3} epochs")
    print(f'{"=" * 60}')

    fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Load Phase 1 best weights
    p1_path = str(WEIGHT_DIR / f"phase1_best_fold{fold}.pth")
    model = SEDModel(cfg, pretrained=False).to(DEVICE)
    ckpt = torch.load(p1_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Phase 1 loaded: AUC={ckpt['metrics']['macro_auc']:.4f}")

    mel_transform = MelCacheTransform(cfg).to(DEVICE)
    spec_aug = SpecAugmentations(cfg).to(DEVICE)
    mixup = MelMixUp(prob=cfg.mixup_prob, alpha=cfg.mixup_alpha)
    loss_fn = ClipFrameCELoss(cfg.clip_loss_weight, cfg.frame_loss_weight).to(DEVICE)

    # Datasets: train_audio + pseudo-labeled soundscapes
    audio_ds = MelCacheDataset(fold_train_df, SPECIES_TO_IDX, cfg,
                               soundscape_segments=None, mode="train")
    sc_pseudo_ds = SoundscapePseudoMelDataset(pseudo_df, SPECIES, cfg)
    combined_ds = ConcatDataset([audio_ds, sc_pseudo_ds])

    # Balanced sampling
    n_audio = len(audio_ds)
    n_sc = len(sc_pseudo_ds)
    w_audio = 1.0 / n_audio
    w_sc = 1.0 / n_sc
    sample_weights = [w_audio] * n_audio + [w_sc] * n_sc
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=cfg.samples_per_epoch_p3,
                                    replacement=True)

    train_loader = DataLoader(combined_ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_ds = MelCacheDataset(fold_val_df, SPECIES_TO_IDX, cfg,
                             soundscape_segments=None, mode="val")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=True)

    print(f"  Train: {n_audio} audio + {n_sc} pseudo-SC = {len(combined_ds)}")
    print(f"  Sampled/epoch: {cfg.samples_per_epoch_p3}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_p3, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.epochs_p3, eta_min=cfg.lr_min)
    scaler = torch.amp.GradScaler("cuda")

    p3_path = str(WEIGHT_DIR / f"best_fold{fold}.pth")
    best_auc, best_epoch = 0.0, -1
    log_rows = []

    for epoch in range(cfg.epochs_p3):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                  mel_transform, spec_aug, mixup, loss_fn,
                                  DEVICE, cfg, epoch, scaler)
        preds, targets = validate(model, val_loader, mel_transform, DEVICE)
        metrics = compute_metrics(preds, targets)
        elapsed = time.time() - t0

        is_best = metrics["macro_auc"] > best_auc
        print(f'  Ep {epoch+1:02d}/{cfg.epochs_p3}'
              f' | Loss={tr_loss:.4f}'
              f' | AUC={metrics["macro_auc"]:.4f}'
              f' | {elapsed:.0f}s{"  <- best" if is_best else ""}')

        log_rows.append(dict(fold=fold, epoch=epoch+1, tr_loss=tr_loss,
                             va_auc=metrics["macro_auc"], time=elapsed))

        if is_best:
            best_auc = metrics["macro_auc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "fold": fold,
                "model_state_dict": model.state_dict(),
                "metrics": metrics, "cfg": cfg.__dict__,
                "species": SPECIES,
            }, p3_path)

        if epoch - best_epoch >= cfg.early_stopping_p3:
            print(f"  Early stopping: no improvement for {cfg.early_stopping_p3} epochs")
            break

    all_fold_aucs_p3.append(best_auc)
    print(f"  Fold {fold} Best AUC: {best_auc:.4f} @ Epoch {best_epoch + 1}")

    pd.DataFrame(log_rows).to_csv(str(LOG_DIR / f"phase3_fold{fold}.csv"), index=False)

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

total_elapsed = time.time() - total_t0
print(f'\n{"=" * 60}')
print(f"Phase 3 Complete | {total_elapsed/60:.1f} min")
print(f"Mean AUC: {np.mean(all_fold_aucs_p3):.4f}")
for f, auc in enumerate(all_fold_aucs_p3):
    print(f"  Fold {f}: {auc:.4f}")
print(f"Phase 1 Mean AUC: {np.mean(all_fold_aucs):.4f} (comparison)")
print(f'{"=" * 60}')"""))

# ── Summary ──
cells.append(code_cell("summary", r"""# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("Saved weights:")
for p in sorted(WEIGHT_DIR.glob("*.pth")):
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    auc = ckpt.get("metrics", {}).get("macro_auc", 0)
    fold = ckpt.get("fold", "?")
    print(f"  {p.name}: fold={fold}, AUC={auc:.4f}")

print(f"\nPhase 1 Mean AUC: {np.mean(all_fold_aucs):.4f}")
print(f"Phase 3 Mean AUC: {np.mean(all_fold_aucs_p3):.4f}")
print(f"\nOutput: {cfg.output_dir}/weights/")
print("Phase 3 weights (best_fold*.pth) ready for submission notebook.")"""))

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = "C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp009/notebook/train.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
