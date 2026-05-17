"""Generate exp011 Phase 1 training notebook (Boredom-style minimum SED).

Phase 1 scope:
- Single fold (90/10 random split, filename-group safe)
- BC2026 train_audio (mel cache) + labeled_soundscapes (66 files)
- Backbone: tf_efficientnetv2_b0 (Salman 0.937 confirmed)
- SED head with framewise attention (Salman: framewise > LSE)
- ClipFrameCELoss (BCE on clipwise + framewise max)
- 10 sec chunks, 20 epochs, AdamW + CosineAnnealing
- spec mixup α=0.5 + SpecAugment (freq=30, time=40)
- Target LB 0.92-0.93

GPU: T4x2. Output: /kaggle/working/best.pth
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

# ── Header ──
cells.append(md_cell("hdr", """# exp011 Phase 1: Boredom-style SED minimum

timm + SED non-Perch routes をまず動かす最小構成。

- **Backbone**: `tf_efficientnetv2_b0` (Salman Expert 単一 LB 0.937 確定実績、Fused-MBConv で学習安定)
- **Head**: GeMFreqPool + AttentionSEDHead (framewise max, Salman 推奨)
- **Loss**: ClipFrameCELoss (clipwise BCE + framewise max BCE)
- **Input**: 10 秒 (Salman/hengck23 long input rationale)
- **Augmentation**: spec mixup α=0.5, SpecAug freq=30/time=40
- **Single fold** (90/10 random split, filename group safe)
- **Epochs 20**, AdamW lr=5e-4, CosineAnnealing
- **GPU**: T4x2

Goal: LB **0.92-0.93** baseline。Phase 2 で過去年 pretrain、Phase 3 で iter pseudo を順次積む。
"""))

# ── Imports ──
cells.append(code_cell("imports", r"""!pip install -q timm torchaudio scikit-learn

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

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPUs: {torch.cuda.device_count()}")
WALL_START = time.time()"""))

# ── Config ──
cells.append(code_cell("config", r"""# ══════════════════════════════════════════════════════════════
# CONFIG (Phase 1)
# ══════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # Audio / Mel (matches mel_cache_256 spec)
    sr: int = 32_000
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 16_000
    top_db: float = 80.0
    db_min: float = -80.0
    db_max: float = 20.0

    # Chunk: 10 sec window
    chunk_duration: float = 10.0
    target_size: tuple = (256, 256)  # resize after crop

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
    batch_size: int = 32
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    num_workers: int = 4

    # Augmentation
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.5
    freq_mask_param: int = 30
    time_mask_param: int = 40

    # Loss weights
    clip_loss_weight: float = 0.5
    frame_loss_weight: float = 0.5

    # Data
    seed: int = 42
    val_ratio: float = 0.10
    use_secondary_labels: bool = True
    include_soundscape_labels: bool = True

    # Paths
    data_root: str = "/kaggle/input/competitions/birdclef-2026"
    mel_cache_train_audio: str = ""
    mel_cache_train_sc: str = ""
    output_dir: str = "/kaggle/working"

    @property
    def chunk_frames(self) -> int:
        return int(self.chunk_duration * self.sr / self.hop_length) + 1  # 626 for 10s

    @property
    def db_range(self) -> float:
        return self.db_max - self.db_min


cfg = Config()
set_seed(cfg.seed)
print(f"Chunk: {cfg.chunk_duration}s = {cfg.chunk_frames} frames")
print(f"Backbone: {cfg.backbone}")
print(f"Epochs: {cfg.epochs}, Batch: {cfg.batch_size}, LR: {cfg.lr}")"""))

# ── Paths & Species ──
cells.append(code_cell("paths", r"""# ══════════════════════════════════════════════════════════════
# PATHS & SPECIES (autodetect Kaggle mount points)
# ══════════════════════════════════════════════════════════════
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

WEIGHT_DIR = Path(cfg.output_dir) / "weights"
LOG_DIR = Path(cfg.output_dir) / "logs"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def find_mel_dir(slug):
    # Mount candidates (kernel_sources は /kaggle/input/notebooks/{user}/{slug}/)
    bases = [Path(f"/kaggle/input/notebooks/maekeso/{slug}"),  # kernel_sources の正規パス
             Path(f"/kaggle/input/datasets/maekeso/{slug}"),   # dataset_sources
             Path(f"/kaggle/input/{slug}")]                     # 単純 mount fallback
    for base in bases:
        if not base.exists():
            continue
        # walk to find dir with .npy files
        for p in [base / "mel_cache" / "train_audio", base / "mel_cache" / "train_soundscapes",
                  base / "mel_cache", base]:
            if p.exists() and any(p.rglob("*.npy")):
                return p
    return None


MEL_TRAIN_DIR = find_mel_dir("birdclef2026-mel-cache-train-audio-256")
MEL_SC_DIR = find_mel_dir("birdclef2026-mel-cache-train-sc-256")
assert MEL_TRAIN_DIR is not None, "train_audio mel cache not found"
assert MEL_SC_DIR is not None, "train_soundscapes mel cache not found"
cfg.mel_cache_train_audio = str(MEL_TRAIN_DIR)
cfg.mel_cache_train_sc = str(MEL_SC_DIR)
print(f"MEL train_audio: {MEL_TRAIN_DIR} (n={sum(1 for _ in MEL_TRAIN_DIR.rglob('*.npy'))})")
print(f"MEL soundscape: {MEL_SC_DIR} (n={sum(1 for _ in MEL_SC_DIR.rglob('*.npy'))})")

sub_df = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
SPECIES = list(sub_df.columns[1:])
SPECIES_TO_IDX = {sp: i for i, sp in enumerate(SPECIES)}
print(f"Species: {len(SPECIES)}")"""))

# ── Mel utilities ──
cells.append(code_cell("mel-utils", r"""# ══════════════════════════════════════════════════════════════
# MEL CACHE UTILITIES
# ══════════════════════════════════════════════════════════════
def crop_mel(mel, target_frames, mode="train"):
    Tf = mel.shape[1]
    if Tf >= target_frames:
        start = np.random.randint(0, Tf - target_frames + 1) if mode == "train" else 0
        return mel[:, start:start + target_frames]
    padded = np.zeros((mel.shape[0], target_frames), dtype=mel.dtype)
    start = np.random.randint(0, target_frames - Tf + 1) if mode == "train" else 0
    padded[:, start:start + Tf] = mel
    return padded


def dequantize_mel(mel_uint8, db_min=-80.0, db_range=100.0):
    return mel_uint8.astype(np.float32) / 255.0 * db_range + db_min"""))

# ── Transforms / Augment ──
cells.append(code_cell("transform", r"""# ══════════════════════════════════════════════════════════════
# TRANSFORM + AUGMENT (on-GPU)
# ══════════════════════════════════════════════════════════════
class MelCacheTransform(nn.Module):
    # Resize + per-sample min-max normalize + 3ch repeat.
    def __init__(self, cfg):
        super().__init__()
        self.resize = torchvision.transforms.Resize(cfg.target_size, antialias=True)

    @torch.no_grad()
    def forward(self, mel_db):
        # mel_db: (B, n_mels, T) float
        with torch.amp.autocast("cuda", enabled=False):
            x = mel_db.float()
            x = self.resize(x.unsqueeze(1)).squeeze(1)  # (B, H, W)
            B = x.shape[0]
            flat = x.reshape(B, -1)
            mn = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            mx = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
            x = (x - mn) / (mx - mn + 1e-7)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return x


class SpecAugmentations(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)

    def forward(self, mel):
        return self.time_mask(self.freq_mask(mel))


class MelMixUp:
    # Spec-domain MixUp on mel_db (before transform/normalize).
    def __init__(self, prob=0.5, alpha=0.5):
        self.prob = prob; self.alpha = alpha

    def __call__(self, mels, labels):
        if torch.rand(1).item() > self.prob:
            return mels, labels
        idx = torch.randperm(mels.size(0), device=mels.device)
        mixed = self.alpha * mels + (1.0 - self.alpha) * mels[idx]
        mixed_labels = torch.max(labels, labels[idx])
        return mixed, mixed_labels"""))

# ── Model ──
cells.append(code_cell("model", r"""# ══════════════════════════════════════════════════════════════
# MODEL: timm v2B0 + GeMFreqPool + AttentionSEDHead
# ══════════════════════════════════════════════════════════════
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
        # x: (B, feat, T)
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
        feat = self.backbone(x)             # (B, C, H, W)
        pooled = self.gem_pool(feat)        # (B, C, W) — frequency-pooled
        return self.head(pooled)


_m = SEDModel(cfg, pretrained=False)
print(f"Backbone features: {_m.backbone.num_features}")
print(f"Total params: {sum(p.numel() for p in _m.parameters())/1e6:.2f}M")
del _m"""))

# ── Loss ──
cells.append(code_cell("loss", r"""# ══════════════════════════════════════════════════════════════
# LOSS: ClipFrameCELoss (BCE on clipwise + framewise max)
# ══════════════════════════════════════════════════════════════
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

# ── Dataset ──
cells.append(code_cell("dataset", r"""# ══════════════════════════════════════════════════════════════
# DATASET (mel cache reader)
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
    s = str(t)
    if ":" in s:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(s)


def prepare_soundscape_segments(sc_labels_df, species_to_idx, frame_rate):
    out = []
    nC = len(species_to_idx)
    for _, row in sc_labels_df.iterrows():
        label = np.zeros(nC, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                label[species_to_idx[sp]] = 1.0
        start_sec = _parse_time_to_seconds(row["start"])
        out.append({
            "filename": row["filename"],
            "stem": Path(row["filename"]).stem,
            "start_frame": int(start_sec * frame_rate),
            "label": label,
        })
    return out


class MelCacheDataset(Dataset):
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
        # mel cache mirrors train_audio/<plabel>/<stem>.ogg → <plabel>/<stem>.npy
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
            for sec in _parse_secondary_labels(row.get("secondary_labels", "[]")):
                if sec in self.species_to_idx:
                    label[self.species_to_idx[sec]] = 1.0
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


print("Dataset class defined")"""))

# ── Train / Val utilities ──
cells.append(code_cell("train-utils", r"""# ══════════════════════════════════════════════════════════════
# TRAIN / VAL UTILITIES
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler,
                    mel_transform, spec_aug, mixup, loss_fn, scaler, cfg, epoch):
    model.train()
    total = 0.0; n = 0
    for batch_idx, (mel_db, labels) in enumerate(tqdm(loader, desc="  train", leave=False)):
        mel_db = mel_db.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # MixUp on raw mel_db (pre-resize)
        mel_db, labels = mixup(mel_db, labels)

        # Resize + normalize + 3ch
        mel = mel_transform(mel_db)
        # SpecAugment
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
    for mel_db, labels in tqdm(loader, desc="  valid", leave=False):
        mel_db = mel_db.to(DEVICE, non_blocking=True)
        mel = mel_transform(mel_db)
        with torch.amp.autocast("cuda"):
            out = model(mel)
        pp.append(out["clipwise_prob"].float().cpu().numpy())
        tt.append(labels.numpy())
    return np.concatenate(pp), np.concatenate(tt)


def compute_metrics(preds, targets):
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
            except ValueError:
                pass
    return {"macro_auc": float(np.mean(aucs)) if aucs else 0.0,
            "num_classes_evaluated": len(aucs)}


def split_train_val(train_df, cfg):
    # GroupShuffleSplit by author (avoids same author in both splits → leak guard).
    if "author" in train_df.columns and train_df["author"].notna().all():
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_ratio, random_state=cfg.seed)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["author"]))
    else:
        # Fallback random split
        rng = np.random.RandomState(cfg.seed)
        idx = rng.permutation(len(train_df))
        n_val = max(1, int(len(train_df) * cfg.val_ratio))
        va_idx = idx[:n_val]; tr_idx = idx[n_val:]
    tr = train_df.iloc[tr_idx].reset_index(drop=True)
    va = train_df.iloc[va_idx].reset_index(drop=True)
    return tr, va"""))

# ── Phase 1 training loop ──
cells.append(code_cell("train", r"""# ══════════════════════════════════════════════════════════════
# PHASE 1 TRAINING (single fold)
# ══════════════════════════════════════════════════════════════
train_df = pd.read_csv(TRAIN_CSV)
print(f"Train recordings: {len(train_df)}")

# Soundscape labeled segments
sc_segments = None
if cfg.include_soundscape_labels:
    sc_labels = pd.read_csv(SC_LABELS_CSV)
    frame_rate = cfg.sr / cfg.hop_length
    sc_segments = prepare_soundscape_segments(sc_labels, SPECIES_TO_IDX, frame_rate)
    print(f"Labeled SC segments: {len(sc_segments)}")

# 90/10 split
tr_df, va_df = split_train_val(train_df, cfg)
print(f"Train: {len(tr_df)} | Val: {len(va_df)}")

# Datasets
train_ds = MelCacheDataset(tr_df, SPECIES_TO_IDX, cfg,
                           soundscape_segments=sc_segments, mode="train")
val_ds = MelCacheDataset(va_df, SPECIES_TO_IDX, cfg,
                         soundscape_segments=None, mode="val")

# DataLoader: num_workers=0 で確実に動かす (Kaggle の SHM hang 回避)
# DataParallel との相性も悪いので一旦 single GPU に落とす
DEBUG_NUM_WORKERS = 0
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=DEBUG_NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                        num_workers=DEBUG_NUM_WORKERS, pin_memory=True)

# Model — DataParallel は一旦無効化 (Phase 1 動作確認後に戻す)
model = SEDModel(cfg).to(DEVICE)
print(f"Model on single GPU (DataParallel disabled for debug)")

mel_transform = MelCacheTransform(cfg).to(DEVICE)
spec_aug = SpecAugmentations(cfg).to(DEVICE)
mixup = MelMixUp(prob=cfg.mixup_prob, alpha=cfg.mixup_alpha)
loss_fn = ClipFrameCELoss(cfg.clip_loss_weight, cfg.frame_loss_weight).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
scaler = torch.amp.GradScaler("cuda")

WEIGHT_PATH = str(WEIGHT_DIR / "best.pth")
best_auc, best_epoch = 0.0, -1
log_rows = []

# ── DEBUG: smoke test before main loop ──
print(f"\n[DEBUG] dataset[0] load test...")
_t0 = time.time()
_s = train_ds[0]
print(f"[DEBUG] dataset[0] OK in {time.time()-_t0:.2f}s, mel.shape={tuple(_s[0].shape)}, label.sum={float(_s[1].sum()):.1f}")

print(f"[DEBUG] First batch fetch test...")
_t0 = time.time()
_iter = iter(train_loader)
_b = next(_iter)
print(f"[DEBUG] First batch OK in {time.time()-_t0:.2f}s, mel.shape={tuple(_b[0].shape)}")

print(f"[DEBUG] Forward+backward test...")
_t0 = time.time()
_mel_db = _b[0].to(DEVICE, non_blocking=True)
_lab = _b[1].to(DEVICE, non_blocking=True)
_mel = mel_transform(_mel_db)
_mel = spec_aug(_mel)
with torch.amp.autocast("cuda"):
    _out = model(_mel)
    _loss = loss_fn(_out, _lab)
scaler.scale(_loss).backward()
optimizer.zero_grad()
print(f"[DEBUG] Forward+backward OK in {time.time()-_t0:.2f}s, loss={_loss.item():.4f}")
del _iter, _b, _mel_db, _lab, _mel, _out, _loss
torch.cuda.empty_cache()

print(f'\n{"=" * 60}\nPHASE 1 | epochs={cfg.epochs}\n{"=" * 60}')
for epoch in range(cfg.epochs):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                              mel_transform, spec_aug, mixup, loss_fn,
                              scaler, cfg, epoch)
    preds, targets = validate(model, val_loader, mel_transform)
    metrics = compute_metrics(preds, targets)
    elapsed = time.time() - t0

    is_best = metrics["macro_auc"] > best_auc
    print(f'  Ep {epoch+1:02d}/{cfg.epochs}'
          f' | Loss={tr_loss:.4f}'
          f' | AUC={metrics["macro_auc"]:.4f} ({metrics["num_classes_evaluated"]} cls)'
          f' | {elapsed:.0f}s{"  <- best" if is_best else ""}')

    log_rows.append(dict(epoch=epoch+1, tr_loss=tr_loss,
                         va_auc=metrics["macro_auc"], time=elapsed))

    if is_best:
        best_auc = metrics["macro_auc"]; best_epoch = epoch
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            "epoch": epoch, "model_state_dict": state,
            "metrics": metrics, "cfg": cfg.__dict__, "species": SPECIES,
        }, WEIGHT_PATH)

print(f'\n{"=" * 60}\nBest AUC: {best_auc:.4f} @ Ep {best_epoch + 1}\n{"=" * 60}')
pd.DataFrame(log_rows).to_csv(str(LOG_DIR / "train.csv"), index=False)"""))

# ── Summary ──
cells.append(code_cell("summary", r"""# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
elapsed_total = time.time() - WALL_START
print(f"Wall time: {elapsed_total/60:.1f} min")
print(f"Best AUC: {best_auc:.4f}")

ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
print(f"Saved checkpoint: {WEIGHT_PATH}")
print(f"  epoch: {ckpt['epoch'] + 1}")
print(f"  macro_auc: {ckpt['metrics']['macro_auc']:.4f}")
print(f"  classes_eval: {ckpt['metrics']['num_classes_evaluated']}")
print(f"  cfg.backbone: {ckpt['cfg']['backbone']}")
print(f"  output: /kaggle/working/weights/best.pth + /kaggle/working/logs/train.csv")"""))

# ── Assemble ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb1_train_p1.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
