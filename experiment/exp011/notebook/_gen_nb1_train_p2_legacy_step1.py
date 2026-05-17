"""Generate exp011 Phase 2 Step 1 training notebook.

Step 1 changes from Phase 1:
- ogg direct read (no mel cache) + 20s chunks (was 10s mel cache)
- soft CE loss on multi-hot (was BCE clipwise + framewise max)
- raw waveform mixup (was spec mixup)
- Tucker dual val: Val-A=labeled SS hold-out, Val-B=train_audio author hold-out
- labeled SS partly held out for Val-A; rest still in training
- num_workers=4 with smoke test (Phase 1 hung at 0 -> we're more careful)
- target_size=(256, 384) for 20s temporal resolution
- batch_size=16

GPU: T4x2 (single GPU via DataParallel disabled — Phase 1 lesson).
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
cells.append(md_cell("hdr", """# exp011 Phase 2 Step 1: ogg + 20s + CE + raw mixup + dual val

Phase 1 -> Step 1 changes:
- **Input**: ogg direct read + **20-sec** chunks (was 10s mel cache)
- **Loss**: soft CE on multi-hot (was BCE clipwise + framewise max)
- **Aug**: raw waveform mixup + SpecAug (was spec mixup)
- **Dual val**: Val-A=labeled SS hold-out (16 files), Val-B=train_audio author hold-out (10%)
- **target_size**: (256, 384) for 20s temporal info
- **batch_size**: 16
- **num_workers**: 4 with smoke test

Goal: LB **0.87-0.89** (+0.02-0.04 vs Phase 1 v2 LB 0.854).
"""))

# ── Imports ──
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
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")
WALL_START = time.time()"""))

# ── Config ──
cells.append(code_cell("config", r"""# ==============================================================
# CONFIG (Phase 2 Step 1)
# ==============================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # Audio / Mel
    sr: int = 32_000
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 16_000
    top_db: float = 80.0
    db_min: float = -80.0
    db_max: float = 20.0

    # Chunk: 20 sec window
    chunk_duration: float = 20.0
    target_size: tuple = (256, 384)  # (H, W) — 384 for 20s temporal info

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
    batch_size: int = 16
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    num_workers: int = 4

    # Augmentation
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.5  # Beta(alpha, alpha)
    freq_mask_param: int = 30
    time_mask_param: int = 40

    # Loss
    clip_loss_weight: float = 0.5
    frame_loss_weight: float = 0.5

    # Data
    seed: int = 42
    val_b_ratio: float = 0.10
    val_a_files_keep_for_train: int = 50  # 50 of 66 labeled SS files in training, 16 in Val-A

    use_secondary_labels: bool = True

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
print(f"Backbone: {cfg.backbone}, Epochs: {cfg.epochs}, Batch: {cfg.batch_size}")"""))

# ── Paths & Species ──
cells.append(code_cell("paths", r"""# ==============================================================
# PATHS & SPECIES
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

sub_df = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
SPECIES = list(sub_df.columns[1:])
SPECIES_TO_IDX = {sp: i for i, sp in enumerate(SPECIES)}
NUM_CLASSES = len(SPECIES)
print(f"Species: {NUM_CLASSES}")
print(f"train_audio dir: {TRAIN_AUDIO_DIR.exists()}, train_soundscapes dir: {TRAIN_SC_DIR.exists()}")"""))

# ── Audio loading ──
cells.append(code_cell("audio", r"""# ==============================================================
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
    # Random crop, pad zeros if too short
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

# ── Raw waveform mixup ──
cells.append(code_cell("raw-mixup", r"""# ==============================================================
# RAW WAVEFORM MIXUP (Beta(alpha,alpha) lambda, label = max)
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

# ── Mel transform on GPU ──
cells.append(code_cell("mel-transform", r"""# ==============================================================
# MEL TRANSFORM (on GPU, raw waveform -> mel_db -> resize+norm+3ch)
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
            mel = self.mel_spec(audio)             # (B, n_mels, T)
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

# ── Model ──
cells.append(code_cell("model", r"""# ==============================================================
# MODEL: timm v2B0 + GeMFreqPool + AttentionSEDHead (same as Phase 1)
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

# ── Loss: soft CE on multi-hot ──
cells.append(code_cell("loss", r"""# ==============================================================
# LOSS: soft CE on multi-hot (Salman recipe; Sigmoid at infer)
# ==============================================================
class SoftCEClipFrameLoss(nn.Module):
    def __init__(self, clip_weight=0.5, frame_weight=0.5):
        super().__init__()
        self.cw = clip_weight; self.fw = frame_weight

    def forward(self, outputs, targets):
        # targets: (B, n_classes) multi-hot (after raw mixup label OR; values in {0,1})
        target_sum = targets.sum(dim=1, keepdim=True).clamp(min=1e-7)
        soft_targets = targets / target_sum

        clip_logit = outputs["clipwise_logit"]
        clip_logp = F.log_softmax(clip_logit, dim=1)
        loss_clip = -(soft_targets * clip_logp).sum(dim=1).mean()

        frame_max_logit = outputs["segmentwise_logit"].max(dim=1)[0]
        frame_logp = F.log_softmax(frame_max_logit, dim=1)
        loss_frame = -(soft_targets * frame_logp).sum(dim=1).mean()

        return self.cw * loss_clip + self.fw * loss_frame


print("SoftCEClipFrameLoss ready")"""))

# ── Dataset ──
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
    # train_audio rows + (optional) labeled-SS file-level positive labels.
    # mode='train': random 20s crop. mode='val': center 20s.
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
    # labeled SS segments. Chunk = 20s ending at start_sec+5 (matches inference).
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


def prepare_sc_segments(sc_labels_df, species_to_idx):
    out = []
    nC = len(species_to_idx)
    for _, row in sc_labels_df.iterrows():
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


print("Datasets ready")"""))

# ── Dual val split ──
cells.append(code_cell("dual-val-split", r"""# ==============================================================
# DUAL VAL SPLIT
#  - Val-A: 16 of 66 labeled SS files (held out from training)
#  - Val-B: 10% of train_audio by author group (held out)
#  - Train: train_audio author 90% + 50 labeled SS files (still in training)
# ==============================================================
train_df = pd.read_csv(TRAIN_CSV)
print(f"train_audio rows: {len(train_df)}")

sc_labels = pd.read_csv(SC_LABELS_CSV)
sc_segments_all = prepare_sc_segments(sc_labels, SPECIES_TO_IDX)
print(f"labeled SS segments total: {len(sc_segments_all)}")

# Hold out files by deterministic hash
sc_files_all = sorted(set(sc_labels["filename"]))
print(f"labeled SS files unique: {len(sc_files_all)}")
n_keep = cfg.val_a_files_keep_for_train  # 50
# Deterministic split: hash filename, smallest 16 -> Val-A
def _h(s): return int(hashlib.md5(str(s).encode()).hexdigest(), 16)
ranked = sorted(sc_files_all, key=_h)
val_a_files = set(ranked[n_keep:])  # 16 files for Val-A
train_sc_files = set(ranked[:n_keep])  # 50 in training
print(f"  Val-A files: {len(val_a_files)}, Train SS files: {len(train_sc_files)}")

val_a_segments = [s for s in sc_segments_all if s["filename"] in val_a_files]
train_sc_segments = [s for s in sc_segments_all if s["filename"] in train_sc_files]
print(f"  Val-A segments: {len(val_a_segments)}, Train SS segments: {len(train_sc_segments)}")

# train_audio author hold-out (Val-B)
if "author" in train_df.columns and train_df["author"].notna().all():
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_b_ratio, random_state=cfg.seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["author"]))
else:
    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(len(train_df))
    n_val = max(1, int(len(train_df) * cfg.val_b_ratio))
    va_idx = idx[:n_val]; tr_idx = idx[n_val:]
ta_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
ta_val_b_df = train_df.iloc[va_idx].reset_index(drop=True)
print(f"train_audio Train: {len(ta_train_df)} | Val-B: {len(ta_val_b_df)}")

# Build datasets
train_ds_audio = TrainAudioDataset(ta_train_df, SPECIES_TO_IDX, cfg, mode="train")
val_b_ds = TrainAudioDataset(ta_val_b_df, SPECIES_TO_IDX, cfg, mode="val")
train_ds_sc = SoundscapeSegmentDataset(train_sc_segments, cfg)
val_a_ds = SoundscapeSegmentDataset(val_a_segments, cfg)


# Concat train: audio + sc training segments
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


train_ds = ConcatDataset(train_ds_audio, train_ds_sc)
print(f"\nFinal sizes -> Train: {len(train_ds)} | Val-A: {len(val_a_ds)} | Val-B: {len(val_b_ds)}")"""))

# ── Train utilities ──
cells.append(code_cell("train-utils", r"""# ==============================================================
# TRAIN / VAL UTILITIES
# ==============================================================
def train_one_epoch(model, loader, optimizer, scheduler,
                    mel_transform, spec_aug, mixup, loss_fn, scaler, cfg, epoch):
    model.train()
    total = 0.0; n = 0
    for batch_idx, (audio, labels) in enumerate(tqdm(loader, desc="  train", leave=False)):
        audio = audio.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # Raw waveform mixup
        audio, labels = mixup(audio, labels)

        # Mel transform + SpecAug
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
def validate(model, loader, mel_transform, name=""):
    model.eval()
    pp, tt = [], []
    for audio, labels in tqdm(loader, desc=f"  val-{name}", leave=False):
        audio = audio.to(DEVICE, non_blocking=True)
        mel = mel_transform(audio)
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


print("Train utils ready")"""))

# ── Train loop ──
cells.append(code_cell("train", r"""# ==============================================================
# PHASE 2 STEP 1 TRAINING (early stop on Val-A AUC)
# ==============================================================
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=(cfg.num_workers > 0))
val_a_loader = DataLoader(val_a_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=(cfg.num_workers > 0))
val_b_loader = DataLoader(val_b_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=(cfg.num_workers > 0))

# Single GPU only (Phase 1 lesson: DataParallel hung on 0 epochs)
model = SEDModel(cfg).to(DEVICE)
print(f"Model on single GPU (DataParallel disabled)")

mel_transform = MelTransform(cfg).to(DEVICE)
spec_aug = SpecAugmentations(cfg).to(DEVICE)
mixup = RawMixUp(prob=cfg.mixup_prob, alpha=cfg.mixup_alpha)
loss_fn = SoftCEClipFrameLoss(cfg.clip_loss_weight, cfg.frame_loss_weight).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
scaler = torch.amp.GradScaler("cuda")

WEIGHT_PATH = str(WEIGHT_DIR / "best.pth")
best_auc_a, best_auc_b, best_epoch = 0.0, 0.0, -1
log_rows = []

# ── Smoke test ──
print(f"\n[DEBUG] dataset[0] load test...")
_t0 = time.time()
_s = train_ds[0]
print(f"[DEBUG] dataset[0] OK in {time.time()-_t0:.2f}s, audio.shape={tuple(_s[0].shape)}, label.sum={float(_s[1].sum()):.1f}")

print(f"[DEBUG] First batch fetch...")
_t0 = time.time()
_iter = iter(train_loader)
_b = next(_iter)
print(f"[DEBUG] First batch OK in {time.time()-_t0:.2f}s, audio.shape={tuple(_b[0].shape)}")

print(f"[DEBUG] Forward+backward test...")
_t0 = time.time()
_audio = _b[0].to(DEVICE, non_blocking=True)
_lab = _b[1].to(DEVICE, non_blocking=True)
_audio, _lab = mixup(_audio, _lab)
_mel = mel_transform(_audio); _mel = spec_aug(_mel)
with torch.amp.autocast("cuda"):
    _out = model(_mel)
    _loss = loss_fn(_out, _lab)
scaler.scale(_loss).backward()
optimizer.zero_grad()
print(f"[DEBUG] Forward+backward OK in {time.time()-_t0:.2f}s, loss={_loss.item():.4f}")
del _iter, _b, _audio, _lab, _mel, _out, _loss
torch.cuda.empty_cache()

print(f'\n{"=" * 60}\nPHASE 2 STEP 1 | epochs={cfg.epochs}\n{"=" * 60}')
for epoch in range(cfg.epochs):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                              mel_transform, spec_aug, mixup, loss_fn,
                              scaler, cfg, epoch)
    preds_a, targets_a = validate(model, val_a_loader, mel_transform, "A")
    preds_b, targets_b = validate(model, val_b_loader, mel_transform, "B")
    metrics_a = compute_metrics(preds_a, targets_a)
    metrics_b = compute_metrics(preds_b, targets_b)
    elapsed = time.time() - t0

    auc_a = metrics_a["macro_auc"]; auc_b = metrics_b["macro_auc"]
    is_best = auc_a > best_auc_a  # early stop on Val-A (Tucker 推奨)
    print(f'  Ep {epoch+1:02d}/{cfg.epochs}'
          f' | Loss={tr_loss:.4f}'
          f' | Val-A AUC={auc_a:.4f} ({metrics_a["num_classes_evaluated"]} cls)'
          f' | Val-B AUC={auc_b:.4f} ({metrics_b["num_classes_evaluated"]} cls)'
          f' | {elapsed:.0f}s{"  <- best (A)" if is_best else ""}')

    log_rows.append(dict(epoch=epoch+1, tr_loss=tr_loss,
                         val_a_auc=auc_a, val_a_cls=metrics_a["num_classes_evaluated"],
                         val_b_auc=auc_b, val_b_cls=metrics_b["num_classes_evaluated"],
                         time=elapsed))

    if is_best:
        best_auc_a = auc_a; best_auc_b = auc_b; best_epoch = epoch
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "metrics_val_a": metrics_a, "metrics_val_b": metrics_b,
            "cfg": cfg.__dict__, "species": SPECIES,
        }, WEIGHT_PATH)

print(f'\n{"=" * 60}\nBest Val-A AUC: {best_auc_a:.4f} (Val-B {best_auc_b:.4f}) @ Ep {best_epoch + 1}\n{"=" * 60}')
pd.DataFrame(log_rows).to_csv(str(LOG_DIR / "train.csv"), index=False)"""))

# ── Summary ──
cells.append(code_cell("summary", r"""# ==============================================================
# SUMMARY
# ==============================================================
elapsed_total = time.time() - WALL_START
print(f"Wall time: {elapsed_total/60:.1f} min")
print(f"Best Val-A AUC: {best_auc_a:.4f} (early stop metric)")
print(f"Best Val-B AUC: {best_auc_b:.4f}")

ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
print(f"\nSaved checkpoint: {WEIGHT_PATH}")
print(f"  epoch: {ckpt['epoch'] + 1}")
print(f"  val_a macro_auc: {ckpt['metrics_val_a']['macro_auc']:.4f} ({ckpt['metrics_val_a']['num_classes_evaluated']} cls)")
print(f"  val_b macro_auc: {ckpt['metrics_val_b']['macro_auc']:.4f} ({ckpt['metrics_val_b']['num_classes_evaluated']} cls)")
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

out_path = HERE / "nb1_train_p2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
