"""Generate exp013 fine-tuning notebook for Nikita 1st place backbone init.

Input data sources (Kaggle):
  - birdclef-2026 (competition data, ogg files)
  - nikitababich/birdclef2025-1st-place-ensemble (backbone init weights)

Strategy:
  - Load Nikita backbone weights (eca_nfnet_l0 / b3 / b0)
  - Replace head with fresh SED head (234-class)
  - Fine-tune on BC2026 train_audio + labeled SS
  - 10 epochs, BCE, mixup, SpecAug
  - Discriminative LR (backbone 1e-4, head 1e-3)
  - Mel params match Nikita exactly: n_mels=224, n_fft=4096, hop=1252
  - 20s window, 3-channel mel repeat
  - Resumable via kernel_source self-reference

Generates: nb_finetune_{model_key}.ipynb
  where model_key in {"eca_nfnet_l0", "b3", "b0_amphibia"}

Run: python _gen_nb_finetune.py <model_key>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Per-model config
MODEL_CONFIGS = {
    "eca_nfnet_l0": {
        "timm_name": "eca_nfnet_l0",
        "ckpt_filename": "eca_nfnet_l0.ra2_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_128_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_additional_data_full_data_22_seed_15_epoch.pt",
        "kaggle_slug": "birdclef2026-exp013-eca-nfnet-l0",
        "kaggle_title": "birdclef2026 exp013 eca nfnet l0",
        "batch_size": 12,  # large model, conservative
    },
    "b3": {
        "timm_name": "tf_efficientnet_b3.ns_jft_in1k",
        "ckpt_filename": "tf_efficientnet_b3.ns_jft_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_54_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_1_fold_25_epoch.pt",
        "kaggle_slug": "birdclef2026-exp013-b3",
        "kaggle_title": "birdclef2026 exp013 b3",
        "batch_size": 16,
    },
    "b0_amphibia": {
        "timm_name": "tf_efficientnet_b0.ns_jft_in1k",
        "ckpt_filename": "tf_efficientnet_b0.ns_jft_in1k_incest_amphibia_128_bs_0.0_drop_path_rate_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_full_data_22_seed_40_epoch.pt",
        "kaggle_slug": "birdclef2026-exp013-b0-amphibia",
        "kaggle_title": "birdclef2026 exp013 b0 amphibia",
        "batch_size": 24,
    },
}


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


def build_notebook(model_key):
    cfg = MODEL_CONFIGS[model_key]

    cells = []

    cells.append(md_cell(rf"""# exp013 — Fine-tune Nikita 1st place {model_key.upper()} on BC2026

Backbone init from `nikitababich/birdclef2025-1st-place-ensemble`.
Head replaced with fresh 234-class SED head for BC2026.
10 epochs, BCE, mixup + SpecAug, discriminative LR.

## Input
- competition: `birdclef-2026`
- dataset: `nikitababich/birdclef2025-1st-place-ensemble` (backbone init)
- (resume) kernel_source: `maekeso/{cfg["kaggle_slug"]}` (self)

## Output
- `fold0_best.pt` (best val ns22 macro AUC)
- `fold0_history.json`
- `sed_{model_key}_fold0.onnx` (when fold completes)

## Mel params (Nikita 完全一致)
- sr=32000, n_mels=224, n_fft=4096, hop=1252, fmin=0, fmax=16000, top_db=80
- 20秒 window, 3-channel mel repeat
- Image: (3, 224, 512)

## Backbone: `{cfg["timm_name"]}`
""", "hdr"))

    # ============================================================
    # Setup, resume, time tracking
    # ============================================================
    cells.append(code_cell(rf"""# ============================================================
# Setup + resume + time tracking
# ============================================================
!pip install -q timm

import os, sys, time, json, gc, random, shutil, math
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}, GPUs: {{torch.cuda.device_count()}}")

# ---- Wall clock guard ----
WALL_START = time.time()
SOFT_LIMIT_SEC  = 11 * 3600 + 30 * 60   # 11h30m
EXIT_MARGIN_SEC = 30 * 60                # 30 min cushion

def time_left():
    return SOFT_LIMIT_SEC - (time.time() - WALL_START)

def time_low():
    return time_left() < EXIT_MARGIN_SEC

# ---- Working dirs ----
WORK_DIR = Path("/kaggle/working"); WORK_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = WORK_DIR / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)

def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp)); os.replace(str(tmp), str(path))

# ---- Resume from prev kernel_source ----
PREV = Path("/kaggle/input/notebooks/maekeso/{cfg['kaggle_slug']}")
if PREV.exists():
    print(f"PREV_KERNEL: {{PREV}}")
    n = 0
    for src in PREV.rglob("*"):
        if not src.is_file(): continue
        rel = src.relative_to(PREV)
        if rel.name.startswith("__") or rel.suffix in (".log", ".html"):
            continue
        if "kernel-metadata.json" in str(rel):
            continue
        dst = WORK_DIR / rel
        if dst.exists(): continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst); n += 1
        except Exception as e:
            print(f"  copy err: {{e}}")
    print(f"  copied {{n}} files from prev kernel")
else:
    print("PREV_KERNEL not mounted (first run)")

print(f"WORK_DIR: {{sorted(p.name for p in WORK_DIR.iterdir())[:30]}}")
""", "setup"))

    # ============================================================
    # Config
    # ============================================================
    cells.append(code_cell(rf"""# ============================================================
# Config
# ============================================================
MODEL_KEY = "{model_key}"
TIMM_NAME = "{cfg['timm_name']}"
CKPT_FILENAME = "{cfg['ckpt_filename']}"
BATCH_SIZE = {cfg['batch_size']}

# Paths
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
TRAIN_AUDIO_DIR = COMP_DIR / "train_audio"
TRAIN_SC_DIR = COMP_DIR / "train_soundscapes"
TRAIN_CSV = COMP_DIR / "train.csv"
SC_LABEL_CSV = COMP_DIR / "train_soundscapes_labels.csv"
SAMPLE_SUB = COMP_DIR / "sample_submission.csv"
TAXONOMY = COMP_DIR / "taxonomy.csv"

# Nikita backbone ckpt
NIKITA_DIR = Path("/kaggle/input/birdclef2025-1st-place-ensemble")
if not NIKITA_DIR.exists():
    # fallback: search
    cands = list(Path("/kaggle/input").rglob(CKPT_FILENAME))
    assert len(cands) > 0, f"ckpt not found: {{CKPT_FILENAME}}"
    NIKITA_CKPT = cands[0]
else:
    NIKITA_CKPT = NIKITA_DIR / CKPT_FILENAME
    assert NIKITA_CKPT.exists(), f"ckpt missing: {{NIKITA_CKPT}}"
print(f"Nikita ckpt: {{NIKITA_CKPT}} ({{NIKITA_CKPT.stat().st_size/1e6:.1f}}MB)")

# Mel params (Nikita 完全一致)
SR        = 32000
N_MELS    = 224
N_FFT     = 4096
HOP       = 1252
FMIN      = 0
FMAX      = 16000
TOP_DB    = 80.0
WINDOW_SEC = 20
WINDOW_SAMPLES = SR * WINDOW_SEC  # 640_000
N_FRAMES  = WINDOW_SAMPLES // HOP + 1  # ~512 frames

# Train hyperparameters
EPOCHS = 10
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WD = 1e-4
WARMUP_EPOCHS = 1
NUM_CLASSES = 234

# Augmentation
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
SPEC_FREQ_MASK = 24
SPEC_TIME_MASK = 64
AUG_GAIN_DB = (-6.0, 6.0)
AUG_NOISE_SNR = (10.0, 30.0)
AUG_PROB = 0.5

# Source mix (Tucker style)
SHARES = {{"focal": 0.9, "sc": 0.1}}

# Validation
SC_VAL_FRAC = 0.5  # half of labeled SS files for validation

print(f"Model: {{TIMM_NAME}} | Batch: {{BATCH_SIZE}} | Epochs: {{EPOCHS}}")
print(f"Mel: n_mels={{N_MELS}}, n_fft={{N_FFT}}, hop={{HOP}}, window={{WINDOW_SEC}}s")
""", "config"))

    # ============================================================
    # Data load
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Load data
# ============================================================
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"NUM_CLASSES: {NUM_CLASSES}")

# Train audio metadata
train_df = pd.read_csv(TRAIN_CSV)
# filter to existing labels
train_df = train_df[train_df["primary_label"].astype(str).isin(LABEL2IDX)].reset_index(drop=True)
print(f"train_audio: {len(train_df)} files")

# Soundscape labels
sc_labels = pd.read_csv(SC_LABEL_CSV)
print(f"SC labeled rows: {len(sc_labels)}")
# Build (filename, start_sec) -> label vector
sc_labels["start_sec"] = pd.to_timedelta(sc_labels["start"]).dt.total_seconds().astype(int)
sc_files = sc_labels["filename"].unique().tolist()
print(f"SC labeled files: {len(sc_files)}")

# SC validation split (half files)
rng = np.random.default_rng(SEED)
sc_files_shuffled = sc_files.copy(); rng.shuffle(sc_files_shuffled)
n_val = max(2, int(len(sc_files_shuffled) * SC_VAL_FRAC))
val_sc_files = set(sc_files_shuffled[:n_val])
train_sc_files = set(sc_files_shuffled[n_val:])
print(f"  val SC files: {len(val_sc_files)}, train SC files: {len(train_sc_files)}")

# Build SC label matrix
sc_label_records = []
for _, row in sc_labels.iterrows():
    fname = row["filename"]
    start = int(row["start_sec"])
    label_str = str(row["primary_label"])
    labels = [l.strip() for l in label_str.split(";") if l.strip() in LABEL2IDX]
    if labels:
        sc_label_records.append({
            "filename": fname, "start_sec": start, "labels": labels,
            "is_val": fname in val_sc_files,
        })
sc_label_df = pd.DataFrame(sc_label_records)
print(f"SC label records: {len(sc_label_df)}, val={sc_label_df['is_val'].sum()}")

# Validate audio file existence
def audio_path(filename):
    # filename in train.csv is like "{taxon_id}/{stem}.ogg"
    return TRAIN_AUDIO_DIR / filename

def sc_audio_path(filename):
    return TRAIN_SC_DIR / filename

# Spot check
ok = sum(1 for fn in train_df["filename"].head(5) if audio_path(fn).exists())
print(f"train_audio existence (5/5 expected): {ok}")
ok = sum(1 for fn in sc_files[:5] if sc_audio_path(fn).exists())
print(f"train_sc existence (5/5 expected): {ok}")
""", "load_data"))

    # ============================================================
    # Mel spectrogram + augmentation
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Mel spectrogram (GPU) + SpecAug
# ============================================================
class MelTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

    def forward(self, wav):
        # wav: (B, T) -> (B, n_mels, frames)
        m = self.db(self.mel(wav))
        # 0-1 normalize per-instance (Nikita 流)
        m_min = m.amin(dim=(-2, -1), keepdim=True)
        m_max = m.amax(dim=(-2, -1), keepdim=True)
        m = (m - m_min) / (m_max - m_min + 1e-6)
        # repeat to 3 channels
        m = m.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, n_mels, frames)
        return m


class SpecAug(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq = torchaudio.transforms.FrequencyMasking(freq_mask_param=SPEC_FREQ_MASK)
        self.time = torchaudio.transforms.TimeMasking(time_mask_param=SPEC_TIME_MASK)

    def forward(self, mel):
        return self.time(self.time(self.freq(mel)))


# Wave aug (CPU, applied before mel)
def apply_wave_aug(w):
    if np.random.random() < AUG_PROB:
        gain_db = np.random.uniform(*AUG_GAIN_DB)
        w = w * (10 ** (gain_db / 20))
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            snr_db = np.random.uniform(*AUG_NOISE_SNR)
            noise_p = sp / (10 ** (snr_db / 10))
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(noise_p)
    return w


print("Mel + augmentation ready")
""", "mel_aug"))

    # ============================================================
    # Dataset
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Datasets (ogg direct decode)
# ============================================================
import soundfile as sf
import librosa


def load_audio(path, target_sr=SR):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  audio load err {path}: {e}")
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)


def crop_or_pad(wav, n_samples, mode="center"):
    L = len(wav)
    if L >= n_samples:
        if mode == "random":
            start = np.random.randint(0, L - n_samples + 1)
        else:
            start = (L - n_samples) // 2
        return wav[start:start + n_samples]
    else:
        # left-pad with zeros (Nikita 流)
        return np.pad(wav, (n_samples - L, 0))


# Audio cache (LRU)
_FOCAL_CACHE = {}
_SC_CACHE = {}

def cached_load_focal(filename):
    if filename in _FOCAL_CACHE:
        return _FOCAL_CACHE[filename]
    w = load_audio(audio_path(filename))
    if len(_FOCAL_CACHE) >= 1500:
        _FOCAL_CACHE.pop(next(iter(_FOCAL_CACHE)))
    _FOCAL_CACHE[filename] = w
    return w

def cached_load_sc(filename):
    if filename in _SC_CACHE:
        return _SC_CACHE[filename]
    w = load_audio(sc_audio_path(filename))
    if len(_SC_CACHE) >= 200:
        _SC_CACHE.pop(next(iter(_SC_CACHE)))
    _SC_CACHE[filename] = w
    return w


def build_label_vec(primary, secondary):
    lb = np.zeros(NUM_CLASSES, dtype=np.float32)
    if str(primary) in LABEL2IDX:
        lb[LABEL2IDX[str(primary)]] = 1.0
    if isinstance(secondary, str) and secondary not in ("", "[]"):
        try:
            sec_list = eval(secondary) if isinstance(secondary, str) else []
            for s in sec_list:
                if str(s) in LABEL2IDX:
                    lb[LABEL2IDX[str(s)]] = 1.0
        except Exception:
            pass
    return lb


class FocalDS(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        w = cached_load_focal(row["filename"])
        chunk = crop_or_pad(w, WINDOW_SAMPLES, mode="random" if self.train else "center")
        if self.train:
            chunk = apply_wave_aug(chunk)
        label = build_label_vec(row["primary_label"], row.get("secondary_labels", "[]"))
        return torch.from_numpy(chunk).float(), torch.from_numpy(label), "focal"


class SCDS(Dataset):
    def __init__(self, label_df, train=True):
        self.df = label_df[~label_df["is_val"]].reset_index(drop=True) if train else label_df[label_df["is_val"]].reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        w = cached_load_sc(row["filename"])
        # window = [start_sec, start_sec + 5]; expand to 20s centered
        center = (row["start_sec"] + 2.5) * SR
        s = max(0, int(center - WINDOW_SAMPLES // 2))
        e = s + WINDOW_SAMPLES
        if e > len(w):
            e = len(w); s = max(0, e - WINDOW_SAMPLES)
        chunk = crop_or_pad(w[s:e], WINDOW_SAMPLES, mode="center")
        if self.train:
            chunk = apply_wave_aug(chunk)
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for l in row["labels"]:
            label[LABEL2IDX[l]] = 1.0
        return torch.from_numpy(chunk).float(), torch.from_numpy(label), "sc"


# Multi-source sampler (Tucker style: 90% focal, 10% SC)
class MixSampler(torch.utils.data.Sampler):
    def __init__(self, focal_n, sc_n, batch_size, n_steps, shares=SHARES, seed=0):
        self.focal_n = focal_n
        self.sc_n = sc_n
        self.bs = batch_size
        self.nst = n_steps
        per_src = [
            max(1, int(round(batch_size * shares["focal"]))),
            max(1, int(round(batch_size * shares["sc"]))),
        ]
        if sum(per_src) != batch_size:
            per_src[0] += (batch_size - sum(per_src))
        self.per_src = per_src  # [focal, sc]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.nst

    def __iter__(self):
        focal_off = 0
        sc_off = self.focal_n
        for _ in range(self.nst):
            batch = []
            if self.per_src[0] > 0 and self.focal_n > 0:
                ix = self.rng.integers(0, self.focal_n, size=self.per_src[0])
                batch.extend([focal_off + int(i) for i in ix])
            if self.per_src[1] > 0 and self.sc_n > 0:
                ix = self.rng.integers(0, self.sc_n, size=self.per_src[1])
                batch.extend([sc_off + int(i) for i in ix])
            self.rng.shuffle(batch)
            yield batch


def collate(batch):
    waves = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    sources = [b[2] for b in batch]
    return waves, labels, sources


print("Datasets ready")
""", "data_pipeline"))

    # ============================================================
    # Model: backbone (Nikita ckpt) + SED head
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Model: Nikita backbone + SED head
# ============================================================
class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class SEDHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512):
        super().__init__()
        self.gem = GeMFreqPool(3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden, num_classes, 1)
        self.cla = nn.Conv1d(hidden, num_classes, 1)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.cla.weight)
        self.att.bias.data.fill_(0.)
        self.cla.bias.data.fill_(0.)

    def forward(self, feat):
        # feat: (B, C, F, T)
        h = self.gem(feat)             # (B, C, T)
        h = h.permute(0, 2, 1)
        h = self.dense(h)              # (B, T, 512)
        h = h.permute(0, 2, 1)         # (B, 512, T)
        norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        framewise = self.cla(h)
        clip = torch.sum(norm_att * framewise, dim=2)
        return clip, framewise.permute(0, 2, 1)


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes, ckpt_path=None,
                 in_chans=3, drop_path=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="", drop_path_rate=drop_path)
        with torch.no_grad():
            dummy = torch.randn(1, in_chans, N_MELS, N_FRAMES)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
            print(f"backbone out: {tuple(feat.shape)}, dim={self.backbone_dim}")

        self.head = SEDHead(self.backbone_dim, num_classes)

        if ckpt_path is not None:
            self._load_backbone_ckpt(ckpt_path)

    def _load_backbone_ckpt(self, ckpt_path):
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"torch.load err: {e}")
            return
        # Various key formats; try to extract backbone-relevant weights
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        # remap keys: Nikita might use prefix "backbone." or "model."
        new_sd = {}
        for k, v in sd.items():
            for prefix in ["backbone.", "model.backbone.", "model."]:
                if k.startswith(prefix):
                    new_sd[k[len(prefix):]] = v
                    break
            else:
                new_sd[k] = v

        missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
        print(f"  backbone load: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"  sample missing: {missing[:3]}")
        if unexpected[:3]:
            print(f"  sample unexpected: {unexpected[:3]}")

    def forward(self, mel, return_framewise=True):
        feat = self.backbone(mel)
        clip, framewise = self.head(feat)
        if return_framewise:
            return clip, framewise
        return clip


# Build model
model = BirdSEDModel(
    backbone_name=TIMM_NAME,
    num_classes=NUM_CLASSES,
    ckpt_path=NIKITA_CKPT,
    in_chans=3,
    drop_path=0.0,
).to(device)

# Discriminative LR groups
backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
print(f"backbone params: {sum(p.numel() for p in backbone_params)/1e6:.2f}M")
print(f"head params: {sum(p.numel() for p in head_params)/1e6:.2f}M")
""", "model"))

    # ============================================================
    # Training utilities
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Training: optimizer + loss + eval
# ============================================================
from torch.amp import GradScaler, autocast


def compute_macro_auc(y_true, y_pred):
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def loss_clip_frame(clip_logit, framewise, target):
    bce_clip = F.binary_cross_entropy_with_logits(clip_logit, target)
    frame_max = framewise.max(dim=1).values
    bce_frame = F.binary_cross_entropy_with_logits(frame_max, target)
    return 0.5 * bce_clip + 0.5 * bce_frame


def mixup_apply(wave, label, alpha=MIXUP_ALPHA, prob=MIXUP_PROB):
    if torch.rand(1).item() > prob:
        return wave, label
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(wave.size(0), device=wave.device)
    wave_mix = lam * wave + (1.0 - lam) * wave[idx]
    label_mix = torch.maximum(label, label[idx])  # union
    return wave_mix, label_mix


# Build datasets
focal_train = FocalDS(train_df, train=True)
sc_train = SCDS(sc_label_df, train=True)
sc_val = SCDS(sc_label_df, train=False)
print(f"focal train={len(focal_train)}, sc train={len(sc_train)}, sc val={len(sc_val)}")

# Concat for training
train_ds = ConcatDataset([focal_train, sc_train])
val_loader = DataLoader(sc_val, batch_size=BATCH_SIZE * 2, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate)

# Optimizer
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params,     "lr": LR_HEAD},
], weight_decay=WD)

# Scheduler: warmup + cosine
n_steps_per_epoch = max(100, int(len(train_ds) / BATCH_SIZE))
total_steps = EPOCHS * n_steps_per_epoch
warmup_steps = WARMUP_EPOCHS * n_steps_per_epoch
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

scaler = GradScaler("cuda")
mel_transform = MelTransform().to(device)
spec_aug = SpecAug().to(device)

print(f"steps/ep: {n_steps_per_epoch}, total_steps: {total_steps}")
""", "train_setup"))

    # ============================================================
    # Validation function
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for wave, label, _src in loader:
        wave = wave.to(device, non_blocking=True)
        mel = mel_transform(wave)
        with autocast("cuda"):
            clip, framewise = model(mel, return_framewise=True)
            frame_max = framewise.max(dim=1).values
            p_clip = torch.sigmoid(clip)
            p_frame = torch.sigmoid(frame_max)
            p = 0.5 * p_clip + 0.5 * p_frame
        all_pred.append(p.cpu().numpy())
        all_true.append(label.numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    auc, n = compute_macro_auc(true, pred)
    return auc, n


print("Validation function ready")
""", "validate"))

    # ============================================================
    # Train loop with resume
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Train loop (resumable)
# ============================================================
HIST_PATH = WORK_DIR / f"fold0_history.json"
BEST_PATH = WORK_DIR / f"fold0_best.pt"
LAST_PATH = WORK_DIR / f"fold0_last.pt"

history = {"ep": [], "train_loss": [], "val_auc": [], "n_eval": []}
start_ep = 0
best_auc = -1.0

if HIST_PATH.exists():
    try:
        saved = json.loads(HIST_PATH.read_text())
        for k in history.keys():
            if k in saved:
                history[k] = list(saved[k])
        start_ep = (max(history["ep"]) + 1) if history["ep"] else 0
        best_auc = max(history["val_auc"]) if history["val_auc"] else -1.0
        print(f"RESUME from ep{start_ep}, best_auc={best_auc:.4f}")
    except Exception as e:
        print(f"resume failed: {e}")
        start_ep = 0

if LAST_PATH.exists() and start_ep > 0:
    try:
        ckpt = torch.load(str(LAST_PATH), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sch"])
        scaler.load_state_dict(ckpt["scaler"])
        print(f"loaded last.pt (ep {ckpt['epoch']})")
    except Exception as e:
        print(f"load last.pt failed: {e}, starting fresh")
        start_ep = 0

if start_ep >= EPOCHS:
    print(f"already finished {start_ep} epochs")
else:
    print(f"training ep {start_ep} to {EPOCHS}")
    for ep in range(start_ep, EPOCHS):
        if time_low():
            print(f"TIME LOW at start of ep{ep}, exiting cleanly")
            sys.exit(0)

        # Build sampler (different seed per ep)
        sampler = MixSampler(
            focal_n=len(focal_train), sc_n=len(sc_train),
            batch_size=BATCH_SIZE, n_steps=n_steps_per_epoch, seed=42 + ep)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, num_workers=2,
            pin_memory=True, collate_fn=collate)

        model.train()
        ep_loss, nb = 0.0, 0
        t0 = time.time()
        for wave, label, _src in train_loader:
            wave = wave.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            wave, label = mixup_apply(wave, label)

            with torch.no_grad():
                mel = mel_transform(wave)
                mel = spec_aug(mel)

            with autocast("cuda"):
                clip, framewise = model(mel, return_framewise=True)
                loss = loss_clip_frame(clip, framewise, label)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_loss += loss.item(); nb += 1

        # Validation
        val_auc, n_eval = evaluate(model, val_loader)
        history["ep"].append(ep)
        history["train_loss"].append(round(ep_loss / nb, 5))
        history["val_auc"].append(round(val_auc, 4) if not np.isnan(val_auc) else None)
        history["n_eval"].append(n_eval)

        tag = ""
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            atomic_save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                        BEST_PATH)
            tag = " *best"

        # Atomic save history + last.pt
        tmp_h = HIST_PATH.with_suffix(".tmp.json")
        tmp_h.write_text(json.dumps(history, indent=1))
        os.replace(str(tmp_h), str(HIST_PATH))

        torch.save({"epoch": ep, "model": model.state_dict(),
                    "opt": optimizer.state_dict(), "sch": scheduler.state_dict(),
                    "scaler": scaler.state_dict()}, str(LAST_PATH))

        elapsed = time.time() - t0
        lr_b = optimizer.param_groups[0]["lr"]
        lr_h = optimizer.param_groups[1]["lr"]
        val_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        print(f"  ep{ep:02d}: loss={ep_loss/nb:.4f} val_auc={val_str} (n={n_eval}) "
              f"lr_b={lr_b:.1e} lr_h={lr_h:.1e} [{elapsed:.0f}s]{tag} "
              f"time_left={time_left()/3600:.1f}h")

    # Cleanup last.pt
    if LAST_PATH.exists():
        LAST_PATH.unlink()
""", "train_loop"))

    # ============================================================
    # ONNX export
    # ============================================================
    cells.append(code_cell(rf"""# ============================================================
# ONNX export (best checkpoint)
# ============================================================
ONNX_PATH = WORK_DIR / f"sed_{model_key}_fold0.onnx"

if BEST_PATH.exists() and not ONNX_PATH.exists():
    # Load best
    best = torch.load(str(BEST_PATH), map_location=device, weights_only=False)
    model.load_state_dict(best)
    model.eval()

    # ONNX export wrapper (mel input directly)
    class ExportModel(nn.Module):
        def __init__(self, m, mel_t):
            super().__init__()
            self.backbone = m.backbone
            self.head = m.head
            self.mel_t = mel_t
        def forward(self, mel):
            # mel: (B, 3, n_mels, T)
            feat = self.backbone(mel)
            clip, framewise = self.head(feat)
            return clip, framewise

    export_model = ExportModel(model, mel_transform).to(device).eval()
    dummy = torch.randn(1, 3, N_MELS, N_FRAMES).to(device)

    torch.onnx.export(
        export_model, dummy, str(ONNX_PATH),
        input_names=["mel"],
        output_names=["clip_logits", "framewise_logits"],
        dynamic_axes={{"mel": {{0: "batch"}},
                       "clip_logits": {{0: "batch"}},
                       "framewise_logits": {{0: "batch"}}}},
        opset_version=17,
    )

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {{"mel": dummy.cpu().numpy()}})
    with torch.no_grad():
        ref_clip, ref_frame = export_model(dummy)
    diff = np.abs(ref_clip.cpu().numpy() - onnx_out[0]).max()
    print(f"ONNX max|diff|={{diff:.3e}}")
    print(f"Exported {{ONNX_PATH}} ({{ONNX_PATH.stat().st_size/1e6:.1f}}MB)")
elif ONNX_PATH.exists():
    print(f"ONNX already exists: {{ONNX_PATH}}")
else:
    print(f"BEST_PATH not yet, skipping ONNX export")
""", "onnx_export"))

    cells.append(md_cell(r"""## Done

Best checkpoint and ONNX saved. Use ONNX in inference NB.
""", "footer"))

    # Build notebook
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12",
                              "mimetype": "text/x-python",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                              "file_extension": ".py"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

    out_path = HERE / f"nb_finetune_{model_key}.ipynb"
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote: {out_path}  ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        if model_key not in MODEL_CONFIGS:
            print(f"unknown model_key: {model_key}; choices: {list(MODEL_CONFIGS)}")
            sys.exit(1)
        build_notebook(model_key)
    else:
        # build all 3
        for k in MODEL_CONFIGS:
            build_notebook(k)
