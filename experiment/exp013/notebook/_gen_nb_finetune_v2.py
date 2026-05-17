"""Generate exp013 v2 fine-tune NB using Nikita's CLEFClassifierSED class.

Day-1 v1 failed because:
  - timm `features_only=True` uses `stem_conv1` naming, not `stem.conv1`
  - Nikita's ckpt is full model state_dict (mel_spec_generator + backbone + head)
  - Our v1 tried to load only backbone keys → 180 missing / 189 unexpected

v2 fix:
  - Embed Nikita's CLEFClassifierSED class verbatim
  - Load full ckpt via model.load_state_dict (strict=False, only head class
    layers replaced for 234-class)
  - Raw waveform input (B, 20*32000=640000); model does mel inside
  - 3-channel stack happens inside the model

Generates: nb_finetune_v2_{model_key}.ipynb

Run:
  python _gen_nb_finetune_v2.py <model_key>
  python _gen_nb_finetune_v2.py            # all 3
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

MODEL_CONFIGS = {
    "eca_nfnet_l0": {
        "timm_name": "timm/eca_nfnet_l0.ra2_in1k",
        "ckpt_filename": "eca_nfnet_l0.ra2_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_128_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_additional_data_full_data_22_seed_15_epoch.pt",
        "kaggle_slug": "birdclef2026-exp013-eca-nfnet-l0",
        "kaggle_title": "birdclef2026 exp013 eca nfnet l0",
        "batch_size": 12,
    },
    "b3": {
        "timm_name": "timm/tf_efficientnet_b3.ns_jft_in1k",
        "ckpt_filename": "tf_efficientnet_b3.ns_jft_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_54_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_1_fold_25_epoch.pt",
        "kaggle_slug": "birdclef2026-exp013-b3",
        "kaggle_title": "birdclef2026 exp013 b3",
        "batch_size": 16,
    },
    "b0_amphibia": {
        "timm_name": "timm/tf_efficientnet_b0.ns_jft_in1k",
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

    cells.append(md_cell(rf"""# exp013 v2 — Fine-tune Nikita 1st place {model_key.upper()} on BC2026

**Day-1 v1 failure root cause**: timm `features_only=True` uses `stem_conv1` naming
(not `stem.conv1`), and Nikita ckpts are full-model state_dicts (mel_spec_generator
+ backbone + head). v1 tried to load only backbone keys → 180 missing / 189 unexpected.

**v2 fix**: Embed Nikita's `CLEFClassifierSED` class verbatim. Load full ckpt with
`strict=False`, only replace head class layers for 234-class output.

## Input
- competition: `birdclef-2026`
- dataset: `nikitababich/birdclef2025-1st-place-ensemble`
- (resume) kernel_source: `maekeso/{cfg["kaggle_slug"]}` (self)

## Output
- `fold0_best.pt`, `fold0_history.json`, `sed_{model_key}_fold0.onnx`

## Backbone: `{cfg["timm_name"]}`
""", "hdr"))

    # ============================================================
    # Setup + resume
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
import torchaudio.transforms as T
import timm
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}, GPUs: {{torch.cuda.device_count()}}")

WALL_START = time.time()
SOFT_LIMIT_SEC  = 11 * 3600 + 30 * 60
EXIT_MARGIN_SEC = 30 * 60
def time_left(): return SOFT_LIMIT_SEC - (time.time() - WALL_START)
def time_low():  return time_left() < EXIT_MARGIN_SEC

WORK_DIR = Path("/kaggle/working"); WORK_DIR.mkdir(parents=True, exist_ok=True)
def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp)); os.replace(str(tmp), str(path))

# Resume from prev kernel_source
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
# Config (Nikita 完全互換)
# ============================================================
MODEL_KEY = "{model_key}"
TIMM_NAME = "{cfg['timm_name']}"
CKPT_FILENAME = "{cfg['ckpt_filename']}"
BATCH_SIZE = {cfg['batch_size']}

NUM_CLASSES = 234   # BC2026
EPOCHS = 10
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WD = 1e-4
WARMUP_EPOCHS = 1

# Nikita のオリジナル mel/window 設定 (cell 5 参照)
SR             = 32000
DURATION_SEC   = 20
WINDOW_SAMPLES = SR * DURATION_SEC   # 640_000
IMG_SIZE       = (224, 512)          # (n_mels, time_bins)
N_MELS         = 224
HOP_LENGTH     = (DURATION_SEC * SR) // (IMG_SIZE[1] - 1)   # 1252
N_FFT          = 4096
WIN_LENGTH     = 4096
F_MIN, F_MAX   = 0, 16000

NIKITA_SPECTROGRAM_CONFIG = {{
    "n_fft":      N_FFT,
    "hop_length": HOP_LENGTH,
    "win_length": WIN_LENGTH,
    "sample_rate": SR,
    "n_mels":      N_MELS,
    "f_min":       F_MIN,
    "f_max":       F_MAX,
    "normalized":  True,
    "top_db":      80,
    "sample_mel_normalize": "default",
    "output_size": None,
}}

# Source mix (Tucker-style focal:sc = 9:1)
SHARES = {{"focal": 0.9, "sc": 0.1}}

# Augmentation
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
SPEC_FREQ_MASK = 24
SPEC_TIME_MASK = 64
AUG_GAIN_DB = (-6.0, 6.0)
AUG_NOISE_SNR = (10.0, 30.0)
AUG_PROB = 0.5

# Validation: half of labeled SS files
SC_VAL_FRAC = 0.5

# Paths
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
TRAIN_AUDIO_DIR = COMP_DIR / "train_audio"
TRAIN_SC_DIR    = COMP_DIR / "train_soundscapes"
TRAIN_CSV       = COMP_DIR / "train.csv"
SC_LABEL_CSV    = COMP_DIR / "train_soundscapes_labels.csv"
SAMPLE_SUB      = COMP_DIR / "sample_submission.csv"

# Nikita ckpt
NIKITA_DIR = Path("/kaggle/input/birdclef2025-1st-place-ensemble")
if not NIKITA_DIR.exists():
    cands = list(Path("/kaggle/input").rglob(CKPT_FILENAME))
    assert cands, f"ckpt not found: {{CKPT_FILENAME}}"
    NIKITA_CKPT = cands[0]
else:
    NIKITA_CKPT = NIKITA_DIR / CKPT_FILENAME
    assert NIKITA_CKPT.exists(), f"ckpt missing: {{NIKITA_CKPT}}"
print(f"Nikita ckpt: {{NIKITA_CKPT}} ({{NIKITA_CKPT.stat().st_size/1e6:.1f}}MB)")

print(f"Mel: n_mels={{N_MELS}}, n_fft={{N_FFT}}, hop={{HOP_LENGTH}}, window={{DURATION_SEC}}s")
print(f"Image: {{IMG_SIZE}}, Batch: {{BATCH_SIZE}}, Epochs: {{EPOCHS}}")
""", "config"))

    # ============================================================
    # Nikita's model classes (verbatim copy)
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Nikita's BirdCLEF 2025 1st place model classes (copied verbatim)
# Source: nikitababich/birdclef2025-1st-place-inference (cell 3)
# ============================================================

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
    def __init__(self, in_chans, p=0.5, num_class=397, hidden_dim=512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_dim, num_class, 1, 1, 0, bias=True)
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, 1, 1, 0, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)   # (B, T, ch)
        feat = self.dense_layers(feat).permute(0, 2, 1)          # (B, 512, T)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


class NormalizeMelSpec(nn.Module):
    def __init__(self, norm_type="default", eps=1e-6, constant=80):
        super().__init__()
        self.eps, self.norm_type, self.constant = eps, norm_type, constant

    def forward(self, X):
        if self.norm_type == "default":
            mean = X.mean((1, 2), keepdim=True)
            std  = X.std((1, 2), keepdim=True)
            Xstd = (X - mean) / (std + self.eps)
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)
        elif self.norm_type == "top_db":
            return (X + 80) / 80
        elif self.norm_type == "constant":
            return X / self.constant


class SpecFeatureExtractor(nn.Module):
    def __init__(self, n_fft, hop_length, win_length=None,
                 sample_rate=200, f_max=20, f_min=0.5, n_mels=128,
                 top_db=120, normalized=False,
                 sample_mel_normalize=None, output_size=(256, 256)):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.append(
            T.MelSpectrogram(sample_rate=sample_rate, normalized=normalized,
                             n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                             f_max=f_max, n_mels=n_mels, f_min=f_min))
        self.feature_extractor.append(T.AmplitudeToDB(top_db=top_db))
        if sample_mel_normalize is not None:
            self.feature_extractor.append(NormalizeMelSpec(norm_type=sample_mel_normalize))
        self.resize = nn.UpsamplingBilinear2d(size=output_size) if output_size is not None else None

    def forward(self, x):
        img = self.feature_extractor(x)
        if self.resize is not None:
            img = self.resize(img.unsqueeze(1)).squeeze(1)
        return img


class CLEFClassifierSED(nn.Module):
    # Nikita's BC2025 1st place model. Adapted for fine-tuning (BC2026 234 classes).
    # forward(input): input=raw waveform (B, T), returns dict {"framewise_logit": (B, num_class, T_frames)}
    # v3: SpecAug applied to mel during training (was missing in v2)
    def __init__(self, config, spec_freq_mask=24, spec_time_mask=64):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor(**config['spectrogram'])
        self.backbone = timm.create_model(
            config['backbone']['backbone_name'],
            pretrained=config['backbone'].get('pretrained', False),
            features_only=True)
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim,
                            p=config['head']['dropout'],
                            num_class=config['head']['num_classes'])
        self.duration       = config['head'].get('duration', 20)
        self.infer_duration = config['head'].get('infer_duration', 5)
        # v3 fix: SpecAug applied to mel during training
        self.freq_mask = T.FrequencyMasking(freq_mask_param=spec_freq_mask)
        self.time_mask = T.TimeMasking(time_mask_param=spec_time_mask)

    def forward(self, input):
        # input: (B, T) raw waveform
        x = self.mel_spectr_generator(input)         # (B, n_mels, T_frames)
        # v3: SpecAug during training only
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
            x = self.time_mask(x)  # double time mask (Nikita default 2x time)
        x = torch.stack([x, x, x], 1)                # (B, 3, n_mels, T_frames)
        feats = self.backbone(x)                     # list of feature maps
        last = feats[-1]                             # (B, C, F', T')
        head_out = self.head(last)
        return head_out


print("Nikita model classes loaded (v3 with SpecAug)")
""", "nikita_classes"))

    # ============================================================
    # Build model + load Nikita ckpt
    # ============================================================
    cells.append(code_cell(rf"""# ============================================================
# Build model with Nikita config + load ckpt with strict=False
# (head class layers replaced for 234-class output)
# ============================================================

config = {{
    "spectrogram": NIKITA_SPECTROGRAM_CONFIG,
    "backbone": {{
        "backbone_name": TIMM_NAME,
        "pretrained": False,
    }},
    "head": {{
        "dropout": 0.5,
        "num_classes": NUM_CLASSES,
        "infer_duration": 5,
        "duration": DURATION_SEC,
    }},
}}

model = CLEFClassifierSED(config).to(device)

# Sanity probe
with torch.no_grad():
    dummy = torch.randn(1, WINDOW_SAMPLES).to(device)
    out = model(dummy)
    print(f"forward sanity: framewise_logit shape={{tuple(out['framewise_logit'].shape)}}")

# Load Nikita ckpt (full state_dict)
ckpt = torch.load(str(NIKITA_CKPT), map_location='cpu', weights_only=False)
print(f"ckpt total keys: {{len(ckpt)}}")

# Drop head class-dependent layers (will be re-initialized for 234-class)
keys_to_drop = [k for k in ckpt.keys()
                if k.startswith('head.fix_scale') or k.startswith('head.attention')]
for k in keys_to_drop:
    del ckpt[k]
print(f"  dropped head class-dep keys: {{len(keys_to_drop)}}")

missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"  load: missing={{len(missing)}}, unexpected={{len(unexpected)}}")
if missing[:5]:
    print(f"  sample missing: {{missing[:5]}}")
if unexpected[:5]:
    print(f"  sample unexpected: {{unexpected[:5]}}")

# Discriminative LR groups
backbone_params       = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
mel_gen_params        = [p for n, p in model.named_parameters() if n.startswith('mel_spectr_generator.')]
head_params           = [p for n, p in model.named_parameters() if n.startswith('head.')]
print(f"backbone: {{sum(p.numel() for p in backbone_params)/1e6:.2f}}M")
print(f"mel_gen:  {{sum(p.numel() for p in mel_gen_params)/1e3:.1f}}k")
print(f"head:     {{sum(p.numel() for p in head_params)/1e6:.2f}}M")
""", "build_model"))

    # ============================================================
    # Data load + dataset (raw waveform)
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Data: train.csv + soundscape labels, raw audio loaders
# ============================================================
sample_sub = pd.read_csv(SAMPLE_SUB, nrows=1)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == NUM_CLASSES

train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[train_df["primary_label"].astype(str).isin(LABEL2IDX)].reset_index(drop=True)
print(f"train_audio: {len(train_df)} files")

sc_labels = pd.read_csv(SC_LABEL_CSV)
sc_labels["start_sec"] = pd.to_timedelta(sc_labels["start"]).dt.total_seconds().astype(int)
sc_files = sorted(sc_labels["filename"].unique().tolist())
rng = np.random.default_rng(SEED)
sc_files_shuf = sc_files.copy(); rng.shuffle(sc_files_shuf)
n_val = max(2, int(len(sc_files_shuf) * SC_VAL_FRAC))
val_sc_files = set(sc_files_shuf[:n_val])
print(f"SC files: {len(sc_files)}, val: {len(val_sc_files)}")

sc_label_records = []
for _, row in sc_labels.iterrows():
    fname = row["filename"]; start = int(row["start_sec"])
    label_str = str(row["primary_label"])
    labels = [l.strip() for l in label_str.split(";") if l.strip() in LABEL2IDX]
    if labels:
        sc_label_records.append({
            "filename": fname, "start_sec": start, "labels": labels,
            "is_val": fname in val_sc_files,
        })
sc_label_df = pd.DataFrame(sc_label_records)
print(f"SC records: {len(sc_label_df)}, val={sc_label_df['is_val'].sum()}")


# ---- Raw audio loading ----
import soundfile as sf
import librosa

def load_audio(path, target_sr=SR):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
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
        return np.pad(wav, (n_samples - L, 0))   # left-pad (Nikita 流)


_FOCAL_CACHE = {}
_SC_CACHE = {}

def cached_load_focal(filename):
    if filename in _FOCAL_CACHE: return _FOCAL_CACHE[filename]
    w = load_audio(TRAIN_AUDIO_DIR / filename)
    if len(_FOCAL_CACHE) >= 1500:
        _FOCAL_CACHE.pop(next(iter(_FOCAL_CACHE)))
    _FOCAL_CACHE[filename] = w
    return w

def cached_load_sc(filename):
    if filename in _SC_CACHE: return _SC_CACHE[filename]
    w = load_audio(TRAIN_SC_DIR / filename)
    if len(_SC_CACHE) >= 200:
        _SC_CACHE.pop(next(iter(_SC_CACHE)))
    _SC_CACHE[filename] = w
    return w


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
        self.df = df.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        w = cached_load_focal(row["filename"])
        chunk = crop_or_pad(w, WINDOW_SAMPLES, mode="random" if self.train else "center")
        if self.train: chunk = apply_wave_aug(chunk)
        label = build_label_vec(row["primary_label"], row.get("secondary_labels", "[]"))
        return torch.from_numpy(chunk).float(), torch.from_numpy(label), "focal"


class SCDS(Dataset):
    def __init__(self, label_df, train=True):
        self.df = label_df[~label_df["is_val"]].reset_index(drop=True) if train \
                  else label_df[label_df["is_val"]].reset_index(drop=True)
        self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        w = cached_load_sc(row["filename"])
        center = (row["start_sec"] + 2.5) * SR
        s = max(0, int(center - WINDOW_SAMPLES // 2))
        e = s + WINDOW_SAMPLES
        if e > len(w):
            e = len(w); s = max(0, e - WINDOW_SAMPLES)
        chunk = crop_or_pad(w[s:e], WINDOW_SAMPLES, mode="center")
        if self.train: chunk = apply_wave_aug(chunk)
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for l in row["labels"]:
            label[LABEL2IDX[l]] = 1.0
        return torch.from_numpy(chunk).float(), torch.from_numpy(label), "sc"


class MixSampler(torch.utils.data.Sampler):
    def __init__(self, focal_n, sc_n, batch_size, n_steps, shares=SHARES, seed=0):
        self.focal_n, self.sc_n = focal_n, sc_n
        self.bs, self.nst = batch_size, n_steps
        per_src = [
            max(1, int(round(batch_size * shares["focal"]))),
            max(1, int(round(batch_size * shares["sc"]))),
        ]
        if sum(per_src) != batch_size:
            per_src[0] += (batch_size - sum(per_src))
        self.per_src = per_src
        self.rng = np.random.default_rng(seed)
    def __len__(self): return self.nst
    def __iter__(self):
        focal_off, sc_off = 0, self.focal_n
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


focal_train = FocalDS(train_df, train=True)
sc_train    = SCDS(sc_label_df, train=True)
sc_val      = SCDS(sc_label_df, train=False)
print(f"focal train={len(focal_train)}, sc train={len(sc_train)}, sc val={len(sc_val)}")

train_ds = ConcatDataset([focal_train, sc_train])
val_loader = DataLoader(sc_val, batch_size=BATCH_SIZE * 2, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate)
print("Datasets ready")
""", "data_pipeline"))

    # ============================================================
    # Training utilities + loss
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Training: optimizer, loss, eval, mixup
# ============================================================
from torch.amp import GradScaler, autocast


def compute_macro_auc(y_true, y_pred):
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col): continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def loss_clip_frame(framewise_logit, target):
    # framewise_logit: (B, num_class, T'); target: (B, num_class)
    frame_max = framewise_logit.max(dim=-1).values    # (B, num_class)
    frame_avg = framewise_logit.mean(dim=-1)          # (B, num_class) -- clip-like proxy
    bce_max = F.binary_cross_entropy_with_logits(frame_max, target)
    bce_avg = F.binary_cross_entropy_with_logits(frame_avg, target)
    return 0.5 * bce_max + 0.5 * bce_avg


def mixup_apply(wave, label, alpha=MIXUP_ALPHA, prob=MIXUP_PROB):
    if torch.rand(1).item() > prob: return wave, label
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(wave.size(0), device=wave.device)
    wave_mix = lam * wave + (1.0 - lam) * wave[idx]
    label_mix = torch.maximum(label, label[idx])
    return wave_mix, label_mix


optimizer = torch.optim.AdamW([
    {"params": backbone_params + mel_gen_params, "lr": LR_BACKBONE},
    {"params": head_params,                       "lr": LR_HEAD},
], weight_decay=WD)

n_steps_per_epoch = max(100, int(len(train_ds) / BATCH_SIZE))
total_steps  = EPOCHS * n_steps_per_epoch
warmup_steps = WARMUP_EPOCHS * n_steps_per_epoch
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

scaler = GradScaler("cuda")

# SpecAug applied on mel inside model — we hook after the mel_spectr_generator output
class SpecAug(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq = T.FrequencyMasking(freq_mask_param=SPEC_FREQ_MASK)
        self.time = T.TimeMasking(time_mask_param=SPEC_TIME_MASK)
    def forward(self, mel):
        return self.time(self.time(self.freq(mel)))

spec_aug = SpecAug().to(device)

print(f"steps/ep={n_steps_per_epoch}, total={total_steps}")
""", "train_setup"))

    # ============================================================
    # Validation
    # ============================================================
    cells.append(code_cell(r"""# ============================================================
# Validation: framewise_logit -> sigmoid -> mean over time as clip prob
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for wave, label, _src in loader:
        wave = wave.to(device, non_blocking=True)
        with autocast("cuda"):
            out = model(wave)
            framewise = out["framewise_logit"]    # (B, num_class, T')
            prob_max = torch.sigmoid(framewise).max(dim=-1).values
            prob_avg = torch.sigmoid(framewise).mean(dim=-1)
            p = 0.5 * prob_max + 0.5 * prob_avg
        all_pred.append(p.cpu().numpy())
        all_true.append(label.numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    return compute_macro_auc(true, pred)


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
        best_auc = max([v for v in history["val_auc"] if v is not None], default=-1.0)
        print(f"RESUME from ep{start_ep}, best_auc={best_auc:.4f}")
    except Exception as e:
        print(f"resume failed: {e}")
        start_ep = 0

if LAST_PATH.exists() and start_ep > 0:
    try:
        ckpt2 = torch.load(str(LAST_PATH), map_location=device, weights_only=False)
        model.load_state_dict(ckpt2["model"])
        optimizer.load_state_dict(ckpt2["opt"])
        scheduler.load_state_dict(ckpt2["sch"])
        scaler.load_state_dict(ckpt2["scaler"])
        print(f"loaded last.pt (ep {ckpt2['epoch']})")
    except Exception as e:
        print(f"load last.pt failed: {e}")
        start_ep = 0

if start_ep >= EPOCHS:
    print(f"already finished {start_ep} epochs")
else:
    print(f"training ep {start_ep} to {EPOCHS}")
    for ep in range(start_ep, EPOCHS):
        if time_low():
            print(f"TIME LOW at start of ep{ep}, exiting cleanly")
            sys.exit(0)

        sampler = MixSampler(focal_n=len(focal_train), sc_n=len(sc_train),
                             batch_size=BATCH_SIZE, n_steps=n_steps_per_epoch, seed=42 + ep)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2,
                                  pin_memory=True, collate_fn=collate)

        model.train()
        ep_loss, nb = 0.0, 0
        t0 = time.time()
        for wave, label, _src in train_loader:
            wave = wave.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            wave, label = mixup_apply(wave, label)

            with autocast("cuda"):
                out = model(wave)
                framewise = out["framewise_logit"]
                loss = loss_clip_frame(framewise, label)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_loss += loss.item(); nb += 1

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

        tmp = HIST_PATH.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(history, indent=1))
        os.replace(str(tmp), str(HIST_PATH))

        torch.save({"epoch": ep, "model": model.state_dict(),
                    "opt": optimizer.state_dict(), "sch": scheduler.state_dict(),
                    "scaler": scaler.state_dict()}, str(LAST_PATH))

        elapsed = time.time() - t0
        lr_b = optimizer.param_groups[0]["lr"]; lr_h = optimizer.param_groups[1]["lr"]
        val_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        print(f"  ep{ep:02d}: loss={ep_loss/nb:.4f} val_auc={val_str} (n={n_eval}) "
              f"lr_b={lr_b:.1e} lr_h={lr_h:.1e} [{elapsed:.0f}s]{tag} "
              f"time_left={time_left()/3600:.1f}h")

    if LAST_PATH.exists():
        LAST_PATH.unlink()
""", "train_loop"))

    # ============================================================
    # ONNX export (placeholder; submission NB handles inference details)
    # ============================================================
    cells.append(md_cell(rf"""## ONNX 出力

ONNX export は推論 NB で行う (Nikita は OpenVINO + 二段構成なので、別途処理)。
ここでは `fold0_best.pt` を出力するだけ。
""", "onnx_md"))

    cells.append(md_cell(r"""## 完了

`fold0_best.pt` が WORK_DIR に保存されている。これを Kaggle Dataset として登録 → 推論 NB で使う。
""", "footer"))

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

    out_path = HERE / f"nb_finetune_v2_{model_key}.ipynb"
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote: {out_path}  ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        if model_key not in MODEL_CONFIGS:
            print(f"unknown: {model_key}; choices: {list(MODEL_CONFIGS)}")
            sys.exit(1)
        build_notebook(model_key)
    else:
        for k in MODEL_CONFIGS:
            build_notebook(k)
