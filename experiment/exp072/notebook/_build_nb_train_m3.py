"""Build M3 train NB: eca_nfnet_l0 + 20s Babych mel + Babych init + Perch OFF + 3-fold + R3 pseudo.

M3 = ★ mel軸 core ★ (Babych pretrain transfer 検証 model)
Reference: framework/cfg_models.py M3 dict、M7 v2 template (proven taxon-aware logging)
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_train_m3.ipynb")
cells = []


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


# Cell 0: Title
cells.append(md_cell("""# exp072 (M3): eca_nfnet_l0 + 20s Babych mel + Babych iter3 init + 3-fold R3

**M-R3 mel軸 core (Babych pretrain transfer 検証 model)**

**Spec**:
  - Backbone: `eca_nfnet_l0.ra2_in1k`
  - Init: ★ Babych BC25 iter3 eca_nfnet_l0 ckpt (`strict=False`) ★
  - Mel: 20s × 224 × 4096 × 1252 (Babych spec)
  - Output: 234 classes (full BC26)
  - Loss: BCE clip + framewise max + ★ Hybrid pseudo distill ★
  - Aug: MixUp (ratio 1.0 + blend 0.5 fixed、Babych spec)
  - drop_path: 0.15 (Babych R2+ noise)
  - Pseudo: exp069c `pseudo_hybrid_234.npz` (overlap=Babych + non-overlap=exp048)
  - Power transform k = 1.54 (Babych iter2 spec)
  - 3-fold StratifiedKFold + 20 epoch

**Logging**: framework 統一 format (step / epoch summary / taxon AUC / class stats / BEST)

**Output**: m3_fold{0,1,2}_ckpt_best.pth + ONNX export per fold + history JSON
"""))


# Cell 1: Setup
cells.append(code_cell("""!pip install onnxruntime --quiet
import sys
print(f"Python: {sys.version[:50]}")
"""))


# Cell 2: Imports
cells.append(code_cell("""import os, time, json, gc, math, random
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tqdm.auto as tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, torch {torch.__version__}, timm {timm.__version__}")
START = time.time()
"""))


# Cell 3: CFG
cells.append(code_cell("""# M3 CFG (Babych spec)
SR = 32_000
WINDOW_SEC = 20                     # ★ Babych 20s
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS_PSEUDO = 12               # pseudo file 12 chunks (test_soundscape format)
WINDOW_SEC_PSEUDO = 5

# Mel (Babych spec)
N_MELS = 224
N_FFT = 4096
HOP_LENGTH = 1252
F_MIN = 0
F_MAX = 16000
TOP_DB = 80

# Training
N_FOLDS = 3
N_EPOCHS = 20
BATCH_SIZE = 24                     # 20s mel で memory 配慮
LR = 5e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
DROP_PATH = 0.15                    # ★ Babych R2+ noise

# Pseudo
PSEUDO_WEIGHT = 0.20                # focal/sc/pseudo = 0.70/0.10/0.20 spec
PSEUDO_POWER_K = 1.54               # Babych iter2 spec
PSEUDO_LOSS_WEIGHT = 0.5            # pseudo BCE / hard BCE blend

# Aug
MIXUP_BLEND_FIXED = 0.5             # ★ Babych: lambda 固定 0.5
MIXUP_P = 1.0                       # ★ 100% mixup (Babych spec)
SPECAUG_FREQ = 10
SPECAUG_TIME = 10

# Validation
STEP_LOG_INTERVAL = 100
NS22_K = 22

SEED = 42

# Paths
_data_path_candidates = [
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
]
DATA_PATH = None
for _p in _data_path_candidates:
    if Path(_p).exists():
        DATA_PATH = _p; break
assert DATA_PATH is not None

TRAIN_CSV = Path(DATA_PATH) / "train.csv"
TRAIN_AUDIO_DIR = Path(DATA_PATH) / "train_audio"
TRAIN_SC_DIR = Path(DATA_PATH) / "train_soundscapes"
TRAIN_SC_LABELS_CSV = Path(DATA_PATH) / "train_soundscapes_labels.csv"
TAXONOMY_CSV = Path(DATA_PATH) / "taxonomy.csv"

# Babych nfnet_l0 weight
BABYCH_DIR = None
for _p in ["/kaggle/input/birdclef2025-1st-place-ensemble", "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble"]:
    if Path(_p).exists():
        BABYCH_DIR = Path(_p); break
assert BABYCH_DIR is not None
BABYCH_NFNET_CKPT = None
for f in BABYCH_DIR.glob("eca_nfnet_l0*.pt"):
    BABYCH_NFNET_CKPT = f; break
assert BABYCH_NFNET_CKPT is not None
print(f"Babych ckpt: {BABYCH_NFNET_CKPT.name}")

# exp069c pseudo
EXP069C_DIR = None
for _p in ["/kaggle/input/birdclef2026-exp069c-hybrid-merge", "/kaggle/input/notebooks/maekeso/birdclef2026-exp069c-hybrid-merge"]:
    if Path(_p).exists():
        EXP069C_DIR = Path(_p); break
assert EXP069C_DIR is not None, "exp069c output not attached"
print(f"exp069c dir: {EXP069C_DIR}")

OUT_DIR = Path("/kaggle/working")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
"""))


# Cell 4: Data + taxonomy
cells.append(code_cell("""# Load BC26 taxonomy (full 234 species)
taxo = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = taxo["primary_label"].astype(str).tolist()
N_CLASSES = len(PRIMARY_LABELS)
assert N_CLASSES == 234
LABEL2IDX = {label: i for i, label in enumerate(PRIMARY_LABELS)}

label_to_taxon = dict(zip(taxo["primary_label"].astype(str), taxo["class_name"].astype(str)))
TAXON_MASKS = {
    t: np.array([i for i, lbl in enumerate(PRIMARY_LABELS) if label_to_taxon.get(lbl, "") == t])
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
}
print(f"Taxon counts: " + " ".join([f"{t}={len(m)}" for t, m in TAXON_MASKS.items()]))

# train.csv
train_df = pd.read_csv(TRAIN_CSV)
train_df["primary_label"] = train_df["primary_label"].astype(str)
train_df = train_df[train_df["primary_label"].isin(LABEL2IDX)].reset_index(drop=True)
train_df["exists"] = train_df["filename"].map(lambda fn: (TRAIN_AUDIO_DIR / fn).exists())
train_df = train_df[train_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
print(f"train.csv: {len(train_df)} focal recordings")

# 3-fold StratifiedKFold by primary_label
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fi, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fi
print(f"Fold dist: {train_df['fold'].value_counts().sort_index().to_dict()}")
"""))


# Cell 5: Hybrid pseudo loader
cells.append(code_cell("""# Load Hybrid pseudo (from exp069c)
pseudo_npz = dict(np.load(EXP069C_DIR / "pseudo_hybrid_234.npz", allow_pickle=True))
pseudo_probs = pseudo_npz["probs"].astype(np.float32)   # (n_files, 12, 234) probabilities
pseudo_file_ids = pseudo_npz["file_ids"]
print(f"Hybrid pseudo: {pseudo_probs.shape}, n_files={len(pseudo_file_ids)}")

# Apply power transform k=1.54 (Babych iter2)
pseudo_probs_k = pseudo_probs ** PSEUDO_POWER_K
print(f"Power transform k={PSEUDO_POWER_K} applied, mean={pseudo_probs_k.mean():.4f}")

# Build pseudo dataset index (each (file_id, chunk_idx) row is a sample)
N_PSEUDO_FILES = len(pseudo_file_ids)
N_PSEUDO_CHUNKS = N_PSEUDO_FILES * N_WINDOWS_PSEUDO   # 12 chunks per file
print(f"Total pseudo chunks: {N_PSEUDO_CHUNKS}")
"""))


# Cell 6: Model architecture (Babych SED)
cells.append(code_cell("""# Babych SED architecture (234 class output)
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
    def __init__(self, in_chans, p=0.5, num_class=N_CLASSES, hidden_dim=512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


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
    def __init__(self, backbone_name="eca_nfnet_l0.ra2_in1k", num_classes=N_CLASSES, drop_path_rate=DROP_PATH):
        super().__init__()
        self.mel_spectr_generator = SpecFeatureExtractor()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            in_chans=3, drop_path_rate=drop_path_rate,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(in_chans=backbone_dim, num_class=num_classes)

    def forward(self, wav, return_framewise=False):
        spec = self.mel_spectr_generator(wav)
        spec3 = torch.stack([spec, spec, spec], 1)
        feat = self.backbone(spec3)[-1]
        head_output = self.head(feat)
        framewise_logit = head_output["framewise_logit"]
        clip_logit = framewise_logit.max(dim=-1).values
        if return_framewise:
            return clip_logit, framewise_logit.permute(0, 2, 1)
        return clip_logit


def make_model_with_babych_init():
    model = CLEFClassifierSED()
    state = torch.load(str(BABYCH_NFNET_CKPT), weights_only=True, map_location="cpu")
    # Filter: backbone + mel only (skip head with 206-class mismatch)
    fstate = {k: v for k, v in state.items() if k.startswith("backbone.") or k.startswith("mel_spectr_generator.")}
    msg = model.load_state_dict(fstate, strict=False)
    return model

_tmp = make_model_with_babych_init()
print(f"M3 model: {sum(p.numel() for p in _tmp.parameters())/1e6:.1f}M params (Babych init)")
del _tmp; gc.collect()
"""))


# Cell 7: Datasets (Focal + Pseudo)
cells.append(code_cell("""# Datasets
class FocalDS(Dataset):
    def __init__(self, df, label2idx, train_audio_dir, train_mode=True):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.train_audio_dir = Path(train_audio_dir)
        self.train_mode = train_mode

    def __len__(self):
        return len(self.df)

    def load_audio(self, filename):
        try:
            y, _ = librosa.load(str(self.train_audio_dir / filename), sr=SR, mono=True)
            return y.astype(np.float32)
        except Exception:
            return np.zeros(SR * 5, dtype=np.float32)

    def crop(self, y):
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
        y = self.crop(y)
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
        return torch.from_numpy(y), torch.from_numpy(label), 1.0  # source_weight = 1.0 (hard)


class PseudoSCDS(Dataset):
    \"\"\"Pseudo dataset: each item = (file_id, chunk_idx) pair, returns 20s audio crop + soft label.\"\"\"
    def __init__(self, pseudo_file_ids, pseudo_probs_k, train_sc_dir, max_chunks_per_file=12):
        self.pseudo_file_ids = pseudo_file_ids
        self.pseudo_probs_k = pseudo_probs_k
        self.train_sc_dir = Path(train_sc_dir)
        self.max_chunks_per_file = max_chunks_per_file
        # Build (file_idx, chunk_idx) index
        self.index = [(fi, ci) for fi in range(len(pseudo_file_ids)) for ci in range(max_chunks_per_file)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, chunk_idx = self.index[idx]
        fid = str(self.pseudo_file_ids[file_idx])
        fpath = self.train_sc_dir / f"{fid}.ogg"
        # Load 60s, crop centered 20s around chunk position
        try:
            y, _ = librosa.load(str(fpath), sr=SR, mono=True)
        except Exception:
            y = np.zeros(SR * 60, dtype=np.float32)
        # 60s audio: chunk_idx * 5s = chunk center, 20s window = ±10s around chunk
        chunk_center_sec = (chunk_idx + 0.5) * 5
        win_half = WINDOW_SEC / 2
        start = max(0, int(chunk_center_sec - win_half) * SR)
        end = start + WINDOW_SAMPLES
        if end > len(y):
            y = np.pad(y, (0, max(0, end - len(y))))
        y = y[start:end].astype(np.float32)
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        elif len(y) > WINDOW_SAMPLES:
            y = y[:WINDOW_SAMPLES]
        m = np.abs(y).max()
        if m > 0: y = y / m
        soft_label = self.pseudo_probs_k[file_idx, chunk_idx].astype(np.float32)
        return torch.from_numpy(y), torch.from_numpy(soft_label), PSEUDO_LOSS_WEIGHT


print("FocalDS + PseudoSCDS defined")
"""))


# Cell 8: Augs (MixUp ratio 1.0 + blend 0.5)
cells.append(code_cell("""def mixup_audio_babych(wav, label, blend=MIXUP_BLEND_FIXED, p=MIXUP_P):
    if np.random.random() >= p:
        return wav, label
    idx = torch.randperm(wav.size(0), device=wav.device)
    return blend * wav + (1 - blend) * wav[idx], blend * label + (1 - blend) * label[idx]


class SpecAug(nn.Module):
    def __init__(self, freq=SPECAUG_FREQ, time_mask=SPECAUG_TIME):
        super().__init__()
        self.fa = T.FrequencyMasking(freq)
        self.ta = T.TimeMasking(time_mask)
    def forward(self, spec):
        return self.ta(self.fa(spec))
"""))


# Cell 9: Eval helpers (taxon-aware, framework 統一)
cells.append(code_cell("""def compute_per_species_auc(y_true, y_pred, class_mask=None):
    indices = range(y_true.shape[1]) if class_mask is None else class_mask
    aucs = []
    for c in indices:
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            auc = roc_auc_score(col, y_pred[:, c])
            aucs.append((int(c), float(auc)))
        except ValueError:
            continue
    return aucs


def macro_auc_from_list(aucs):
    return float(np.mean([a for _, a in aucs])) if len(aucs) > 0 else float("nan")


def lowest_k_mean(aucs, k=NS22_K):
    if len(aucs) == 0: return float("nan")
    sorted_aucs = sorted([a for _, a in aucs])
    k_eff = min(k, len(sorted_aucs))
    return float(np.mean(sorted_aucs[:k_eff]))


def class_stats_str(aucs):
    if len(aucs) == 0:
        return "n=0 median=nan p25=nan p75=nan #>0.5=0 #>0.7=0 #>0.9=0 #perfect=0"
    vals = np.array([a for _, a in aucs])
    return (f"n={len(vals)} median={np.median(vals):.3f} p25={np.percentile(vals,25):.3f} "
            f"p75={np.percentile(vals,75):.3f} "
            f"#>0.5={int((vals>0.5).sum())} #>0.7={int((vals>0.7).sum())} "
            f"#>0.9={int((vals>0.9).sum())} #perfect={int((vals>=1.0).sum())}")


def taxon_str(y_true, y_pred):
    parts = []
    for t in ["Insecta", "Reptilia", "Amphibia", "Mammalia", "Aves"]:
        mask = TAXON_MASKS[t]
        if len(mask) == 0:
            parts.append(f"{t}=nan"); continue
        aucs = compute_per_species_auc(y_true, y_pred, class_mask=mask)
        m = macro_auc_from_list(aucs)
        parts.append(f"{t}={m:.3f}" if not np.isnan(m) else f"{t}=nan")
    return "taxon: " + " ".join(parts)


@torch.no_grad()
def evaluate_on_val(model, val_dl, device):
    model.eval()
    all_preds, all_labels = [], []
    val_loss_sum = 0.0
    n_val = 0
    for wav, label, _ in val_dl:
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with autocast():
            clip_logit = model(wav)
            val_loss = F.binary_cross_entropy_with_logits(clip_logit, label)
        val_loss_sum += val_loss.item() * wav.size(0)
        n_val += wav.size(0)
        all_preds.append(torch.sigmoid(clip_logit).float().cpu().numpy())
        all_labels.append(label.cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels), val_loss_sum / max(n_val, 1))
"""))


# Cell 10: Training loop
cells.append(code_cell("""from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def train_fold(fold_k):
    print(f"\\n{'='*60}\\n[Fold {fold_k}] M3 training {N_EPOCHS} epochs (Babych init + Hybrid pseudo)\\n{'='*60}")
    t0_fold = time.time()

    tr_df = train_df[train_df["fold"] != fold_k].reset_index(drop=True)
    val_df = train_df[train_df["fold"] == fold_k].reset_index(drop=True)
    print(f"  train: {len(tr_df)}, val: {len(val_df)}")

    tr_focal = FocalDS(tr_df, LABEL2IDX, TRAIN_AUDIO_DIR, train_mode=True)
    pseudo_ds = PseudoSCDS(pseudo_file_ids, pseudo_probs_k, TRAIN_SC_DIR)
    print(f"  focal: {len(tr_focal)}, pseudo: {len(pseudo_ds)}")

    # ConcatDataset + WeightedRandomSampler (focal 0.80 / pseudo 0.20)
    combined = ConcatDataset([tr_focal, pseudo_ds])
    weights = np.concatenate([
        np.full(len(tr_focal), 0.80 / max(len(tr_focal), 1)),
        np.full(len(pseudo_ds), 0.20 / max(len(pseudo_ds), 1)),
    ])
    sampler = WeightedRandomSampler(weights, num_samples=len(tr_focal), replacement=True)
    tr_dl = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=2, pin_memory=True, drop_last=True)

    val_ds = FocalDS(val_df, LABEL2IDX, TRAIN_AUDIO_DIR, train_mode=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = make_model_with_babych_init().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_iters = WARMUP_EPOCHS * len(tr_dl)
    total_iters = N_EPOCHS * len(tr_dl)
    sched_warmup = LinearLR(optimizer, start_factor=1/25, end_factor=1.0, total_iters=warmup_iters)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=LR_MIN)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_iters])
    scaler = GradScaler()

    best_ns22 = -1.0
    history = []
    total_steps = len(tr_dl)

    for epoch in range(N_EPOCHS):
        t0_ep = time.time()
        model.train()
        tr_loss_sum = 0.0
        bce_sum = 0.0
        n_seen = 0
        for step, (wav, label, src_weight) in enumerate(tr_dl):
            wav = wav.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            src_weight = src_weight.to(DEVICE, non_blocking=True).float()
            wav, label = mixup_audio_babych(wav, label)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                clip_logit, framewise_logit = model(wav, return_framewise=True)
                frame_max_logit = framewise_logit.max(dim=1).values
                loss_clip = F.binary_cross_entropy_with_logits(clip_logit, label, reduction="none").mean(dim=1)
                loss_frame = F.binary_cross_entropy_with_logits(frame_max_logit, label, reduction="none").mean(dim=1)
                bce_per_sample = 0.5 * loss_clip + 0.5 * loss_frame
                bce = (bce_per_sample * src_weight).mean()
                distill = torch.tensor(0.0, device=DEVICE)  # M3: no Perch distill
                loss = bce + distill
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss_sum += loss.item() * wav.size(0)
            bce_sum += bce.item() * wav.size(0)
            n_seen += wav.size(0)

            if step % STEP_LOG_INTERVAL == 0 or step == total_steps - 1:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  [ep{epoch+1} step {step}/{total_steps}] loss={loss.item():.4f} bce={bce.item():.4f} distill=0.0000 lr={cur_lr:.2e}")

        tr_loss = tr_loss_sum / max(n_seen, 1)
        bce_avg = bce_sum / max(n_seen, 1)
        distill_avg = 0.0

        all_preds, all_labels, val_bce = evaluate_on_val(model, val_dl, DEVICE)
        per_species = compute_per_species_auc(all_labels, all_preds)
        val_macro = macro_auc_from_list(per_species)
        val_ns22 = lowest_k_mean(per_species, k=NS22_K)
        tax_line = taxon_str(all_labels, all_preds)
        cls_line = class_stats_str(per_species)

        ep_time = (time.time() - t0_ep) / 60
        total_time = (time.time() - START) / 60
        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = (not math.isnan(val_ns22)) and (val_ns22 > best_ns22)
        best_tag = "BEST " if is_best else ""

        print(f"=== Ep {epoch+1}/{N_EPOCHS}: loss={tr_loss:.4f} (bce={bce_avg:.4f} distill={distill_avg:.4f}) "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} {best_tag}lr={cur_lr:.2e} "
              f"({ep_time:.1f}min, total {total_time:.1f}min) ===")
        print(f"    {tax_line}")
        print(f"    class: {cls_line}")

        if is_best:
            best_ns22 = val_ns22
            torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "val_ns22": val_ns22, "val_macro": val_macro},
                       OUT_DIR / f"m3_fold{fold_k}_ckpt_best.pth")
            print(f"    BEST saved val_ns22={val_ns22:.4f}")

        history.append({
            "ep": epoch, "tr_loss": tr_loss, "bce": bce_avg, "distill": distill_avg,
            "val_ns22": val_ns22, "val_macro": val_macro, "val_bce": val_bce,
            "lr": cur_lr, "ep_time_min": ep_time, "best": is_best,
        })

    torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": N_EPOCHS - 1, "history": history},
               OUT_DIR / f"m3_fold{fold_k}_ckpt_final.pth")
    with open(OUT_DIR / f"m3_fold{fold_k}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Fold {fold_k}] DONE in {(time.time()-t0_fold)/60:.1f}min, best_ns22={best_ns22:.4f}")
    return best_ns22


fold_results = {}
for fold_k in range(N_FOLDS):
    bns22 = train_fold(fold_k)
    fold_results[fold_k] = bns22
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\\nAll folds DONE: {fold_results}")
print(f"Total time: {(time.time()-START)/60:.1f} min")
"""))


# Cell 11: ONNX export per fold
cells.append(code_cell("""# ONNX export per fold (for blend NB inference)
import onnxruntime as ort

class _M3ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, mel_input):
        # mel_input: (B, 3, n_mels, T) — pre-computed mel
        feat = self.model.backbone(mel_input)[-1]
        head_output = self.model.head(feat)
        framewise_logit = head_output["framewise_logit"]
        clip_logit = framewise_logit.max(dim=-1).values
        return clip_logit, framewise_logit.permute(0, 2, 1)


print("=== ONNX export per fold ===")
for fold_k in range(N_FOLDS):
    ckpt = torch.load(OUT_DIR / f"m3_fold{fold_k}_ckpt_best.pth", map_location="cpu", weights_only=False)
    model = make_model_with_babych_init()
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    wrapper = _M3ONNXWrapper(model)
    # Dummy mel input: (1, 3, 224, ~512)
    n_tf = WINDOW_SAMPLES // HOP_LENGTH + 1
    dummy_mel = torch.randn(1, 3, N_MELS, n_tf)
    onnx_path = OUT_DIR / f"m3_fold{fold_k}.onnx"
    try:
        torch.onnx.export(wrapper, dummy_mel, str(onnx_path),
                          input_names=["mel"], output_names=["clip_logit", "framewise"],
                          dynamic_axes={"mel": {0: "batch"}, "clip_logit": {0: "batch"}, "framewise": {0: "batch"}},
                          opset_version=17, do_constant_folding=True)
        print(f"  fold {fold_k}: ONNX exported ({onnx_path.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  fold {fold_k}: ONNX export FAILED ({str(e)[:100]})、fallback to .pth")
"""))


# Cell 12: Save summary
cells.append(code_cell("""with open(OUT_DIR / "m3_summary.json", "w") as f:
    json.dump({
        "n_classes": N_CLASSES,
        "n_folds": N_FOLDS,
        "n_epochs": N_EPOCHS,
        "backbone": "eca_nfnet_l0",
        "mel_window_sec": WINDOW_SEC,
        "init": "babych_iter3",
        "fold_best_ns22": fold_results,
        "total_time_min": (time.time() - START) / 60,
    }, f, indent=2)
print(f"OK M3 DONE: {sorted(OUT_DIR.glob('m3_*'))}")
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
print(f"Built: {OUT_PATH} ({len(cells)} cells)")
