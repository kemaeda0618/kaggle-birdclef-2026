"""Generate Colab NB for Stage 1 XC pretrain.

Target environment: Colab Pro+ Blackwell (RTX PRO 6000, 96GB VRAM)
Data: maekeso/birdclef2026-xc-api-dl-part1 + part2 from Kaggle
Output: backbone state_dict to Drive
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

# Read embedded 162 Aves species (already extracted for exp047)
aves_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_aves_list.py").read_text(encoding="utf-8")
aves_lines = []
for line in aves_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        aves_lines.append("    " + s)
aves_block = "\n".join(aves_lines)

# Read embedded 234 BC2026 species (all taxa)
all_species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species.txt").read_text(encoding="utf-8")
all_species_lines = []
for line in all_species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        all_species_lines.append("    " + s)
all_species_block = "\n".join(all_species_lines)

nb = {
    "cells": [
        md_cell("hdr", """# exp052 Stage 1: XC Backbone Pretrain (Colab Blackwell)

**目的**: paper 256 流 XC pretrain (paper 256 評価 +0.046 LB の simplified version)。

## Data

| Source | files | size |
|---|---|---|
| Kaggle Dataset Part 1 | 13,399 mp3 | 17.6 GB |
| Kaggle Dataset Part 2 | 12,978 mp3 | 15.3 GB |
| **合計** | **26,377** | **32.9 GB** |

カバー: 159 BC2026 Aves species (162 中 -3 = XC 未登録)

## Design (paper 256 準拠)

| Item | Value |
|---|---|
| Backbone | `tf_efficientnetv2_s.in21k_ft_in1k` |
| Output | **234-class multi-label** (BC2026 全 taxonomy) |
| Loss | BCEWithLogitsLoss |
| Input | 5s random chunk, mel n_mels=128, sr=32000 |
| Aug | Spec mixup α=0.5 + SpecAugment + waveform mixup α=0.5 |
| Optimizer | AdamW lr=1e-3 wd=1e-4 (warmup 500 step) |
| Scheduler | CosineAnnealingLR T_max=30 |
| Epochs | **30** |
| Batch | **256** (Blackwell 96GB) |
| Label smoothing | 0.05 |

## Output

- Backbone state_dict to Drive (`stage1_backbone.pth`)
- history.json with rich log per epoch

## 期待

- Aves AUC 0.90-0.95 (BC train_audio より高密度な XC で学習)
- Backbone audio domain adapted、Stage 2 finetune 効率良
- Total time on Blackwell: **2.5-3.5h**
"""),

        code_cell("install", """# ============================================================
# Cell 1: Install + Drive mount + Kaggle API setup
# ============================================================
!pip install -q timm kaggle soundfile librosa torchaudio tqdm

from google.colab import drive
drive.mount("/content/drive")

import os, json
from pathlib import Path

# Drive paths
DRIVE_ROOT = Path("/content/drive/MyDrive/birdclef2026/exp052")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = DRIVE_ROOT / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)

# Kaggle credentials (upload kaggle.json to Drive or paste here)
KAGGLE_JSON_PATH = Path("/content/drive/MyDrive/birdclef2026/kaggle.json")
assert KAGGLE_JSON_PATH.exists(), f"Upload kaggle.json to {KAGGLE_JSON_PATH}"

os.makedirs(Path.home() / ".kaggle", exist_ok=True)
import shutil
shutil.copy(KAGGLE_JSON_PATH, Path.home() / ".kaggle" / "kaggle.json")
os.chmod(Path.home() / ".kaggle" / "kaggle.json", 0o600)
!kaggle datasets list -m | head -5
print("Kaggle API ready")
"""),

        code_cell("dl_data", """# ============================================================
# Cell 2: Download XC Datasets (Part 1 + Part 2)
# ============================================================
DATA_ROOT = Path("/content/xc_data")
DATA_ROOT.mkdir(exist_ok=True)

# Part 1
part1_dir = DATA_ROOT / "part1"
part2_dir = DATA_ROOT / "part2"

if not (part1_dir / "xc_api" / "_metadata.csv").exists():
    print("Downloading Part 1...")
    part1_dir.mkdir(exist_ok=True)
    !kaggle datasets download -d maekeso/birdclef2026-xc-api-dl-part1 -p {part1_dir} --unzip
    print("Part 1 done")

if not (part2_dir / "xc_api_part2" / "_metadata.csv").exists():
    print("Downloading Part 2...")
    part2_dir.mkdir(exist_ok=True)
    !kaggle datasets download -d maekeso/birdclef2026-xc-api-dl-part2 -p {part2_dir} --unzip
    print("Part 2 done")

# Verify
import pandas as pd
p1_meta = pd.read_csv(part1_dir / "xc_api" / "_metadata.csv")
p2_meta = pd.read_csv(part2_dir / "xc_api_part2" / "_metadata.csv")
print(f"Part 1: {len(p1_meta)} files, {p1_meta['scientific_name'].nunique()} species")
print(f"Part 2: {len(p2_meta)} files, {p2_meta['scientific_name'].nunique()} species")
"""),

        code_cell("setup", """# ============================================================
# Cell 3: Setup torch + GPU check
# ============================================================
import os, sys, json, time, math, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import timm
import soundfile as sf
import librosa
from tqdm.auto import tqdm

print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}, devs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} ({p.total_memory/1e9:.1f} GB)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""),

        code_cell("species", """# ============================================================
# Cell 4: BC2026 species list (embedded, 234 species)
# ============================================================
__BC2026_SPECIES = [
__ALL_SPECIES_PLACEHOLDER__
]

species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
SCI_NAME_LC_TO_LABEL = {str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}
LABEL_TO_IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))

N_CLASSES = len(PRIMARY_LABELS)
print(f"BC2026 species: {N_CLASSES}")
print(f"  class breakdown: {species_df['class_name'].value_counts().to_dict()}")
"""),

        code_cell("config", """# ============================================================
# Cell 5: Config
# ============================================================
class CFG:
    SEED = 42
    BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"
    N_CLASSES = N_CLASSES  # 234

    # Audio
    SR = 32000
    CHUNK_SEC = 5
    CHUNK_LEN = SR * CHUNK_SEC

    # Mel
    N_MELS = 128
    N_FFT = 2048
    HOP = 512
    FMIN = 20
    FMAX = 16000

    # Train
    EPOCHS = 30
    BATCH_SIZE = 256  # Blackwell 96GB easy
    LR = 1e-3
    WD = 1e-4
    NUM_WORKERS = 8
    WARMUP_STEPS = 500

    # Aug
    MIXUP_PROB = 0.5
    MIXUP_ALPHA = 0.5
    WAVE_MIXUP_PROB = 0.3
    FREQ_MASK = 30
    TIME_MASK = 40
    LABEL_SMOOTH = 0.05

    # Output
    CKPT_DIR = CKPT_DIR
    BEST_CKPT = CKPT_DIR / "stage1_backbone_best.pth"
    LAST_CKPT = CKPT_DIR / "stage1_backbone_last.pth"
    HIST_JSON = CKPT_DIR / "stage1_history.json"

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)
torch.backends.cudnn.benchmark = True
print(f"Backbone: {CFG.BACKBONE}")
print(f"Train: {CFG.EPOCHS} ep × batch {CFG.BATCH_SIZE}")
print(f"Output: {CFG.CKPT_DIR}")
"""),

        code_cell("data_meta", """# ============================================================
# Cell 6: Build dataset metadata (merge Part 1 + Part 2)
# ============================================================
# Build (audio_path, primary_label, secondary_labels) tuples
records = []

for meta_path, audio_root in [
    (part1_dir / "xc_api" / "_metadata.csv", part1_dir / "xc_api" / "audio"),
    (part2_dir / "xc_api_part2" / "_metadata.csv", part2_dir / "xc_api_part2" / "audio"),
]:
    if not meta_path.exists():
        print(f"SKIP {meta_path} (not found)")
        continue
    df = pd.read_csv(meta_path)
    print(f"\\n[{meta_path.parent.name}] {len(df)} files")

    for _, r in df.iterrows():
        sci = str(r.get("scientific_name", "")).strip()
        label = SCI_NAME_LC_TO_LABEL.get(sci.lower())
        if label is None:
            continue
        filename = r.get("filename", f"{sci.replace(' ', '_')}/XC{r['xc_id']}.mp3")
        audio_path = audio_root / filename
        # Try alt path (in case 'filename' isn't a full relative path)
        if not audio_path.exists():
            audio_path = audio_root / f"{sci.replace(' ', '_')}/XC{r['xc_id']}.mp3"
        if not audio_path.exists():
            continue
        records.append({
            "audio_path": str(audio_path),
            "primary_label": label,
            "scientific_name": sci,
            "duration": r.get("length_sec", 30),
        })

train_df = pd.DataFrame(records)
print(f"\\nTotal pretrain records: {len(train_df)}")
print(f"Unique species: {train_df['primary_label'].nunique()} / {N_CLASSES}")
print(f"Per-species recordings (sample top 5):")
print(train_df['primary_label'].value_counts().head())
"""),

        code_cell("dataset", """# ============================================================
# Cell 7: Dataset class (random 5s crop + multi-label)
# ============================================================
class XCDataset(Dataset):
    def __init__(self, df, sr=CFG.SR, chunk_len=CFG.CHUNK_LEN, training=True,
                 label_smooth=CFG.LABEL_SMOOTH):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.chunk_len = chunk_len
        self.training = training
        self.label_smooth = label_smooth

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        try:
            with sf.SoundFile(audio_path) as f:
                sr = f.samplerate
                src_chunk = int(self.chunk_len * sr / self.sr)
                total = f.frames
                if total > src_chunk:
                    if self.training:
                        start = np.random.randint(0, total - src_chunk + 1)
                    else:
                        start = (total - src_chunk) // 2
                    f.seek(start)
                    wav = f.read(src_chunk, dtype="float32")
                else:
                    wav = f.read(dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        except Exception:
            wav = np.zeros(self.chunk_len, dtype=np.float32)

        if len(wav) >= self.chunk_len:
            wav = wav[:self.chunk_len]
        else:
            wav = np.pad(wav, (0, self.chunk_len - len(wav)))

        # Multi-label with label smoothing
        label = np.full(N_CLASSES, self.label_smooth / 2, dtype=np.float32)
        primary = row["primary_label"]
        if primary in LABEL_TO_IDX:
            label[LABEL_TO_IDX[primary]] = 1.0 - self.label_smooth / 2

        return torch.from_numpy(wav), torch.from_numpy(label)


# Sanity check
ds_check = XCDataset(train_df.head(10), training=True)
wav, label = ds_check[0]
print(f"wav: {wav.shape} dtype={wav.dtype}, range=[{wav.min().item():.3f}, {wav.max().item():.3f}]")
print(f"label: sum={label.sum().item():.2f}, max={label.max().item():.3f}")
"""),

        code_cell("model", """# ============================================================
# Cell 8: Model (SED with framewise attention)
# ============================================================
import torchaudio

class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.SR, n_fft=CFG.N_FFT, hop_length=CFG.HOP,
            n_mels=CFG.N_MELS, f_min=CFG.FMIN, f_max=CFG.FMAX,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def forward(self, wav):
        mel = self.mel(wav)
        mel = self.db(mel)
        mel = torch.clamp(mel, -80.0, 0.0)
        mel = (mel + 40.0) / 40.0
        return mel


class SEDHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.att = nn.Linear(in_dim, n_classes)
        self.cla = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        att = torch.tanh(self.att(x))
        cla = self.cla(x)
        norm_att = F.softmax(att, dim=1)
        clipwise = (norm_att * cla).sum(dim=1)
        return clipwise, cla


class SEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.BACKBONE, pretrained=True, in_chans=3,
            num_classes=0, global_pool="",
        )
        feat_dim = self.backbone.num_features
        self.head = SEDHead(feat_dim, CFG.N_CLASSES)

    def forward(self, mel):
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x)
        feat = feat.mean(dim=2)
        feat = feat.transpose(1, 2)
        clipwise, framewise = self.head(feat)
        return clipwise, framewise


# Sanity check
model_check = SEDModel().to(DEVICE)
dummy = torch.randn(2, CFG.N_MELS, 313).to(DEVICE)
with torch.no_grad():
    out, fr = model_check(dummy)
print(f"clipwise: {out.shape}, framewise: {fr.shape}")
print(f"Model params: {sum(p.numel() for p in model_check.parameters())/1e6:.1f}M")
del model_check; torch.cuda.empty_cache()
"""),

        code_cell("aug", """# ============================================================
# Cell 9: Augmentations
# ============================================================
def spec_mixup(mel, label, alpha=CFG.MIXUP_ALPHA, p=CFG.MIXUP_PROB):
    if np.random.random() > p:
        return mel, label
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(mel.size(0), device=mel.device)
    mel_mix = lam * mel + (1 - lam) * mel[idx]
    label_mix = torch.maximum(label, label[idx])
    return mel_mix, label_mix


def wave_mixup(wav, label, alpha=0.5, p=CFG.WAVE_MIXUP_PROB):
    \"\"\"Waveform domain mixup (paper 256 流).\"\"\"
    if np.random.random() > p:
        return wav, label
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(wav.size(0), device=wav.device)
    wav_mix = lam * wav + (1 - lam) * wav[idx]
    label_mix = torch.maximum(label, label[idx])
    return wav_mix, label_mix


def spec_augment(mel, freq_mask=CFG.FREQ_MASK, time_mask=CFG.TIME_MASK):
    B, F_, T = mel.shape
    for b in range(B):
        if freq_mask > 0:
            f = np.random.randint(0, freq_mask)
            f0 = np.random.randint(0, max(1, F_ - f))
            mel[b, f0:f0+f, :] = 0
        if time_mask > 0:
            t = np.random.randint(0, time_mask)
            t0 = np.random.randint(0, max(1, T - t))
            mel[b, :, t0:t0+t] = 0
    return mel
"""),

        code_cell("train_setup", """# ============================================================
# Cell 10: Train/val split + DataLoaders
# ============================================================
train_df_s = train_df.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
n_val = int(len(train_df_s) * 0.05)
train_split = train_df_s.iloc[n_val:].reset_index(drop=True)
val_split = train_df_s.iloc[:n_val].reset_index(drop=True)
print(f"train: {len(train_split)}, val: {len(val_split)}")

train_ds = XCDataset(train_split, training=True)
val_ds = XCDataset(val_split, training=False)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True,
                          drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                         num_workers=CFG.NUM_WORKERS, pin_memory=True,
                         persistent_workers=True)
print(f"steps/ep: train={len(train_loader)}, val={len(val_loader)}")
"""),

        code_cell("model_setup", """# ============================================================
# Cell 11: Model + Optimizer + Scheduler
# ============================================================
model = SEDModel().to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

mel_extractor = MelExtractor().to(DEVICE)

# AdamW with warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
scaler = GradScaler("cuda", init_scale=2**10)
loss_fn = nn.BCEWithLogitsLoss()
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"LR: {CFG.LR}, WD: {CFG.WD}, Warmup: {CFG.WARMUP_STEPS} steps")
"""),

        code_cell("eval_fn", """# ============================================================
# Cell 12: rich_evaluate (val_ns22 + val_macro + per-taxon AUC + class dist)
# ============================================================
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def rich_evaluate(model, mel_ex, loader):
    model.eval()
    all_logits, all_labels = [], []
    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        mel = mel_ex(wav)
        with autocast("cuda", dtype=torch.float16):
            logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    # Threshold label > 0.5 (label smooth により 0.99-1.0 が positive)
    bin_labels = (labels > 0.5).astype(np.float32)

    per_class = []
    for c in range(bin_labels.shape[1]):
        n_pos = int(bin_labels[:, c].sum())
        if n_pos >= 2 and n_pos < bin_labels.shape[0]:
            try:
                auc = roc_auc_score(bin_labels[:, c], logits[:, c])
                per_class.append((PRIMARY_LABELS[c], float(auc), n_pos))
            except Exception:
                pass

    aucs_n22 = [auc for _, auc, n in per_class if n >= 22]
    val_ns22 = float(np.mean(aucs_n22)) if aucs_n22 else 0.0
    val_macro = float(np.mean([auc for _, auc, _ in per_class])) if per_class else 0.0

    taxon_aucs = {}
    for taxon in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        aucs_in = [auc for label, auc, _ in per_class if LABEL_TO_CLASS.get(label) == taxon]
        taxon_aucs[taxon] = float(np.mean(aucs_in)) if aucs_in else float("nan")

    if aucs_n22:
        arr = np.array(aucs_n22)
        cs = {"n_valid": len(arr), "median": float(np.median(arr)),
              "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
              "n_above_05": int(np.sum(arr > 0.5)), "n_above_07": int(np.sum(arr > 0.7)),
              "n_above_09": int(np.sum(arr > 0.9)), "n_perfect": int(np.sum(arr >= 0.9999))}
    else:
        cs = {"n_valid": 0, "median": 0.0, "p25": 0.0, "p75": 0.0,
              "n_above_05": 0, "n_above_07": 0, "n_above_09": 0, "n_perfect": 0}

    return val_ns22, val_macro, taxon_aucs, cs
"""),

        code_cell("train_loop", """# ============================================================
# Cell 13: Training loop (30 ep, safe AMP, rich log)
# ============================================================
import sys
def _p(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# Smoke test
_p(f"[smoke] iter + forward + backward...")
smoke_iter = iter(train_loader)
test_wav, test_label = next(smoke_iter)
test_wav_g = test_wav.to(DEVICE)
test_label_g = test_label.to(DEVICE)
test_mel = mel_extractor(test_wav_g)
_p(f"[smoke] mel: {test_mel.shape}, range=[{test_mel.min().item():.2f}, {test_mel.max().item():.2f}]")
with autocast("cuda", dtype=torch.float16):
    test_logit, _ = model(test_mel)
    test_loss = loss_fn(test_logit, test_label_g)
_p(f"[smoke] loss={test_loss.item():.4f}")
if not (torch.isnan(test_loss) or torch.isinf(test_loss)):
    scaler.scale(test_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    _p(f"[smoke] OK")
del smoke_iter, test_wav, test_label, test_wav_g, test_label_g, test_mel, test_logit, test_loss
torch.cuda.empty_cache()
_p(f"\\n=== Training start ===\\n")

history = {"train_loss": [], "val_ns22": [], "val_macro": [],
           "taxon_aucs": [], "class_stats": [], "lr": [], "elapsed_min": []}
best_val = 0.0
start_t = time.time()
nan_skip_count = 0
global_step = 0

for epoch in range(1, CFG.EPOCHS + 1):
    ep_start = time.time()
    _p(f"[ep{epoch}] start")
    model.train()
    train_losses = []
    pbar = tqdm(train_loader, desc=f"Ep {epoch}/{CFG.EPOCHS}", leave=False, file=sys.stdout)
    for batch_idx, (wav, label) in enumerate(pbar):
        # LR warmup
        if global_step < CFG.WARMUP_STEPS:
            lr_scale = (global_step + 1) / CFG.WARMUP_STEPS
            for g in optimizer.param_groups:
                g["lr"] = CFG.LR * lr_scale

        wav = wav.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        # Waveform mixup (before mel)
        wav, label = wave_mixup(wav, label)

        mel = mel_extractor(wav)
        # Spec mixup
        mel, label = spec_mixup(mel, label)
        mel = spec_augment(mel)

        with autocast("cuda", dtype=torch.float16):
            logit, _ = model(mel)
            loss = loss_fn(logit, label)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        global_step += 1

        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
            torch.cuda.empty_cache()
        del mel, logit, loss

    if global_step >= CFG.WARMUP_STEPS:
        scheduler.step()
    torch.cuda.empty_cache()

    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    _p(f"[ep{epoch}] train done (loss={train_loss:.4f}), val...")
    val_ns22, val_macro, taxon_aucs, class_stats = rich_evaluate(model, mel_extractor, val_loader)
    lr_now = optimizer.param_groups[0]["lr"]
    elapsed_min = (time.time() - start_t) / 60
    ep_min = (time.time() - ep_start) / 60

    history["train_loss"].append(train_loss)
    history["val_ns22"].append(val_ns22)
    history["val_macro"].append(val_macro)
    history["taxon_aucs"].append(taxon_aucs)
    history["class_stats"].append(class_stats)
    history["lr"].append(lr_now)
    history["elapsed_min"].append(elapsed_min)

    is_best = val_ns22 > best_val
    if is_best:
        best_val = val_ns22
        torch.save(
            {"epoch": epoch, "val_ns22": val_ns22, "val_macro": val_macro,
             "taxon_aucs": taxon_aucs, "class_stats": class_stats,
             "backbone": model.module.backbone.state_dict() if isinstance(model, nn.DataParallel) else model.backbone.state_dict(),
             "head": model.module.head.state_dict() if isinstance(model, nn.DataParallel) else model.head.state_dict(),
             "config": {k: str(v) for k, v in CFG.__dict__.items() if not k.startswith("_")},
            },
            CFG.BEST_CKPT,
        )

    json.dump(history, open(CFG.HIST_JSON, "w"), indent=2)
    taxon_str = " ".join(f"{t}={v:.3f}" for t, v in taxon_aucs.items())
    cs = class_stats
    _p(f"=== Ep {epoch}/{CFG.EPOCHS}: loss={train_loss:.4f} val_ns22={val_ns22:.4f} val_macro={val_macro:.4f}"
       f" {'BEST' if is_best else ''} lr={lr_now:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min, nan_skips={nan_skip_count}) ===")
    _p(f"    taxon: {taxon_str}")
    _p(f"    class: n={cs['n_valid']} median={cs['median']:.3f} p25={cs['p25']:.3f} p75={cs['p75']:.3f}"
       f" #>0.5={cs['n_above_05']} #>0.7={cs['n_above_07']} #>0.9={cs['n_above_09']} #perfect={cs['n_perfect']}")

# Save final
torch.save(
    {"epoch": CFG.EPOCHS, "val_ns22": history["val_ns22"][-1],
     "backbone": model.module.backbone.state_dict() if isinstance(model, nn.DataParallel) else model.backbone.state_dict(),
     "head": model.module.head.state_dict() if isinstance(model, nn.DataParallel) else model.head.state_dict(),
    },
    CFG.LAST_CKPT,
)
_p(f"\\n=== Stage 1 DONE. Best val_ns22={best_val:.4f} ===")
_p(f"Saved: {CFG.BEST_CKPT}")
_p(f"Saved: {CFG.LAST_CKPT}")
"""),

        code_cell("summary", """# ============================================================
# Cell 14: Summary + transfer prep
# ============================================================
print(f"=== Stage 1 Summary ===")
print(f"  Backbone: {CFG.BACKBONE}")
print(f"  Epochs trained: {CFG.EPOCHS}")
print(f"  Best val_ns22: {best_val:.4f}")
print(f"  Total time: {(time.time() - start_t)/60:.1f} min")
print(f"  NaN skips: {nan_skip_count}")
print()
print(f"Best ckpt path: {CFG.BEST_CKPT}")
print(f"  size: {CFG.BEST_CKPT.stat().st_size/1e6:.1f} MB")
print()
print(f"Next steps (Stage 2):")
print(f"  1. Load backbone from {CFG.BEST_CKPT}['backbone']")
print(f"  2. Re-init head for BC2026 finetune (head_reinit=True)")
print(f"  3. Train on BC2026 train_audio for 20-30 epoch")
print(f"  4. Expected LB +0.008-0.025")
print()
# Disconnect runtime to save credits
print("Disconnecting runtime to save Colab credits...")
from google.colab import runtime
# runtime.unassign()  # uncomment to auto-disconnect
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Inject species list
for c in nb["cells"]:
    if c.get("id") == "species":
        src = "".join(c["source"])
        src = src.replace("__ALL_SPECIES_PLACEHOLDER__", all_species_block)
        c["source"] = src.splitlines(keepends=True)
        break

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id'):15s} L={n}")
