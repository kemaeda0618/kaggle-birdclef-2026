"""Generate Colab NB for Stage 2A: BC2026 finetune from Stage 1 backbone.

Input:
  - Stage 1 backbone: /content/drive/MyDrive/kaggle/birdclef2026/exp052/ckpt/stage1_backbone_best.pth
  - BC2026 train_audio (downloaded from Kaggle Competition)

Output:
  - Stage 2A finetuned model
  - Kaggle Dataset: maekeso/birdclef2026-exp053-stage2a-effv2s
"""
import json
from pathlib import Path

OUT_DIR = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp053\notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "nb_stage2a_bc2026_finetune.ipynb"

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

# Read 234 species list
all_species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species.txt").read_text(encoding="utf-8")
all_species_lines = []
for line in all_species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        all_species_lines.append("    " + s)
all_species_block = "\n".join(all_species_lines)

nb = {
    "cells": [
        md_cell("hdr", """# exp053 Stage 2A: BC2026 Finetune from Stage 1 Backbone (Colab Blackwell)

**Pipeline**: Stage 1 (XC pretrain) backbone → Stage 2A (BC2026 finetune) → 新 stream blend

## Input

- Stage 1 backbone: `/content/drive/MyDrive/kaggle/birdclef2026/exp052/ckpt/stage1_backbone_best.pth`
  - val_macro 0.9390 (Aves AUC 0.939 over 159 species)
- BC2026 train_audio (Kaggle Competition download)

## Design (Path A: Standalone)

| Item | Value | Reason |
|---|---|---|
| Backbone | Stage 1 ckpt load (XC pretrained EffNetV2-S) | 鳥音響 general 学習済 |
| Head | **Reinit** (random init) | BC2026 234 distribution に合わせ scratch から |
| Data | BC2026 train_audio (35k, 234 species) | 全 taxa 学習 |
| Loss | BCEWithLogitsLoss + label smooth | 多ラベル + soft target |
| Optimizer | AdamW、**differential LR** | backbone=5e-5、head=5e-4 |
| Scheduler | CosineAnnealingLR T_max=20 | 標準 |
| Epochs | **20** | 短い、backbone 既学習済 |
| Batch | 256 (Blackwell) | |
| Aug | Spec mixup + SpecAugment + wave mixup | Stage 1 同 |
| Label smooth | 0.05 | |

## 期待

- val_ns22: **0.92-0.94**
- standalone LB: **0.92-0.94** (Path A standalone target)
- 4-way blend (exp048 + Stage 2A) LB: **0.951-0.955** (gold border 接近)
- Blackwell 訓練時間: **30-45 min** (Stage 1 が想定の 1/10 で終了したので Stage 2A も短い)

## Output

- Drive: `/content/drive/MyDrive/kaggle/birdclef2026/exp053/ckpt/stage2a_best.pth`
- Kaggle Dataset: `maekeso/birdclef2026-exp053-stage2a-effv2s`
"""),

        code_cell("install", """# ============================================================
# Cell 1: Install + Drive mount + Kaggle API setup
# ============================================================
!pip install -q timm kaggle soundfile librosa torchaudio tqdm

from google.colab import drive
drive.mount("/content/drive")

import os, json, shutil
from pathlib import Path

# Drive paths
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/exp053")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = DRIVE_ROOT / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)

# Stage 1 backbone path
STAGE1_CKPT = Path("/content/drive/MyDrive/kaggle/birdclef2026/exp052/ckpt/stage1_backbone_best.pth")
assert STAGE1_CKPT.exists(), f"Stage 1 ckpt not found: {STAGE1_CKPT}"
print(f"Stage 1 ckpt found: {STAGE1_CKPT} ({STAGE1_CKPT.stat().st_size/1e6:.1f} MB)")

# Kaggle credentials - try multiple locations
KAGGLE_JSON_CANDIDATES = [
    Path("/content/drive/MyDrive/kaggle/birdclef2026/kaggle.json"),
    Path("/content/drive/MyDrive/kaggle.json"),
    Path("/content/drive/MyDrive/.kaggle/kaggle.json"),
    Path("/content/kaggle.json"),
    Path.home() / ".kaggle" / "kaggle.json",
]
KAGGLE_JSON_PATH = next((p for p in KAGGLE_JSON_CANDIDATES if p.exists()), None)
assert KAGGLE_JSON_PATH is not None, "kaggle.json not found"
print(f"Using kaggle.json from: {KAGGLE_JSON_PATH}")

# Set KAGGLE_API_TOKEN env var (CLAUDE.md memory)
creds = json.loads(KAGGLE_JSON_PATH.read_text())
os.environ["KAGGLE_API_TOKEN"] = creds["key"]
os.makedirs(Path.home() / ".kaggle", exist_ok=True)
shutil.copy(KAGGLE_JSON_PATH, Path.home() / ".kaggle" / "kaggle.json")
os.chmod(Path.home() / ".kaggle" / "kaggle.json", 0o600)
print(f"OK KAGGLE_API_TOKEN env var set")
"""),

        code_cell("dl_bc2026", """# ============================================================
# Cell 2: Download BC2026 competition data (train_audio + meta)
# ============================================================
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print("OK Kaggle SDK authenticated")

BC_ROOT = Path("/content/bc2026")
BC_ROOT.mkdir(exist_ok=True)

# Download train.csv, taxonomy.csv, train_soundscapes_labels.csv (small files)
for fname in ["train.csv", "taxonomy.csv", "train_soundscapes_labels.csv"]:
    target = BC_ROOT / fname
    if target.exists():
        print(f"  Already exists: {target}")
        continue
    try:
        api.competition_download_file("birdclef-2026", fname, path=str(BC_ROOT))
        # Some downloads come as .zip
        zip_path = BC_ROOT / f"{fname}.zip"
        if zip_path.exists():
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(BC_ROOT)
            zip_path.unlink()
        print(f"  Downloaded: {target}")
    except Exception as e:
        print(f"  ERR {fname}: {str(e)[:200]}")

# Download train_audio (large, ~15 GB)
TRAIN_AUDIO = BC_ROOT / "train_audio"
if not TRAIN_AUDIO.exists() or len(list(TRAIN_AUDIO.iterdir())) < 100:
    print(f"\\nDownloading train_audio (~15 GB, this takes 10-20 min)...")
    api.competition_download_file("birdclef-2026", "train_audio.zip", path=str(BC_ROOT))
    import zipfile
    zip_files = list(BC_ROOT.glob("train_audio*.zip"))
    if zip_files:
        with zipfile.ZipFile(zip_files[0]) as zf:
            zf.extractall(BC_ROOT)
        zip_files[0].unlink()
    print(f"  OK train_audio downloaded")
else:
    print(f"  train_audio already exists")

# Verify
import pandas as pd
train_df = pd.read_csv(BC_ROOT / "train.csv")
print(f"\\ntrain.csv: {len(train_df)} rows")
print(f"  columns: {train_df.columns.tolist()[:10]}")
n_audio = len(list(TRAIN_AUDIO.rglob("*.ogg")))
print(f"  train_audio: {n_audio} .ogg files")
"""),

        code_cell("setup", """# ============================================================
# Cell 3: Setup + GPU check
# ============================================================
import os, sys, json, time, math, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import timm
import soundfile as sf
import librosa
from tqdm.auto import tqdm

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} ({p.total_memory/1e9:.1f} GB)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""),

        code_cell("species", """# ============================================================
# Cell 4: BC2026 species list (234 species)
# ============================================================
__BC2026_SPECIES = [
__ALL_SPECIES_PLACEHOLDER__
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
LABEL_TO_IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
N_CLASSES = len(PRIMARY_LABELS)
print(f"BC2026 species: {N_CLASSES}")
print(f"  taxa: {species_df['class_name'].value_counts().to_dict()}")
"""),

        code_cell("config", """# ============================================================
# Cell 5: Config
# ============================================================
class CFG:
    SEED = 42
    BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"
    N_CLASSES = N_CLASSES

    SR = 32000
    CHUNK_SEC = 5
    CHUNK_LEN = SR * CHUNK_SEC

    N_MELS = 128
    N_FFT = 2048
    HOP = 512
    FMIN = 20
    FMAX = 16000

    # Train
    EPOCHS = 20
    BATCH_SIZE = 256
    LR_BACKBONE = 5e-5   # ★ small (Stage 1 既学習済を保持)
    LR_HEAD = 5e-4       # ★ large (head は scratch)
    WD = 1e-4
    NUM_WORKERS = 4

    MIXUP_PROB = 0.5
    MIXUP_ALPHA = 1.0
    WAVE_MIXUP_PROB = 0.3
    FREQ_MASK = 30
    TIME_MASK = 40
    LABEL_SMOOTH = 0.05

    BC_ROOT = Path("/content/bc2026")
    TRAIN_AUDIO = BC_ROOT / "train_audio"
    TRAIN_CSV = BC_ROOT / "train.csv"

    CKPT_DIR = CKPT_DIR
    BEST_CKPT = CKPT_DIR / "stage2a_best.pth"
    LAST_CKPT = CKPT_DIR / "stage2a_last.pth"
    HIST_JSON = CKPT_DIR / "stage2a_history.json"

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)
torch.backends.cudnn.benchmark = True
print(f"Backbone: {CFG.BACKBONE}")
print(f"Train: {CFG.EPOCHS} ep × batch {CFG.BATCH_SIZE}")
print(f"LR: backbone={CFG.LR_BACKBONE} head={CFG.LR_HEAD}")
print(f"Output: {CFG.CKPT_DIR}")
"""),

        code_cell("data_meta", """# ============================================================
# Cell 6: Build BC2026 train DataFrame
# ============================================================
train_df = pd.read_csv(CFG.TRAIN_CSV)
print(f"BC2026 train: {len(train_df)} rows")
print(f"  columns: {train_df.columns.tolist()[:10]}")

# Build audio path
def _resolve_audio_path(row):
    fname = row["filename"]
    if not str(fname).endswith(".ogg"):
        fname = fname + ".ogg"
    return CFG.TRAIN_AUDIO / fname
train_df["audio_path"] = train_df.apply(_resolve_audio_path, axis=1)
n_exists = train_df["audio_path"].apply(lambda p: p.exists()).sum()
print(f"  audio exists: {n_exists}/{len(train_df)}")

# Sample distribution
print(f"  taxa breakdown:")
print(train_df['class_name'].value_counts())
"""),

        code_cell("dataset", """# ============================================================
# Cell 7: BC2026 Dataset (with secondary labels)
# ============================================================
class BC2026Dataset(Dataset):
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
            with sf.SoundFile(str(audio_path)) as f:
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

        # Multi-label with label smoothing (primary + secondary)
        label = np.full(N_CLASSES, self.label_smooth / 2, dtype=np.float32)
        primary = row["primary_label"]
        if primary in LABEL_TO_IDX:
            label[LABEL_TO_IDX[primary]] = 1.0 - self.label_smooth / 2

        # secondary_labels (BC2026 has this column)
        sec = row.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec not in ("[]", "", "nan"):
            try:
                sec_list = eval(sec) if sec.startswith("[") else []
                for s in sec_list:
                    if s in LABEL_TO_IDX:
                        label[LABEL_TO_IDX[s]] = 1.0 - self.label_smooth / 2
            except Exception:
                pass

        return torch.from_numpy(wav), torch.from_numpy(label)


ds_check = BC2026Dataset(train_df.head(10), training=True)
wav, label = ds_check[0]
print(f"wav: {wav.shape}, range=[{wav.min().item():.3f}, {wav.max().item():.3f}]")
print(f"label: sum={label.sum().item():.2f}")
"""),

        code_cell("model", """# ============================================================
# Cell 8: Model + LOAD Stage 1 backbone
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
            CFG.BACKBONE, pretrained=False, in_chans=3,  # ★ no pretrained (Stage 1 で上書き)
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


# Build model + load Stage 1 backbone
model = SEDModel().to(DEVICE)
print(f"Model created, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ★ Load Stage 1 backbone
stage1_ckpt = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
backbone_state = stage1_ckpt.get("backbone", stage1_ckpt.get("state_dict"))
if backbone_state is None:
    raise ValueError(f"Stage 1 ckpt missing backbone key, has: {list(stage1_ckpt.keys())}")
missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
print(f"\\nStage 1 backbone loaded:")
print(f"  missing keys: {len(missing)} (head 等は無視 OK)")
print(f"  unexpected keys: {len(unexpected)}")
print(f"  Stage 1 val_ns22: {stage1_ckpt.get('val_ns22', 'n/a')}")
print(f"  Stage 1 epoch: {stage1_ckpt.get('epoch', 'n/a')}")
print(f"\\n=> Head は scratch から学習開始 (random init)")

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
mel_extractor = MelExtractor().to(DEVICE)
"""),

        code_cell("aug", """# ============================================================
# Cell 9: Augmentations (same as Stage 1)
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
# Cell 10: Train/val split + dataloaders
# ============================================================
train_df_s = train_df[train_df["audio_path"].apply(lambda p: p.exists())].reset_index(drop=True)
print(f"Valid audio: {len(train_df_s)}/{len(train_df)}")

train_df_s = train_df_s.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
n_val = int(len(train_df_s) * 0.05)
train_split = train_df_s.iloc[n_val:].reset_index(drop=True)
val_split = train_df_s.iloc[:n_val].reset_index(drop=True)
print(f"train: {len(train_split)}, val: {len(val_split)}")

train_ds = BC2026Dataset(train_split, training=True)
val_ds = BC2026Dataset(val_split, training=False)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True,
                          drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                         num_workers=CFG.NUM_WORKERS, pin_memory=True,
                         persistent_workers=True)
print(f"steps/ep: train={len(train_loader)}, val={len(val_loader)}")
"""),

        code_cell("model_setup", """# ============================================================
# Cell 11: Optimizer with differential LR (backbone slow, head fast)
# ============================================================
# Get backbone vs head params
if isinstance(model, nn.DataParallel):
    bk_params = model.module.backbone.parameters()
    hd_params = model.module.head.parameters()
else:
    bk_params = model.backbone.parameters()
    hd_params = model.head.parameters()

optimizer = torch.optim.AdamW([
    {"params": bk_params, "lr": CFG.LR_BACKBONE, "name": "backbone"},
    {"params": hd_params, "lr": CFG.LR_HEAD, "name": "head"},
], weight_decay=CFG.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
loss_fn = nn.BCEWithLogitsLoss()
print(f"Optimizer: AdamW, LR backbone={CFG.LR_BACKBONE} head={CFG.LR_HEAD}")
"""),

        code_cell("eval_fn", """# ============================================================
# Cell 12: rich_evaluate
# ============================================================
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def rich_evaluate(model, mel_ex, loader):
    model.eval()
    all_logits, all_labels = [], []
    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        mel = mel_ex(wav)
        with autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
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
# Cell 13: Training loop (20 ep BC2026 finetune)
# ============================================================
import sys
def _p(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# Smoke test
_p(f"[smoke] iter + forward + backward (bf16)...")
smoke_iter = iter(train_loader)
test_wav, test_label = next(smoke_iter)
test_wav_g = test_wav.to(DEVICE)
test_label_g = test_label.to(DEVICE)
test_mel = mel_extractor(test_wav_g)
_p(f"[smoke] mel: {test_mel.shape}")
with autocast("cuda", dtype=torch.bfloat16):
    test_logit, _ = model(test_mel)
    test_loss = loss_fn(test_logit, test_label_g)
_p(f"[smoke] loss={test_loss.item():.4f}")
test_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
_p(f"[smoke] OK\\n")
del smoke_iter, test_wav, test_label, test_wav_g, test_label_g, test_mel, test_logit, test_loss
torch.cuda.empty_cache()

history = {"train_loss": [], "val_ns22": [], "val_macro": [],
           "taxon_aucs": [], "class_stats": [], "lr_backbone": [], "lr_head": [], "elapsed_min": []}
best_val = 0.0
start_t = time.time()
nan_skip_count = 0

for epoch in range(1, CFG.EPOCHS + 1):
    ep_start = time.time()
    model.train()
    train_losses = []
    pbar = tqdm(train_loader, desc=f"Ep {epoch}/{CFG.EPOCHS}", leave=False, file=sys.stdout)
    for batch_idx, (wav, label) in enumerate(pbar):
        wav = wav.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        wav, label = wave_mixup(wav, label)
        mel = mel_extractor(wav)
        mel, label = spec_mixup(mel, label)
        mel = spec_augment(mel)

        with autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(mel)
            loss = loss_fn(logit, label)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())

        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f}")
        del mel, logit, loss

    scheduler.step()
    torch.cuda.empty_cache()
    train_loss = float(np.mean(train_losses))

    val_ns22, val_macro, taxon_aucs, class_stats = rich_evaluate(model, mel_extractor, val_loader)
    lr_bk = optimizer.param_groups[0]["lr"]
    lr_hd = optimizer.param_groups[1]["lr"]
    elapsed_min = (time.time() - start_t) / 60
    ep_min = (time.time() - ep_start) / 60

    history["train_loss"].append(train_loss)
    history["val_ns22"].append(val_ns22)
    history["val_macro"].append(val_macro)
    history["taxon_aucs"].append(taxon_aucs)
    history["class_stats"].append(class_stats)
    history["lr_backbone"].append(lr_bk)
    history["lr_head"].append(lr_hd)
    history["elapsed_min"].append(elapsed_min)

    is_best = val_ns22 > best_val
    if is_best:
        best_val = val_ns22
        torch.save({
            "epoch": epoch, "val_ns22": val_ns22, "val_macro": val_macro,
            "taxon_aucs": taxon_aucs, "class_stats": class_stats,
            "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "config": {k: str(v) for k, v in CFG.__dict__.items() if not k.startswith("_")},
            "stage1_val_ns22": stage1_ckpt.get("val_ns22", "n/a"),
        }, CFG.BEST_CKPT)

    json.dump(history, open(CFG.HIST_JSON, "w"), indent=2)
    taxon_str = " ".join(f"{t}={v:.3f}" for t, v in taxon_aucs.items())
    cs = class_stats
    _p(f"=== Ep {epoch}/{CFG.EPOCHS}: loss={train_loss:.4f} val_ns22={val_ns22:.4f} val_macro={val_macro:.4f}"
       f" {'BEST' if is_best else ''} lr_bk={lr_bk:.2e} lr_hd={lr_hd:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min) ===")
    _p(f"    taxon: {taxon_str}")
    _p(f"    class: n={cs['n_valid']} median={cs['median']:.3f} p25={cs['p25']:.3f} p75={cs['p75']:.3f}"
       f" #>0.5={cs['n_above_05']} #>0.7={cs['n_above_07']} #>0.9={cs['n_above_09']} #perfect={cs['n_perfect']}")

torch.save({
    "epoch": CFG.EPOCHS, "val_ns22": history["val_ns22"][-1],
    "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
}, CFG.LAST_CKPT)
_p(f"\\n=== Stage 2A DONE. Best val_ns22={best_val:.4f} ===")
"""),

        code_cell("upload_kaggle", """# ============================================================
# Cell 14: Upload Stage 2A ckpt to Kaggle Dataset
# ============================================================
import json, shutil
UPLOAD_DIR = Path("/content/exp053_upload")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

for src_file in [CFG.BEST_CKPT, CFG.LAST_CKPT, CFG.HIST_JSON]:
    if src_file.exists():
        shutil.copy(src_file, UPLOAD_DIR / src_file.name)
        print(f"  Copied: {src_file.name} ({src_file.stat().st_size/1e6:.1f} MB)")

USER = "maekeso"
SLUG = "birdclef2026-exp053-stage2a-effv2s"
meta = {
    "title": "BirdCLEF2026 exp053 Stage 2A EfficientNetV2-S",
    "id": f"{USER}/{SLUG}",
    "licenses": [{"name": "other"}],
}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

import subprocess
def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ})
    print(r.stdout); print(r.stderr[:300] if r.stderr else "")
    return r.returncode

ret = run(f"kaggle datasets version -p {UPLOAD_DIR} -m 'Stage 2A best={best_val:.4f}' -r tar")
if ret != 0:
    print("Version up failed, creating new...")
    ret = run(f"kaggle datasets create -p {UPLOAD_DIR} -r tar")
if ret == 0:
    print(f"\\nOK Dataset uploaded: https://www.kaggle.com/datasets/{USER}/{SLUG}")
else:
    print(f"\\nUpload failed. Manual upload needed.")
"""),

        code_cell("summary", """# ============================================================
# Cell 15: Summary
# ============================================================
print(f"=== Stage 2A Summary ===")
print(f"  Stage 1 backbone: {STAGE1_CKPT}")
print(f"  Stage 1 val_ns22: {stage1_ckpt.get('val_ns22', 'n/a')}")
print(f"  Stage 2A epochs: {CFG.EPOCHS}")
print(f"  Best val_ns22: {best_val:.4f}")
print(f"  Total time: {(time.time() - start_t)/60:.1f} min")
print()
print(f"Best ckpt: {CFG.BEST_CKPT} ({CFG.BEST_CKPT.stat().st_size/1e6:.1f} MB)")
print()
print(f"Next: Inference NB → 4-way blend (exp048 + Stage 2A)")
print(f"  Expected blend LB: 0.951-0.955 (gold border 接近)")
"""),

        code_cell("disconnect", """# ============================================================
# Cell 16: Disconnect runtime
# ============================================================
print("Disconnecting in 30 sec...")
import time
time.sleep(30)
from google.colab import runtime
runtime.unassign()
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
