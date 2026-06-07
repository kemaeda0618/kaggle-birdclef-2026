"""Generate Stage 1 v2 NB for Colab Blackwell.

Inputs (6 Kaggle Datasets):
  - maekeso/birdclef2026-xc-api-aves-audio-part1 (76 Aves)
  - maekeso/birdclef2026-xc-api-aves-audio-part2 (83 Aves)
  - maekeso/birdclef2026-xc-api-aves-audio-part3 (27 non-Aves)
  - maekeso/birdclef2026-exp047-bc-extracted (54 species)
  - maekeso/birdclef2026-exp047-inat-extracted (183 species)
  - maekeso/birdclef2026-exp047-anuraset-extracted (17 Amphibia)

Output: tf_efficientnetv2_s backbone weights
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp054\notebook\nb_stage1_v2.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species_with_inat.txt").read_text(encoding="utf-8")
species_lines = []
for line in species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        species_lines.append("    " + s)
species_block = "\n".join(species_lines)

nb = {
    "cells": [
        md_cell("hdr", """# exp054 Stage 1 v2 — Pretrain (Colab Blackwell)

**Purpose**: Multi-source pretrain (XC + BC + iNat + AnuraSet) for backbone robustness

**Inputs** (6 Kaggle Datasets):
- XC Part 1 (76 Aves)
- XC Part 2 (83 Aves)
- XC Part 3 (27 non-Aves)
- BC extracted (54 species, 過去 BC 2021-2025)
- iNat extracted (183 species)
- AnuraSet extracted (17 Amphibia)

**Coverage**: ~210-220 / 234 unique BC2026 species

**Model**: `tf_efficientnetv2_s.in21k_ft_in1k` (paper 256 採用)
**Train**: 25 epochs, batch 192, lr 1.2e-3, bf16 AMP, MIXUP α=1.5, drop_path 0.2
**Expected**: 30-60 min on Colab Blackwell (RTX PRO 6000)

**Output**: backbone state_dict → Kaggle Dataset `maekeso/birdclef2026-exp054-stage1-effv2s`

vs Stage 1 (Phase 1):
- Phase 1: XC only (~26k files, 159 Aves species) → val_ns22 0.9423
- Phase 2: 6 sources (~46k files, 210+ species) → expected val_ns22 0.94+ + better non-Aves
"""),

        code_cell("install", """# Cell 1: install deps + drive mount
!pip install -q timm==1.0.11 soundfile librosa kaggle 2>&1 | tail -1

from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

# Drive root (per memory: kaggle/birdclef2026/)
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp054")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Drive: {DRIVE_ROOT}")
"""),

        code_cell("dl_data", """# Cell 2: Download 6 Kaggle Datasets to local (SDK + KAGGLE_API_TOKEN env var)
import os, json, zipfile, time
from pathlib import Path

# Setup kaggle.json (used as source of KGAT_ token)
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_DIR.mkdir(exist_ok=True)
if not (KAGGLE_DIR / "kaggle.json").exists():
    src_kg = Path("/content/drive/MyDrive/kaggle/kaggle.json")
    if src_kg.exists():
        import shutil; shutil.copy(src_kg, KAGGLE_DIR / "kaggle.json")
        os.chmod(KAGGLE_DIR / "kaggle.json", 0o600)
        print("  ✓ Copied kaggle.json from Drive")
    else:
        raise SystemExit("Need kaggle.json at /content/drive/MyDrive/kaggle/kaggle.json")

# CRITICAL: set KAGGLE_API_TOKEN env var (KGAT_ Bearer token)
# CLI uses Basic auth (username/key) which fails with Bearer tokens.
# SDK + KAGGLE_API_TOKEN env var = Bearer auth (working).
_kgat = json.loads((KAGGLE_DIR / "kaggle.json").read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat
print(f"  KAGGLE_API_TOKEN set (len={len(_kgat)}, starts with {_kgat[:5]}...)")

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

DATASETS = [
    "maekeso/birdclef2026-xc-api-dl-part1",
    "maekeso/birdclef2026-xc-api-dl-part2",
    "maekeso/birdclef2026-xc-api-dl-part3",
    "maekeso/birdclef2026-exp047-bc-extracted",
    "maekeso/birdclef2026-exp047-inat-extracted",
    "maekeso/birdclef2026-exp047-anuraset-extracted",
]

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)

t0 = time.time()
for ds in DATASETS:
    name = ds.split("/")[-1]
    dst = LOCAL_DATA / name
    if dst.exists() and any(dst.iterdir()):
        n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        if n_files > 0:
            print(f"  ✓ {name} exists ({n_files} files), skip")
            continue
    dst.mkdir(exist_ok=True)
    print(f"  DL {ds}...")
    try:
        api.dataset_download_files(ds, path=str(dst), unzip=True, quiet=False)
        n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        gb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e9
        print(f"    ✓ {n_files} files, {gb:.2f} GB")
    except Exception as e:
        print(f"    ✗ err: {str(e)[:200]}")

print(f"\\nDL total: {(time.time()-t0)/60:.1f} min")

# Final verify
print(f"\\n=== Final verify ===")
total_files = 0; total_gb = 0
for ds in DATASETS:
    name = ds.split("/")[-1]
    dst = LOCAL_DATA / name
    if dst.exists():
        n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        gb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e9
        total_files += n_files; total_gb += gb
        print(f"  {name}: {n_files} files, {gb:.2f} GB")
    else:
        print(f"  ✗ {name}: missing")
print(f"\\nTOTAL: {total_files} files, {total_gb:.2f} GB")
# Soft assertion: warn if any source missing, but continue
missing = [ds.split("/")[-1] for ds in DATASETS
           if not (LOCAL_DATA / ds.split("/")[-1]).exists()
           or sum(1 for _ in (LOCAL_DATA / ds.split("/")[-1]).rglob("*") if _.is_file()) == 0]
if missing:
    print(f"\\n⚠ MISSING sources: {missing}")
    print(f"  Continuing with available data only ({total_files} files)")
assert total_files > 5000, f"Too few files ({total_files}) - check Kaggle auth"
"""),

        code_cell("setup", """# Cell 3: imports + GPU info
import os, sys, json, time, math, random, gc, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Use new torch.amp API (torch.cuda.amp is deprecated in PyTorch 2.x)
from torch.amp import autocast, GradScaler
import torchaudio
import soundfile as sf
import librosa
import timm
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")
"""),

        code_cell("species", f"""# Cell 4: BC2026 species list (234)
__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {{l: i for i, l in enumerate(PRIMARY_LABELS)}}
SCI_LC_TO_LABEL = {{str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
print(f"BC2026: {{N_CLASSES}} species")
print(f"  by class: {{species_df['class_name'].value_counts().to_dict()}}")
"""),

        code_cell("config", """# Cell 5: training config (paper 256 spec, adjusted for Blackwell)
CFG = dict(
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
    sr=32000, chunk_sec=5, n_mels=128, n_fft=2048, hop_length=512,
    fmin=20, fmax=16000,
    epochs=25,
    batch_size=192,
    lr=1.2e-3,                # sqrt(192/32) * 5e-4 ≈ 1.2e-3
    weight_decay=1e-3,
    warmup_steps=800,         # larger batch + higher lr → longer warmup
    label_smoothing=0.1,
    mixup_alpha=1.5,          # ↑ from 1.0 for stronger regularization
    grad_clip=2.0,
    num_workers=8,            # ↑ from 4 for batch 192 I/O
    val_split=0.1,
    val_seed=42,
)
for k, v in CFG.items(): print(f"  {k}: {v}")
CHUNK_LEN = CFG["sr"] * CFG["chunk_sec"]
"""),

        code_cell("data_meta", """# Cell 6: build unified meta_df from all 6 sources
LOCAL_DATA = Path("/content/data")

all_records = []
total_skipped = 0

# Helper: extract primary_label from path or metadata
def from_xc_dataset(src_root, source_name):
    \"\"\"XC: structure is xc_api[_partN]/<species_code>/<files>.mp3
       species_code can be in BC2026 or not (filter to BC2026)\"\"\"
    found = []
    # XC has .mp3 + .ogg + sometimes .wav, check all common audio extensions
    audio_files = []
    for ext in ["*.mp3", "*.ogg", "*.wav", "*.flac", "*.MP3", "*.OGG"]:
        audio_files.extend(list(src_root.rglob(ext)))
    for p in audio_files:
        sp_dir = p.parent.name
        if sp_dir in label_to_idx:
            found.append((str(p), sp_dir, source_name))
        else:
            sp_dir2 = p.parent.parent.name if p.parent.parent != src_root else None
            if sp_dir2 in label_to_idx:
                found.append((str(p), sp_dir2, source_name))
    return found

def from_extracted(src_root, source_name):
    \"\"\"BC/iNat/AnuraSet extracted: metadata.csv has primary_label + filename\"\"\"
    meta_csv = next(src_root.rglob("metadata.csv"), None)
    if not meta_csv:
        return []
    df = pd.read_csv(meta_csv)
    found = []
    # filename in CSV is relative to extracted dir (e.g., "audio/bc2021/banana/XC112602.ogg")
    base = meta_csv.parent
    for _, r in df.iterrows():
        pl = str(r["primary_label"])
        if pl not in label_to_idx:
            continue
        rel = str(r["filename"])
        abs_path = base / rel
        if not abs_path.exists():
            # Try without "extracted/" prefix
            abs_path = base / rel.replace("extracted/", "")
        if abs_path.exists():
            found.append((str(abs_path), pl, source_name))
    return found

# Process each source
SOURCES_XC = [
    ("birdclef2026-xc-api-dl-part1", "xc1"),
    ("birdclef2026-xc-api-dl-part2", "xc2"),
    ("birdclef2026-xc-api-dl-part3", "xc3"),
]
SOURCES_EXTRACTED = [
    ("birdclef2026-exp047-bc-extracted", "bc"),
    ("birdclef2026-exp047-inat-extracted", "inat"),
    ("birdclef2026-exp047-anuraset-extracted", "anuraset"),
]

t0 = time.time()
for ds_name, src in SOURCES_XC:
    src_root = LOCAL_DATA / ds_name
    if not src_root.exists():
        print(f"⚠ {ds_name} not found, skip")
        continue
    records = from_xc_dataset(src_root, src)
    all_records.extend(records)
    print(f"  {src}: {len(records)} files")

for ds_name, src in SOURCES_EXTRACTED:
    src_root = LOCAL_DATA / ds_name
    if not src_root.exists():
        print(f"⚠ {ds_name} not found, skip")
        continue
    records = from_extracted(src_root, src)
    all_records.extend(records)
    print(f"  {src}: {len(records)} files")

meta_df = pd.DataFrame(all_records, columns=["filepath", "primary_label", "source"])
print(f"\\nTotal: {len(meta_df)} files")
print(f"  by source: {meta_df['source'].value_counts().to_dict()}")
print(f"  unique species: {meta_df['primary_label'].nunique()} / {N_CLASSES}")
print(f"  build time: {(time.time()-t0)/60:.1f} min")

# Train/val split (stratified by species)
np.random.seed(CFG["val_seed"])
meta_df["target_idx"] = meta_df["primary_label"].map(label_to_idx)
meta_df = meta_df.sample(frac=1, random_state=CFG["val_seed"]).reset_index(drop=True)

val_mask = np.zeros(len(meta_df), dtype=bool)
# Per-species val sampling — min 2 per species for AUC computability
for sp in meta_df["primary_label"].unique():
    sp_idx = meta_df.index[meta_df["primary_label"] == sp].tolist()
    n_val = max(2, int(len(sp_idx) * CFG["val_split"]))  # min 2 for AUC (need 2+ positive)
    val_picks = np.random.choice(sp_idx, size=min(n_val, len(sp_idx)), replace=False)
    val_mask[val_picks] = True

train_df = meta_df[~val_mask].reset_index(drop=True)
val_df = meta_df[val_mask].reset_index(drop=True)
print(f"\\nTrain: {len(train_df)} files, Val: {len(val_df)} files")
print(f"  val species: {val_df['primary_label'].nunique()}")
"""),

        code_cell("dataset", """# Cell 7: Dataset / DataLoader
class AudioDataset(Dataset):
    def __init__(self, df, sr, chunk_len, training=True):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.chunk_len = chunk_len
        self.training = training

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        target = int(row["target_idx"])
        try:
            wav, sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        except Exception:
            wav = np.zeros(self.chunk_len, dtype=np.float32)

        if len(wav) < self.chunk_len:
            wav = np.pad(wav, (0, self.chunk_len - len(wav)))
        else:
            if self.training:
                start = np.random.randint(0, len(wav) - self.chunk_len + 1)
            else:
                start = (len(wav) - self.chunk_len) // 2
            wav = wav[start:start + self.chunk_len]

        return torch.from_numpy(wav).float(), target
"""),

        code_cell("model", """# Cell 8: Model definition (EffNetV2-S + SED head)
class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"],
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
        # x: (B, T, F)
        att = torch.tanh(self.att(x))
        cla = self.cla(x)
        norm_att = F.softmax(att, dim=1)
        clip = (norm_att * cla).sum(dim=1)
        return clip

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["backbone"], pretrained=True, in_chans=3,
            num_classes=0, global_pool="",
            drop_path_rate=0.2,       # ↑ Stochastic Depth for regularization
        )
        self.feat_dim = self.backbone.num_features
        self.head = SEDHead(self.feat_dim, N_CLASSES)
    def forward(self, mel):
        # mel: (B, F, T) -> (B, 3, F, T)
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x)  # (B, C, H', T')
        feat = feat.mean(dim=2).transpose(1, 2)  # (B, T', C)
        logit = self.head(feat)
        return logit
"""),

        code_cell("aug", """# Cell 9: Augmentations (MixUp + SpecAugment)
def mixup_audio(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = lam * x + (1 - lam) * x[perm]
    return x2, y, y[perm], lam

def spec_augment(mel, freq_mask=30, time_mask=60, n_freq=2, n_time=2):  # ↑ 20→30, 40→60
    B, F_, T = mel.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask + 1)
        f0 = np.random.randint(0, max(1, F_ - f))
        mel[:, f0:f0+f, :] = 0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask + 1)
        t0 = np.random.randint(0, max(1, T - t))
        mel[:, :, t0:t0+t] = 0
    return mel
"""),

        code_cell("train_setup", """# Cell 10: DataLoader + optim setup
train_ds = AudioDataset(train_df, CFG["sr"], CHUNK_LEN, training=True)
val_ds = AudioDataset(val_df, CFG["sr"], CHUNK_LEN, training=False)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True)

n_steps = len(train_loader) * CFG["epochs"]
print(f"Steps/epoch: {len(train_loader)}, total: {n_steps}")
"""),

        code_cell("model_setup", """# Cell 11: model + optimizer + scheduler
mel_extractor = MelExtractor().to(DEVICE)
model = Model().to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

def lr_lambda(step):
    if step < CFG["warmup_steps"]:
        return step / max(1, CFG["warmup_steps"])
    progress = (step - CFG["warmup_steps"]) / max(1, n_steps - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
"""),

        code_cell("eval_fn", """# Cell 12: validation eval (matches exp052/053 log style)
@torch.no_grad()
def evaluate():
    model.eval()
    all_logits = []
    all_targets = []
    for wav, tgt in val_loader:
        wav = wav.to(DEVICE, non_blocking=True)
        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            logit = model(mel)
        all_logits.append(logit.float().cpu())
        all_targets.append(tgt)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(all_logits).numpy()
    targets_onehot = np.eye(N_CLASSES)[all_targets.numpy()]

    # Per-class AUC
    aucs_per_class = []
    class_indices_evaluated = []
    for c in range(N_CLASSES):
        if targets_onehot[:, c].sum() < 2:
            continue
        try:
            auc = roc_auc_score(targets_onehot[:, c], probs[:, c])
            aucs_per_class.append(auc)
            class_indices_evaluated.append(c)
        except Exception:
            pass
    aucs_arr = np.array(aucs_per_class)
    val_macro = float(np.mean(aucs_arr)) if len(aucs_arr) else 0.0

    # val_ns22 = average of top-22 non-saturated (AUC<1.0) species (matches exp052/053 metric)
    ns_aucs = aucs_arr[aucs_arr < 1.0]
    if len(ns_aucs) >= 22:
        val_ns22 = float(np.sort(ns_aucs)[:22].mean())  # 22 lowest (hardest) ns species
    elif len(ns_aucs) > 0:
        val_ns22 = float(ns_aucs.mean())
    else:
        val_ns22 = val_macro

    # Per-taxon AUC
    taxon_aucs = {}
    species_df_ = species_df.copy()
    species_df_["idx"] = species_df_["primary_label"].map(label_to_idx)
    for cls in species_df_["class_name"].unique():
        idx_list = species_df_[species_df_["class_name"] == cls]["idx"].tolist()
        cls_aucs = []
        for c in idx_list:
            if targets_onehot[:, c].sum() < 2:
                continue
            try: cls_aucs.append(roc_auc_score(targets_onehot[:, c], probs[:, c]))
            except: pass
        taxon_aucs[cls] = float(np.mean(cls_aucs)) if cls_aucs else float("nan")

    # Class-level stats (matches exp053 log)
    class_stats = {
        "n": len(aucs_arr),
        "median": float(np.median(aucs_arr)) if len(aucs_arr) else 0.0,
        "p25": float(np.percentile(aucs_arr, 25)) if len(aucs_arr) else 0.0,
        "p75": float(np.percentile(aucs_arr, 75)) if len(aucs_arr) else 0.0,
        "n_gt05": int((aucs_arr > 0.5).sum()),
        "n_gt07": int((aucs_arr > 0.7).sum()),
        "n_gt09": int((aucs_arr > 0.9).sum()),
        "n_perfect": int((aucs_arr == 1.0).sum()),
    }
    return val_macro, val_ns22, taxon_aucs, class_stats
"""),

        code_cell("train_loop", """# Cell 13: main train loop (log format matches exp052/053)
best_val = 0.0
best_path = DRIVE_ROOT / "stage1_v2_best.pth"
log_path = DRIVE_ROOT / "stage1_v2_train.log"

step = 0
total_t0 = time.time()
for ep in range(CFG["epochs"]):
    t0 = time.time()
    model.train()
    losses = []
    for batch_i, (wav, tgt) in enumerate(train_loader):
        wav = wav.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE)

        if CFG["mixup_alpha"] > 0:
            wav, tgt_a, tgt_b, lam = mixup_audio(wav, tgt, CFG["mixup_alpha"])
        else:
            tgt_a = tgt; tgt_b = tgt; lam = 1.0

        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            mel = spec_augment(mel, freq_mask=30, time_mask=60)
            logit = model(mel)
            tgt_one_a = F.one_hot(tgt_a, N_CLASSES).float()
            tgt_one_b = F.one_hot(tgt_b, N_CLASSES).float()
            ls = CFG["label_smoothing"]
            tgt_smooth_a = tgt_one_a * (1 - ls) + ls / N_CLASSES
            tgt_smooth_b = tgt_one_b * (1 - ls) + ls / N_CLASSES
            tgt_smooth = lam * tgt_smooth_a + (1 - lam) * tgt_smooth_b
            loss = F.binary_cross_entropy_with_logits(logit, tgt_smooth)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        step += 1
        if batch_i % 100 == 0:
            print(f"  [ep{ep+1} step {batch_i}/{len(train_loader)}] loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    avg_loss = float(np.mean(losses))
    val_macro, val_ns22, taxon_aucs, cstat = evaluate()
    ep_elapsed = time.time() - t0
    total_elapsed = time.time() - total_t0
    cur_lr = scheduler.get_last_lr()[0]

    # Determine BEST flag
    is_best = val_ns22 > best_val
    best_mark = "BEST" if is_best else ""

    # Log line 1: header (exp052/053 style)
    log_l1 = (f"=== Ep {ep+1}/{CFG['epochs']}: loss={avg_loss:.4f} "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} {best_mark} "
              f"lr={cur_lr:.2e} "
              f"({ep_elapsed/60:.1f}min, total {total_elapsed/60:.1f}min) ===")
    # Log line 2: taxon
    log_l2 = "    taxon: " + " ".join(f"{k}={v:.3f}" if not (v != v) else f"{k}=nan"
                                       for k, v in taxon_aucs.items())
    # Log line 3: class stats
    log_l3 = (f"    class: n={cstat['n']} median={cstat['median']:.3f} "
              f"p25={cstat['p25']:.3f} p75={cstat['p75']:.3f} "
              f"#>0.5={cstat['n_gt05']} #>0.7={cstat['n_gt07']} "
              f"#>0.9={cstat['n_gt09']} #perfect={cstat['n_perfect']}")

    print(log_l1); print(log_l2); print(log_l3)
    with open(log_path, "a") as f:
        f.write(log_l1 + "\\n" + log_l2 + "\\n" + log_l3 + "\\n")

    if is_best:
        best_val = val_ns22
        torch.save({"state_dict": model.state_dict(),
                    "val_ns22": val_ns22, "val_macro": val_macro,
                    "ep": ep+1, "cfg": CFG}, best_path)
        print(f"    BEST saved val_ns22={val_ns22:.4f}")

print(f"\\nDone. Best val_ns22: {best_val:.4f}")
print(f"Best ckpt: {best_path}")
"""),

        code_cell("upload_kaggle", """# Cell 14: upload backbone weights to Kaggle Dataset
import json, os, subprocess
from pathlib import Path

USER = "maekeso"
SLUG = "birdclef2026-exp054-stage1-effv2s"
TITLE = "BirdCLEF2026 exp054 Stage 1 v2 EffNetV2-S backbone"

UPLOAD_DIR = Path("/content/upload_stage1_v2")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(best_path, UPLOAD_DIR / "stage1_v2_best.pth")
shutil.copy(log_path, UPLOAD_DIR / "stage1_v2_train.log")

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print(f"OK new dataset")
except Exception as e:
    print(f"create_new err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="stage1 v2",
                                    dir_mode="zip", quiet=False)
        print("OK version created")
    except Exception as e2:
        print(f"create_version err: {str(e2)[:200]}")
print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),

        code_cell("summary", """# Cell 15: summary
print(f"=== Stage 1 v2 complete ===")
print(f"Best val_macro: {best_val:.4f}")
print(f"Total epochs: {CFG['epochs']}")
print(f"Dataset uploaded: maekeso/{SLUG}")
print(f"\\nNext: Stage 2A v2 (BC2026 finetune) with this backbone")
"""),

        code_cell("disconnect", """# Cell 16: auto-disconnect (save compute units)
from google.colab import runtime
runtime.unassign()
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "machine_shape": "hm"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id'):15s} L={n}")
