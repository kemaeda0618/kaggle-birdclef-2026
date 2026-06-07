"""Generate Stage 2 v2 NB for Colab Blackwell.

Loads Stage 1 v2 backbone (Kaggle Dataset) + finetune on BC2026 train_audio.

Inputs:
  - maekeso/birdclef2026-exp054-stage1-effv2s (Stage 1 v2 backbone)
  - birdclef-2026 competition data (train_audio + labeled_ss)

Output: Stage 2 v2 backbone → maekeso/birdclef2026-exp054-stage2-effv2s
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp054\notebook\nb_stage2_v2.ipynb")

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
        md_cell("hdr", """# exp054 Stage 2 v2 — BC2026 Finetune (Colab Blackwell)

**Purpose**: BC2026 train_audio で finetune from Stage 1 v2 backbone

**Inputs**:
- `maekeso/birdclef2026-exp054-stage1-effv2s` (Stage 1 v2 backbone)
- BC2026 competition: train_audio + train_soundscapes_labels.csv (labeled_ss)

**Train**:
- 20 epochs
- Differential LR: backbone 5e-5, head 5e-4
- MixUp α=1.5, SpecAug (30/60)
- Same drop_path 0.2

**Val**:
- labeled_ss (66 files, 35 species, soundscape domain = test 近い)
- val_macro + val_ns22 + per-taxon AUC

**Output**: `maekeso/birdclef2026-exp054-stage2-effv2s`

**Expected**:
- val_ns22: 0.92+ (vs Stage 2A v1 0.9091)
- Standalone LB: 0.82-0.88 (vs v1 0.749)
- 4-way blend (exp048 + Stage 2 v2 Aves-only): **0.952-0.957** target

vs Stage 2A v1 (exp053):
- Backbone: Stage 1 v1 (159 Aves) → Stage 1 v2 (210+ species)
- Train data: 同じ BC2026 train_audio
- Better non-Aves transfer 期待
"""),

        code_cell("install", """# Cell 1: install deps + drive mount
!pip install -q timm==1.0.11 soundfile librosa kaggle 2>&1 | tail -1

from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp054")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Drive: {DRIVE_ROOT}")
"""),

        code_cell("dl_data", """# Cell 2: Download Stage 1 v2 backbone + BC2026 competition data
import os, json, time
from pathlib import Path

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

_kgat = json.loads((KAGGLE_DIR / "kaggle.json").read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat
print(f"  KAGGLE_API_TOKEN set")

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)
t0 = time.time()

# 1. Download Stage 1 v2 backbone
S1V2_DIR = LOCAL_DATA / "stage1_v2"
if not S1V2_DIR.exists() or not any(S1V2_DIR.iterdir()):
    S1V2_DIR.mkdir(exist_ok=True)
    print("  DL Stage 1 v2 backbone...")
    api.dataset_download_files("maekeso/birdclef2026-exp054-stage1-effv2s",
                                path=str(S1V2_DIR), unzip=True, quiet=False)
    print(f"  ✓ Files: {list(S1V2_DIR.iterdir())}")
else:
    print(f"  ✓ Stage 1 v2 backbone exists, skip")

# 2. Download BC2026 competition data
BC_DIR = LOCAL_DATA / "birdclef-2026"
if not BC_DIR.exists() or not (BC_DIR / "train.csv").exists():
    BC_DIR.mkdir(exist_ok=True)
    print("  DL BC2026 competition data...")
    api.competition_download_files("birdclef-2026", path=str(BC_DIR), quiet=False)
    # Unzip
    import zipfile
    zip_path = BC_DIR / "birdclef-2026.zip"
    if zip_path.exists():
        print("  Unzipping...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(BC_DIR)
        zip_path.unlink()
    print(f"  ✓ BC2026 files extracted")
else:
    print(f"  ✓ BC2026 exists, skip")

print(f"\\nDL total: {(time.time()-t0)/60:.1f} min")

# Verify
backbone_path = next(S1V2_DIR.rglob("*.pth"), None)
assert backbone_path is not None, f"Stage 1 v2 backbone not found in {S1V2_DIR}"
print(f"\\nBackbone: {backbone_path} ({backbone_path.stat().st_size/1e6:.1f} MB)")

train_csv = BC_DIR / "train.csv"
labeled_ss_csv = BC_DIR / "train_soundscapes_labels.csv"
train_audio_dir = BC_DIR / "train_audio"
train_soundscapes_dir = BC_DIR / "train_soundscapes"
print(f"train.csv: {train_csv.exists()}")
print(f"labeled_ss.csv: {labeled_ss_csv.exists()}")
print(f"train_audio: {train_audio_dir.exists()} ({sum(1 for _ in train_audio_dir.rglob('*.ogg') if _.is_file()) if train_audio_dir.exists() else 0} files)")
print(f"train_soundscapes: {train_soundscapes_dir.exists()}")
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
from torch.amp import autocast, GradScaler  # new API
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
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
print(f"BC2026: {{N_CLASSES}} species")
print(f"  by class: {{species_df['class_name'].value_counts().to_dict()}}")
"""),

        code_cell("config", """# Cell 5: training config (Stage 2 finetune)
CFG = dict(
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
    sr=32000, chunk_sec=5, n_mels=128, n_fft=2048, hop_length=512,
    fmin=20, fmax=16000,
    epochs=20,
    batch_size=192,               # batch 192 (Stage 1 と同、Blackwell max)
    lr_backbone=3e-5,             # ↓ from 1e-4 — conservative to preserve Stage 1 v2 knowledge
    lr_head=3e-4,                 # ↓ from 1e-3 — paired reduction
    weight_decay=1e-3,
    warmup_steps=300,             # ↓ from 500 for lower lr
    label_smoothing=0.1,
    mixup_alpha=1.5,
    grad_clip=2.0,
    num_workers=4,
    val_split=0.1,                # 10% from train_audio for backup val
    val_seed=42,
)
for k, v in CFG.items(): print(f"  {k}: {v}")
CHUNK_LEN = CFG["sr"] * CFG["chunk_sec"]
"""),

        code_cell("data_meta", """# Cell 6: build train + val meta_df from BC2026 train_audio (random 10% split)
# NOTE: labeled_ss は per-chunk annotation で current AudioDataset (center 5s) と
#       mismatch するため使用しない。Stage 1 v2 と同じ random split approach。
LOCAL_DATA = Path("/content/data")
BC_DIR = LOCAL_DATA / "birdclef-2026"

# All records from BC2026 train_audio
train_csv = pd.read_csv(BC_DIR / "train.csv")
print(f"train.csv: {len(train_csv)} rows")
print(f"  columns: {train_csv.columns.tolist()}")
print(f"  BC2026 species: {train_csv['primary_label'].nunique()}")

train_audio_dir = BC_DIR / "train_audio"
all_records = []
for _, r in train_csv.iterrows():
    pl = str(r["primary_label"])
    if pl not in label_to_idx:
        continue
    fp = train_audio_dir / str(r["filename"])
    if fp.exists():
        all_records.append((str(fp), pl, "train_audio"))

print(f"\\nTotal records found: {len(all_records)}")

meta_df = pd.DataFrame(all_records, columns=["filepath", "primary_label", "source"])
meta_df["target_idx"] = meta_df["primary_label"].map(label_to_idx)

# Train/val split (stratified by species, min 2 per species for AUC)
np.random.seed(CFG["val_seed"])
meta_df = meta_df.sample(frac=1, random_state=CFG["val_seed"]).reset_index(drop=True)

val_mask = np.zeros(len(meta_df), dtype=bool)
for sp in meta_df["primary_label"].unique():
    sp_idx = meta_df.index[meta_df["primary_label"] == sp].tolist()
    n_val = max(2, int(len(sp_idx) * CFG["val_split"]))
    val_picks = np.random.choice(sp_idx, size=min(n_val, len(sp_idx)), replace=False)
    val_mask[val_picks] = True

train_df = meta_df[~val_mask].reset_index(drop=True)
val_df = meta_df[val_mask].reset_index(drop=True)
print(f"\\nTrain: {len(train_df)} files ({train_df['primary_label'].nunique()} species)")
print(f"Val:   {len(val_df)} files ({val_df['primary_label'].nunique()} species)")
"""),

        code_cell("dataset", """# Cell 7: Dataset / DataLoader (same as Stage 1)
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
            if wav.ndim > 1: wav = wav.mean(axis=1)
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

        code_cell("model", """# Cell 8: Model definition + Stage 1 v2 backbone load
class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"],
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    def forward(self, wav):
        mel = self.mel(wav); mel = self.db(mel)
        mel = torch.clamp(mel, -80.0, 0.0); mel = (mel + 40.0) / 40.0
        return mel

class SEDHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.att = nn.Linear(in_dim, n_classes)
        self.cla = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        att = torch.tanh(self.att(x)); cla = self.cla(x)
        norm_att = F.softmax(att, dim=1)
        return (norm_att * cla).sum(dim=1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["backbone"], pretrained=False, in_chans=3,
            num_classes=0, global_pool="",
            drop_path_rate=0.2,
        )
        self.feat_dim = self.backbone.num_features
        self.head = SEDHead(self.feat_dim, N_CLASSES)
    def forward(self, mel):
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x).mean(dim=2).transpose(1, 2)
        return self.head(feat)

# Load Stage 1 v2 backbone weights
S1V2_BACKBONE = next(Path("/content/data/stage1_v2").rglob("*.pth"), None)
assert S1V2_BACKBONE is not None, "Stage 1 v2 backbone not found"
print(f"Loading Stage 1 v2 backbone: {S1V2_BACKBONE}")
ckpt = torch.load(S1V2_BACKBONE, map_location="cpu", weights_only=False)
print(f"  Stage 1 v2 val_ns22: {ckpt.get('val_ns22', '?')}")
print(f"  Stage 1 v2 val_macro: {ckpt.get('val_macro', '?')}")

mel_extractor = MelExtractor().to(DEVICE)
model = Model().to(DEVICE)

# Load backbone state but keep head random (new task)
state = ckpt["state_dict"]
backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
print(f"  backbone missing: {len(missing)}, unexpected: {len(unexpected)}")

# Head is kept random init (new task = new BC2026 head)
print(f"\\nModel params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"  backbone: {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M (Stage 1 v2 loaded)")
print(f"  head: {sum(p.numel() for p in model.head.parameters())/1e6:.2f}M (random init)")

# === SANITY CHECK: verify backbone weights actually loaded ===
print(f"\\n=== Backbone load sanity check ===")
print(f"  missing keys: {len(missing)} (should be 0 or small)")
if missing: print(f"    first 3: {missing[:3]}")
print(f"  unexpected keys: {len(unexpected)} (should be 0)")
if unexpected: print(f"    first 3: {unexpected[:3]}")

# Compare a sample weight value: model loaded == backbone_state expected
import torch as _torch
sample_keys = [k for k in backbone_state.keys() if "weight" in k][:3]
all_match = True
for sk in sample_keys:
    expected = backbone_state[sk].abs().mean().item()
    actual = model.backbone.state_dict()[sk].abs().mean().item()
    match = abs(expected - actual) < 1e-6
    if not match: all_match = False
    print(f"  {sk[:60]:60s}: expected={expected:.6f}, actual={actual:.6f}, match={match}")
assert all_match, "❌ Stage 1 v2 backbone NOT loaded correctly!"
print(f"\\n✓ Backbone load verified ({len(sample_keys)} keys checked)")
"""),

        code_cell("aug", """# Cell 9: Augmentations
def mixup_audio(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = lam * x + (1 - lam) * x[perm]
    return x2, y, y[perm], lam

def spec_augment(mel, freq_mask=30, time_mask=60, n_freq=2, n_time=2):
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

        code_cell("train_setup", """# Cell 10: DataLoader + differential optimizer
train_ds = AudioDataset(train_df, CFG["sr"], CHUNK_LEN, training=True)
val_ds = AudioDataset(val_df, CFG["sr"], CHUNK_LEN, training=False)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True)

n_steps = len(train_loader) * CFG["epochs"]
print(f"Steps/epoch: {len(train_loader)}, total: {n_steps}")

# Differential LR optimizer (backbone low, head high)
optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": CFG["lr_backbone"]},
    {"params": model.head.parameters(), "lr": CFG["lr_head"]},
], weight_decay=CFG["weight_decay"])

def lr_lambda(step):
    if step < CFG["warmup_steps"]:
        return step / max(1, CFG["warmup_steps"])
    progress = (step - CFG["warmup_steps"]) / max(1, n_steps - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
"""),

        code_cell("eval_fn", """# Cell 11: validation eval (same metric as Stage 1)
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

    aucs_per_class = []
    for c in range(N_CLASSES):
        if targets_onehot[:, c].sum() < 2: continue
        try: aucs_per_class.append(roc_auc_score(targets_onehot[:, c], probs[:, c]))
        except: pass
    aucs_arr = np.array(aucs_per_class)
    val_macro = float(np.mean(aucs_arr)) if len(aucs_arr) else 0.0
    ns_aucs = aucs_arr[aucs_arr < 1.0]
    val_ns22 = float(np.sort(ns_aucs)[:22].mean()) if len(ns_aucs) >= 22 else float(np.mean(ns_aucs)) if len(ns_aucs) > 0 else val_macro

    taxon_aucs = {}
    sp_df_ = species_df.copy(); sp_df_["idx"] = sp_df_["primary_label"].map(label_to_idx)
    for cls in sp_df_["class_name"].unique():
        idx_list = sp_df_[sp_df_["class_name"] == cls]["idx"].tolist()
        cls_aucs = []
        for c in idx_list:
            if targets_onehot[:, c].sum() < 2: continue
            try: cls_aucs.append(roc_auc_score(targets_onehot[:, c], probs[:, c]))
            except: pass
        taxon_aucs[cls] = float(np.mean(cls_aucs)) if cls_aucs else float("nan")

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

        code_cell("train_loop", """# Cell 12: main train loop (same log format as Stage 1)
best_val = 0.0
best_path = DRIVE_ROOT / "stage2_v2_best.pth"
log_path = DRIVE_ROOT / "stage2_v2_train.log"

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
            lrs = [g["lr"] for g in optimizer.param_groups]
            print(f"  [ep{ep+1} step {batch_i}/{len(train_loader)}] loss={loss.item():.4f} lr_bk={lrs[0]:.2e} lr_hd={lrs[1]:.2e}")

    avg_loss = float(np.mean(losses))
    val_macro, val_ns22, taxon_aucs, cstat = evaluate()
    ep_elapsed = time.time() - t0
    total_elapsed = time.time() - total_t0
    lrs = [g["lr"] for g in optimizer.param_groups]

    is_best = val_ns22 > best_val
    best_mark = "BEST" if is_best else ""

    log_l1 = (f"=== Ep {ep+1}/{CFG['epochs']}: loss={avg_loss:.4f} "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} {best_mark} "
              f"lr_bk={lrs[0]:.2e} lr_hd={lrs[1]:.2e} "
              f"({ep_elapsed/60:.1f}min, total {total_elapsed/60:.1f}min) ===")
    log_l2 = "    taxon: " + " ".join(f"{k}={v:.3f}" if not (v != v) else f"{k}=nan"
                                       for k, v in taxon_aucs.items())
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
                    "ep": ep+1, "cfg": CFG,
                    "stage1_v2_val_ns22": ckpt.get("val_ns22", 0)}, best_path)
        print(f"    BEST saved val_ns22={val_ns22:.4f}")

print(f"\\nDone. Best val_ns22: {best_val:.4f}")
print(f"Best ckpt: {best_path}")
"""),

        code_cell("upload_kaggle", """# Cell 13: upload to Kaggle Dataset (self-contained auth)
import os, json, shutil
from pathlib import Path

# Kaggle auth setup
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_DIR.mkdir(exist_ok=True)
if not (KAGGLE_DIR / "kaggle.json").exists():
    src_kg = Path("/content/drive/MyDrive/kaggle/kaggle.json")
    if src_kg.exists():
        shutil.copy(src_kg, KAGGLE_DIR / "kaggle.json")
        os.chmod(KAGGLE_DIR / "kaggle.json", 0o600)
_kgat = json.loads((KAGGLE_DIR / "kaggle.json").read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp054-stage2-effv2s"
TITLE = "BirdCLEF2026 exp054 Stage2 v2 backbone"  # 38 文字 (50 以内)

UPLOAD_DIR = Path("/content/upload_stage2_v2")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(best_path, UPLOAD_DIR / "stage2_v2_best.pth")
shutil.copy(log_path, UPLOAD_DIR / "stage2_v2_train.log")

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset created")
except Exception as e:
    print(f"create_new err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="Stage 2 v2",
                                    dir_mode="zip", quiet=False)
        print("OK version created")
    except Exception as e2:
        print(f"create_version err: {str(e2)[:200]}")
print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),

        code_cell("summary", """# Cell 14: summary
print(f"=== Stage 2 v2 complete ===")
print(f"Best val_ns22: {best_val:.4f}")
print(f"Total epochs: {CFG['epochs']}")
print(f"Dataset uploaded: maekeso/{SLUG}")
print(f"\\nNext: Inference NB → submit (standalone LB confirm) → 4-way blend")
"""),

        code_cell("disconnect", """# Cell 15: auto-disconnect
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
