"""Generate R2 NB: pseudo gen + R2 train for Colab Blackwell.

R1 (Stage 2 v2) teacher → pseudo on BC2026 train_soundscapes (10,592 files)
R2 train: train_audio (real label) + train_soundscapes (pseudo) on Stage 2 v2 R1 warm start
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp054\notebook\nb_r2.ipynb")

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
        md_cell("hdr", """# exp054 R2 — Pseudo Gen + R2 Train (Colab Blackwell)

**Purpose**: Stage 2 v2 R1 を teacher、BC2026 train_soundscapes 10k に pseudo label 付与 → R2 train で soundscape domain 学習

**Inputs**:
- `maekeso/birdclef2026-exp054-stage2-effv2s` (R1 ckpt, LB 0.749)
- birdclef-2026: train_audio + **train_soundscapes 10,592 files** (key!)

**Pipeline**:
1. Pseudo gen: R1 で train_soundscapes 全 file 推論 → (12 chunk × 234 species) soft pseudo
2. R2 train: train_audio (real one-hot) + train_soundscapes (pseudo soft)
3. Backbone: R1 warm start (BC2026 finetune 済を再 finetune)
4. 15-20 ep

**Output**: `maekeso/birdclef2026-exp054-stage2-r2-effv2s`

**Expected**:
- Standalone LB: 0.78-0.85 (R1 0.749 から +0.03-0.10)
- 4-way blend: 0.952-0.957 (gold border 圏)

**Why R2 matters**:
- R1 = focal-only training → soundscape test で domain shift -0.16 gap
- R2 = +soundscape pseudo training → 真の test domain で初学習
- 期待: domain gap 縮小 +0.05-0.10 lift
"""),

        code_cell("install", """# Cell 1: install + drive mount
!pip install -q timm==1.0.11 soundfile librosa kaggle 2>&1 | tail -1

from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

# ★★ 正しい Drive path: output/<exp> 必須 ★★
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp054")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Drive: {DRIVE_ROOT}")
"""),

        code_cell("dl_data", """# Cell 2: Download R1 ckpt + BC2026 data (auth + DL self-contained)
import os, json, time, shutil
from pathlib import Path

# Kaggle auth (KAGGLE_API_TOKEN env var = KGAT_ Bearer)
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
print(f"  ✓ Auth OK")

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)

# 1. Stage 2 v2 R1 ckpt
R1_DIR = LOCAL_DATA / "stage2_v2_r1"
if not R1_DIR.exists() or not any(R1_DIR.iterdir()):
    R1_DIR.mkdir(exist_ok=True)
    print("DL R1 ckpt...")
    api.dataset_download_files("maekeso/birdclef2026-exp054-stage2-effv2s",
                                path=str(R1_DIR), unzip=True, quiet=False)
print(f"  R1 files: {list(R1_DIR.iterdir())[:3]}")

# 2. BC2026 competition
BC_DIR = LOCAL_DATA / "birdclef-2026"
if not BC_DIR.exists() or not (BC_DIR / "train.csv").exists():
    BC_DIR.mkdir(exist_ok=True)
    print("DL BC2026...")
    api.competition_download_files("birdclef-2026", path=str(BC_DIR), quiet=False)
    zip_path = BC_DIR / "birdclef-2026.zip"
    if zip_path.exists():
        print("Unzipping...")
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(BC_DIR)
        zip_path.unlink()
print(f"  BC2026 train_audio exist: {(BC_DIR/'train_audio').exists()}")
print(f"  BC2026 train_soundscapes exist: {(BC_DIR/'train_soundscapes').exists()}")
ss_files = list((BC_DIR/'train_soundscapes').glob('*.ogg')) if (BC_DIR/'train_soundscapes').exists() else []
print(f"  train_soundscapes files: {len(ss_files)}")
"""),

        code_cell("setup", """# Cell 3: imports + GPU
import os, sys, json, time, math, random, gc, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchaudio
import soundfile as sf
import librosa
import timm
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
"""),

        code_cell("species", f"""# Cell 4: BC2026 species (234)
__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {{l: i for i, l in enumerate(PRIMARY_LABELS)}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
print(f"BC2026: {{N_CLASSES}} species")
"""),

        code_cell("config", """# Cell 5: R2 training config
CFG = dict(
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
    sr=32000, chunk_sec=5, n_mels=128, n_fft=2048, hop_length=512,
    fmin=20, fmax=16000,
    epochs=15,                      # ↓ from 20 — R1 warm start で短期
    batch_size=192,
    lr_backbone=2e-5,               # ↓ from 3e-5 — R1 warm start + lower for finer
    lr_head=2e-4,
    weight_decay=1e-3,
    warmup_steps=300,
    label_smoothing=0.1,
    mixup_alpha=1.5,
    grad_clip=2.0,
    num_workers=8,
    val_split=0.1,
    val_seed=42,
    # R2 specific
    pseudo_weight=0.7,              # train_soundscapes pseudo の loss weight (real label より低)
    pseudo_threshold=0.0,           # 0 = soft pseudo そのまま、>0 で threshold-mask
)
for k, v in CFG.items(): print(f"  {k}: {v}")
CHUNK_LEN = CFG["sr"] * CFG["chunk_sec"]
N_WINDOWS = 12  # 60s / 5s
"""),

        code_cell("model", """# Cell 6: Model + R1 backbone load
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
    def __init__(self, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["backbone"], pretrained=False, in_chans=3,
            num_classes=0, global_pool="",
            drop_path_rate=drop_path_rate,
        )
        self.head = SEDHead(self.backbone.num_features, N_CLASSES)
    def forward(self, mel):
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x).mean(dim=2).transpose(1, 2)
        return self.head(feat)

# Load R1 ckpt (teacher + student warm start)
R1_CKPT_PATH = next(Path("/content/data/stage2_v2_r1").rglob("*.pth"), None)
assert R1_CKPT_PATH is not None, "R1 ckpt missing"
print(f"R1 ckpt: {R1_CKPT_PATH}")
r1_ckpt = torch.load(R1_CKPT_PATH, map_location="cpu", weights_only=False)
print(f"  R1 val_ns22: {r1_ckpt.get('val_ns22', '?')}, val_macro: {r1_ckpt.get('val_macro', '?')}")

mel_extractor = MelExtractor().to(DEVICE)

# Teacher (eval mode, for pseudo gen)
teacher = Model(drop_path_rate=0.0).to(DEVICE)
teacher.load_state_dict(r1_ckpt["state_dict"], strict=True)
teacher.eval()
print(f"\\nTeacher loaded (eval mode)")

# Student (train mode, warm start from R1)
student = Model(drop_path_rate=0.2).to(DEVICE)
student.load_state_dict(r1_ckpt["state_dict"], strict=True)
print(f"Student loaded (warm start from R1)")
print(f"  params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")
"""),

        code_cell("pseudo_gen", """# Cell 7: Pseudo gen on BC2026 train_soundscapes
LOCAL_DATA = Path("/content/data")
BC_DIR = LOCAL_DATA / "birdclef-2026"
TRAIN_SS_DIR = BC_DIR / "train_soundscapes"

ss_files = sorted(TRAIN_SS_DIR.glob("*.ogg"))
print(f"train_soundscapes files: {len(ss_files)}")
assert len(ss_files) > 0, "No train_soundscapes found"

PSEUDO_PATH = DRIVE_ROOT / "train_ss_pseudo.npz"

if PSEUDO_PATH.exists():
    print(f"\\nPseudo cache exists, loading...")
    _cache = np.load(PSEUDO_PATH, allow_pickle=True)
    pseudo_arr = _cache["pseudo"]  # (N_files, N_WINDOWS, N_CLASSES)
    pseudo_files = _cache["files"]
    print(f"  Loaded: shape={pseudo_arr.shape}, files={len(pseudo_files)}")
else:
    print(f"\\nRunning pseudo inference (~10 min)...")
    pseudo_arr = np.zeros((len(ss_files), N_WINDOWS, N_CLASSES), dtype=np.float32)
    pseudo_files = []
    t0 = time.time()
    torch.set_grad_enabled(False)

    for fi, fp in enumerate(ss_files):
        try:
            wav, sr = sf.read(str(fp), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
            target_len = N_WINDOWS * CHUNK_LEN
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            chunks = wav[:target_len].reshape(N_WINDOWS, CHUNK_LEN)
            batch = torch.from_numpy(chunks).float().to(DEVICE)
            with autocast('cuda', dtype=torch.bfloat16):
                mel = mel_extractor(batch)
                logit = teacher(mel)
            prob = torch.sigmoid(logit).float().cpu().numpy()
            pseudo_arr[fi] = prob
            pseudo_files.append(fp.name)
        except Exception as e:
            print(f"  [err] {fp.name}: {str(e)[:80]}")
            pseudo_files.append(fp.name)
        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(ss_files)}] {(fi+1)/elapsed*60:.1f}/min ETA {(len(ss_files)-fi-1)/((fi+1)/elapsed)/60:.1f}min")

    pseudo_files = np.array(pseudo_files)
    np.savez_compressed(PSEUDO_PATH, pseudo=pseudo_arr, files=pseudo_files)
    print(f"\\nPseudo done in {(time.time()-t0)/60:.1f}min")
    print(f"Saved {PSEUDO_PATH} ({PSEUDO_PATH.stat().st_size/1e6:.1f} MB)")

torch.set_grad_enabled(True)
# Pseudo stats
print(f"\\nPseudo stats:")
print(f"  mean: {pseudo_arr.mean():.4f}")
print(f"  max: {pseudo_arr.max():.4f}")
print(f"  >0.5 per chunk avg: {(pseudo_arr > 0.5).sum() / (pseudo_arr.shape[0] * pseudo_arr.shape[1]):.2f} species")
print(f"  >0.8 per chunk avg: {(pseudo_arr > 0.8).sum() / (pseudo_arr.shape[0] * pseudo_arr.shape[1]):.2f} species")

# Free teacher (no longer needed)
del teacher
gc.collect()
torch.cuda.empty_cache()
"""),

        code_cell("data_meta", """# Cell 8: build R2 train data (train_audio + train_soundscapes pseudo)
train_csv = pd.read_csv(BC_DIR / "train.csv")
TRAIN_AUDIO_DIR = BC_DIR / "train_audio"

# train_audio records (real one-hot label)
train_audio_records = []
for _, r in train_csv.iterrows():
    pl = str(r["primary_label"])
    if pl not in label_to_idx: continue
    fp = TRAIN_AUDIO_DIR / str(r["filename"])
    if fp.exists():
        train_audio_records.append({
            "filepath": str(fp),
            "target_idx": label_to_idx[pl],
            "is_pseudo": False,
            "pseudo_idx": -1,
        })
print(f"train_audio records: {len(train_audio_records)}")

# train_soundscapes records (pseudo, one per file with chunk randomly sampled at __getitem__)
ss_records = []
ss_path_to_idx = {f: i for i, f in enumerate(pseudo_files)}
for fi, fname in enumerate(pseudo_files):
    fp = TRAIN_SS_DIR / fname
    if fp.exists():
        ss_records.append({
            "filepath": str(fp),
            "target_idx": -1,
            "is_pseudo": True,
            "pseudo_idx": fi,
        })
print(f"train_soundscapes records: {len(ss_records)}")

all_records = train_audio_records + ss_records
meta_df = pd.DataFrame(all_records)
print(f"\\nTotal: {len(meta_df)} records ({(meta_df['is_pseudo']==False).sum()} real + {(meta_df['is_pseudo']==True).sum()} pseudo)")

# Train/val split — val from train_audio only (avoid pseudo noise in val signal)
np.random.seed(CFG["val_seed"])
real_df = meta_df[~meta_df["is_pseudo"]].sample(frac=1, random_state=CFG["val_seed"]).reset_index(drop=True)
val_mask_real = np.zeros(len(real_df), dtype=bool)
# species 別 split (real のみ)
for sp_idx in real_df["target_idx"].unique():
    indices = real_df.index[real_df["target_idx"] == sp_idx].tolist()
    n_val = max(2, int(len(indices) * CFG["val_split"]))
    val_picks = np.random.choice(indices, size=min(n_val, len(indices)), replace=False)
    val_mask_real[val_picks] = True

train_real = real_df[~val_mask_real].reset_index(drop=True)
val_real = real_df[val_mask_real].reset_index(drop=True)
pseudo_df = meta_df[meta_df["is_pseudo"]].reset_index(drop=True)

train_df = pd.concat([train_real, pseudo_df]).reset_index(drop=True)
val_df = val_real

print(f"\\nTrain: {len(train_df)} ({(train_df['is_pseudo']==False).sum()} real + {(train_df['is_pseudo']==True).sum()} pseudo)")
print(f"Val:   {len(val_df)} (real only)")
"""),

        code_cell("dataset", """# Cell 9: Dataset (handles both real one-hot and pseudo soft)
class R2Dataset(Dataset):
    def __init__(self, df, pseudo_arr_, sr, chunk_len, training=True):
        self.df = df.reset_index(drop=True)
        self.pseudo_arr = pseudo_arr_
        self.sr = sr
        self.chunk_len = chunk_len
        self.training = training

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        is_pseudo = bool(row["is_pseudo"])

        try:
            wav, sr = sf.read(path, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        except Exception:
            wav = np.zeros(self.chunk_len, dtype=np.float32)

        if is_pseudo:
            # train_soundscapes: 60s file, random chunk 5s + corresponding pseudo target
            target_len = N_WINDOWS * self.chunk_len
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            chunk_idx = np.random.randint(0, N_WINDOWS) if self.training else N_WINDOWS // 2
            wav_chunk = wav[chunk_idx*self.chunk_len:(chunk_idx+1)*self.chunk_len]
            target = self.pseudo_arr[int(row["pseudo_idx"]), chunk_idx]  # (N_CLASSES,)
            return torch.from_numpy(wav_chunk).float(), torch.from_numpy(target).float(), True
        else:
            # train_audio: focal recording, random 5s window + one-hot target
            if len(wav) < self.chunk_len:
                wav = np.pad(wav, (0, self.chunk_len - len(wav)))
            else:
                if self.training:
                    start = np.random.randint(0, len(wav) - self.chunk_len + 1)
                else:
                    start = (len(wav) - self.chunk_len) // 2
                wav = wav[start:start + self.chunk_len]
            target_idx = int(row["target_idx"])
            target = np.zeros(N_CLASSES, dtype=np.float32)
            target[target_idx] = 1.0
            return torch.from_numpy(wav).float(), torch.from_numpy(target).float(), False
"""),

        code_cell("aug", """# Cell 10: Augmentations
def mixup_audio(x, y, is_pseudo, alpha=1.0):
    if alpha <= 0:
        return x, y, y, is_pseudo, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = lam * x + (1 - lam) * x[perm]
    return x2, y, y[perm], is_pseudo, lam

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

        code_cell("train_setup", """# Cell 11: DataLoader + optim
train_ds = R2Dataset(train_df, pseudo_arr, CFG["sr"], CHUNK_LEN, training=True)
val_ds = R2Dataset(val_df, pseudo_arr, CFG["sr"], CHUNK_LEN, training=False)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True)

n_steps = len(train_loader) * CFG["epochs"]
print(f"Steps/epoch: {len(train_loader)}, total: {n_steps}")

# Differential LR (R1 warm start なので backbone も少し動かす)
optimizer = optim.AdamW([
    {"params": student.backbone.parameters(), "lr": CFG["lr_backbone"]},
    {"params": student.head.parameters(), "lr": CFG["lr_head"]},
], weight_decay=CFG["weight_decay"])

def lr_lambda(step):
    if step < CFG["warmup_steps"]:
        return step / max(1, CFG["warmup_steps"])
    progress = (step - CFG["warmup_steps"]) / max(1, n_steps - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
"""),

        code_cell("eval_fn", """# Cell 12: validation eval
@torch.no_grad()
def evaluate():
    student.eval()
    all_logits = []
    all_targets_idx = []  # val is real, target_idx is well-defined
    for wav, tgt, is_pseudo in val_loader:
        wav = wav.to(DEVICE, non_blocking=True)
        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            logit = student(mel)
        all_logits.append(logit.float().cpu())
        # val: real one-hot, target_idx = argmax
        all_targets_idx.append(tgt.argmax(dim=1))
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets_idx)

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

        code_cell("train_loop", """# Cell 13: R2 train loop (real one-hot + pseudo soft, weighted loss)
best_val = 0.0
best_path = DRIVE_ROOT / "stage2_v2_r2_best.pth"
log_path = DRIVE_ROOT / "stage2_v2_r2_train.log"

step = 0
total_t0 = time.time()
for ep in range(CFG["epochs"]):
    t0 = time.time()
    student.train()
    losses = []
    for batch_i, (wav, tgt, is_pseudo) in enumerate(train_loader):
        wav = wav.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE)
        is_pseudo = is_pseudo.to(DEVICE)  # (B,) bool

        if CFG["mixup_alpha"] > 0:
            wav, tgt_a, tgt_b, is_pseudo_a, lam = mixup_audio(wav, tgt, is_pseudo, CFG["mixup_alpha"])
            # mix is_pseudo = OR (簡単化、もし片方 pseudo なら pseudo weight 適用)
        else:
            tgt_a = tgt; tgt_b = tgt; is_pseudo_a = is_pseudo; lam = 1.0

        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            mel = spec_augment(mel, freq_mask=30, time_mask=60)
            logit = student(mel)
            # Label smoothing (real のみ、pseudo はそのまま soft)
            ls = CFG["label_smoothing"]
            tgt_smooth_a = torch.where(is_pseudo_a.unsqueeze(-1), tgt_a, tgt_a * (1 - ls) + ls / N_CLASSES)
            tgt_smooth_b = tgt_smooth_a  # mixup partner、簡略化
            tgt_smooth = lam * tgt_smooth_a + (1 - lam) * tgt_smooth_b
            # Loss with per-sample weight (pseudo は CFG.pseudo_weight 倍)
            loss_per_sample = F.binary_cross_entropy_with_logits(logit, tgt_smooth, reduction="none").mean(dim=1)
            weights = torch.where(is_pseudo_a, CFG["pseudo_weight"], 1.0)
            loss = (loss_per_sample * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), CFG["grad_clip"])
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
        torch.save({"state_dict": student.state_dict(),
                    "val_ns22": val_ns22, "val_macro": val_macro,
                    "ep": ep+1, "cfg": CFG,
                    "stage2_v2_r1_val_ns22": r1_ckpt.get("val_ns22", 0)}, best_path)
        print(f"    BEST saved val_ns22={val_ns22:.4f}")

print(f"\\nDone. Best val_ns22: {best_val:.4f}")
"""),

        code_cell("upload_kaggle", """# Cell 14: upload R2 ckpt to Kaggle Dataset (self-contained auth)
import os, json, shutil
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
if not (KAGGLE_DIR / "kaggle.json").exists():
    src_kg = Path("/content/drive/MyDrive/kaggle/kaggle.json")
    shutil.copy(src_kg, KAGGLE_DIR / "kaggle.json")
    os.chmod(KAGGLE_DIR / "kaggle.json", 0o600)
_kgat = json.loads((KAGGLE_DIR / "kaggle.json").read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp054-stage2-r2-effv2s"
TITLE = "BirdCLEF2026 exp054 R2 v2 backbone"  # 36 文字

UPLOAD_DIR = Path("/content/upload_r2")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(best_path, UPLOAD_DIR / "stage2_v2_r2_best.pth")
shutil.copy(log_path, UPLOAD_DIR / "stage2_v2_r2_train.log")

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset")
except Exception as e:
    print(f"create_new err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="R2",
                                    dir_mode="zip", quiet=False)
        print("OK version")
    except Exception as e2:
        print(f"err: {str(e2)[:200]}")
print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),

        code_cell("summary", """# Cell 15: summary
print(f"=== R2 complete ===")
print(f"Best val_ns22: {best_val:.4f}")
print(f"Dataset: maekeso/{SLUG}")
print(f"\\nNext: 推論 NB run + submit")
"""),

        code_cell("disconnect", """# Cell 16: auto-disconnect
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
