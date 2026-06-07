"""Generate exp058 NB2: Combined training (effv2_b0 + SED head, BC2026 hard + XC pseudo).

Path: OverfitOracle 1-stage combined training。
NB1 (pseudo gen) で生成した XC frame-level pseudo を、BC2026 train_audio (hard label) と
concat して 1-stage で training。

Inputs:
- maekeso/birdclef2026-exp058-xc-pseudo (NB1 output)
- birdclef-2026 (competition data)

Output:
- maekeso/birdclef2026-exp058-effv2b0-combined (final backbone + SED head ckpt)

Target: Colab Blackwell
Time: ~3-5h
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp058\notebook\nb_train.ipynb")


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# exp058 NB2 — Combined Training (Colab Blackwell)

**Path**: OverfitOracle 1-stage combined training

**Concept**:
- BC2026 train_audio (clip-level hard label) ★ + ★ XC audio (frame-level pseudo from NB1)
- Concat dataset、両 source を 1 epoch 内で混在
- Loss: BCE clip-level on BC2026 + BCE frame-level (weight 0.5) on XC pseudo
- 学生: effv2_b0 + SED head

**Inputs**:
- `maekeso/birdclef2026-exp058-xc-pseudo` (NB1 output、xc_pseudo.npz + xc_pseudo_index.csv)
- birdclef-2026 (train_audio + train.csv + taxonomy.csv)

**Config**:
- Backbone: tf_efficientnetv2_b0.in1k (6M params)
- SED head: GeMFreqPool + dense + att/cla (matches exp020 R2 structure)
- Epoch: 12
- Batch: 128
- LR backbone: 8e-4 (pretrained timm)、head: 8e-4 (scratch SED head)
- bf16 AMP
- SpecAugment 30/60、no MixUp (initial、simplicity)
- XC pseudo weight: 0.5
- Per-frame interpolation: teacher T → student T via F.interpolate

**Output**:
- `maekeso/birdclef2026-exp058-effv2b0-combined` (ckpt_best_ns22.pth)

**Expected**:
- val_macro: 0.91-0.94 (BC2026 random val)
- Standalone LB: 0.85-0.92 (predicted by OverfitOracle path)
"""),

        code_cell("install", """# Cell 1: install + drive mount
!pip install -q timm==1.0.11 soundfile librosa kaggle 2>&1 | tail -1

from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp058")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Drive: {DRIVE_ROOT}")
"""),

        code_cell("dl_data", """# Cell 2: Download exp058 pseudo + BC2026 competition data
import os, json, time, shutil
from pathlib import Path

# Kaggle auth
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
print("  ✓ Auth OK")

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)
t0 = time.time()

# 1. exp058 pseudo
PSEUDO_DIR = LOCAL_DATA / "exp058_pseudo"
if not PSEUDO_DIR.exists() or not any(PSEUDO_DIR.iterdir()):
    PSEUDO_DIR.mkdir(exist_ok=True)
    print("DL exp058 pseudo...")
    api.dataset_download_files("maekeso/birdclef2026-exp058-xc-pseudo",
                                path=str(PSEUDO_DIR), unzip=True, quiet=False)
print(f"  pseudo files: {sorted([f.name for f in PSEUDO_DIR.iterdir()])}")

# 2. XC audio (for combined training; need real audio files)
XC_DATASETS = [
    "maekeso/birdclef2026-xc-api-dl-part1",
    "maekeso/birdclef2026-xc-api-dl-part2",
    "maekeso/birdclef2026-xc-api-dl-part3",
]
for ds in XC_DATASETS:
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

# 3. BC2026 competition (train_audio + train.csv + taxonomy.csv)
BC_DIR = LOCAL_DATA / "birdclef-2026"
if not BC_DIR.exists() or not (BC_DIR / "train.csv").exists():
    BC_DIR.mkdir(exist_ok=True)
    print("DL BC2026...")
    api.competition_download_files("birdclef-2026", path=str(BC_DIR), quiet=False)
    import zipfile
    zip_path = BC_DIR / "birdclef-2026.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf: zf.extractall(BC_DIR)
        zip_path.unlink()
    print(f"  ✓ BC2026 extracted")

print(f"\\nDL total: {(time.time()-t0)/60:.1f} min")
"""),

        code_cell("setup", """# Cell 3: imports + GPU info
import os, sys, json, time, math, gc, re, shutil, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""),

        code_cell("config", """# Cell 4: training config
CFG = dict(
    backbone="tf_efficientnetv2_b0.in1k",
    in_chans=1,                       # mono mel (exp020 R2 matches)
    sr=32000, chunk_sec=5,
    n_mels=256, n_fft=2048, hop_length=512,
    fmin=20, fmax=16000,
    epochs=15,
    batch_size=192,
    lr=8e-4,
    weight_decay=1e-3,
    warmup_steps=500,
    label_smoothing=0.1,
    grad_clip=2.0,
    num_workers=8,
    val_split=0.1,
    val_seed=42,
    xc_pseudo_weight=0.5,             # XC frame-level loss weight
    drop_path_rate=0.1,
    hidden_dim=512,
    use_mixup=False,                  # initial: skip mixup (cross-source mixing complex)
    spec_aug_freq_mask=30,
    spec_aug_time_mask=60,
)
for k, v in CFG.items(): print(f"  {k}: {v}")
CHUNK_SAMPLES = CFG["sr"] * CFG["chunk_sec"]
"""),

        code_cell("species_taxo", """# Cell 5: BC2026 taxonomy + species mapping
BC_DIR = Path("/content/data/birdclef-2026")
taxo = pd.read_csv(BC_DIR / "taxonomy.csv")
PRIMARY_LABELS = taxo["primary_label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}
SCI_TO_LABEL = {str(s).lower(): l for s, l in zip(taxo["scientific_name"], taxo["primary_label"])}
LABEL_TO_CLASS = dict(zip(taxo["primary_label"], taxo["class_name"]))
print(f"BC2026: {N_CLASSES} species")
print(f"  by class: {taxo['class_name'].value_counts().to_dict()}")
"""),

        code_cell("load_pseudo", """# Cell 6: Load XC pseudo (.npz + index.csv)
PSEUDO_DIR = Path("/content/data/exp058_pseudo")

# Find pseudo files (may be in subdir)
npz_path = next(PSEUDO_DIR.rglob("xc_pseudo.npz"), None)
idx_path = next(PSEUDO_DIR.rglob("xc_pseudo_index.csv"), None)
filtered_meta_path = next(PSEUDO_DIR.rglob("filtered_metadata.csv"), None)
assert npz_path is not None, "xc_pseudo.npz not found"
assert idx_path is not None, "xc_pseudo_index.csv not found"
print(f"NPZ: {npz_path}")
print(f"INDEX: {idx_path}")
print(f"FILTERED META: {filtered_meta_path}")

print("\\nLoading xc_pseudo.npz...")
t0 = time.time()
data = np.load(npz_path)
xc_pseudo_arr = data["pseudo"]  # (N, T_teacher, 234) fp16
print(f"  shape: {xc_pseudo_arr.shape}, dtype: {xc_pseudo_arr.dtype}")
print(f"  size: {xc_pseudo_arr.nbytes/1e9:.2f} GB")
print(f"  load: {(time.time()-t0):.1f} s")

xc_pseudo_index = pd.read_csv(idx_path)
print(f"\\nxc_pseudo_index: {len(xc_pseudo_index)} rows")
print(f"  columns: {xc_pseudo_index.columns.tolist()}")

T_TEACHER = xc_pseudo_arr.shape[1]
print(f"\\nT_TEACHER (teacher frame dim): {T_TEACHER}")
"""),

        code_cell("bc_meta", """# Cell 7: BC2026 train_audio metadata + random val split
train_csv = pd.read_csv(BC_DIR / "train.csv")
print(f"BC2026 train.csv: {len(train_csv)} rows")

train_audio_dir = BC_DIR / "train_audio"
bc_records = []
for _, r in train_csv.iterrows():
    pl = str(r["primary_label"])
    if pl not in label_to_idx:
        continue
    fp = train_audio_dir / str(r["filename"])
    if fp.exists():
        bc_records.append((str(fp), pl))
print(f"BC2026 records found: {len(bc_records)}")

bc_df = pd.DataFrame(bc_records, columns=["filepath", "primary_label"])
bc_df["target_idx"] = bc_df["primary_label"].map(label_to_idx)

# Stratified val split (10% per species, min 2 for AUC)
np.random.seed(CFG["val_seed"])
bc_df = bc_df.sample(frac=1, random_state=CFG["val_seed"]).reset_index(drop=True)

val_mask = np.zeros(len(bc_df), dtype=bool)
for sp in bc_df["primary_label"].unique():
    idx = bc_df.index[bc_df["primary_label"] == sp].tolist()
    n_val = max(2, int(len(idx) * CFG["val_split"]))
    pick = np.random.choice(idx, size=min(n_val, len(idx)), replace=False)
    val_mask[pick] = True

bc_train_df = bc_df[~val_mask].reset_index(drop=True)
bc_val_df = bc_df[val_mask].reset_index(drop=True)
print(f"BC train: {len(bc_train_df)} ({bc_train_df['primary_label'].nunique()} species)")
print(f"BC val:   {len(bc_val_df)} ({bc_val_df['primary_label'].nunique()} species)")
"""),

        code_cell("datasets", """# Cell 8: Dataset definitions
class BC2026Dataset(Dataset):
    \"\"\"BC2026 focal: returns (wav, clip_target_idx, frame_pseudo=zeros, source_flag=0).\"\"\"
    def __init__(self, df, training=True):
        self.df = df.reset_index(drop=True)
        self.training = training

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row["filepath"]
        target_idx = int(row["target_idx"])
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)

        if len(wav) < CHUNK_SAMPLES:
            wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
        else:
            if self.training:
                start = np.random.randint(0, len(wav) - CHUNK_SAMPLES + 1)
            else:
                start = (len(wav) - CHUNK_SAMPLES) // 2
            wav = wav[start:start + CHUNK_SAMPLES]

        # source_flag=0 for BC2026
        return {
            "wav": torch.from_numpy(wav).float(),
            "clip_target_idx": target_idx,
            "frame_pseudo": torch.zeros(1, dtype=torch.float16),  # placeholder, masked at loss
            "source_flag": 0,
        }


class XCPseudoDataset(Dataset):
    \"\"\"XC chunks with frame-level pseudo: returns (wav, clip_target_idx=-1, frame_pseudo, source_flag=1).\"\"\"
    def __init__(self, xc_index_df, xc_pseudo_arr):
        self.idx_df = xc_index_df.reset_index(drop=True)
        self.pseudo = xc_pseudo_arr
        self.chunk_stride_sec = 5  # must match NB1

    def __len__(self): return len(self.idx_df)

    def __getitem__(self, idx):
        row = self.idx_df.iloc[idx]
        fp = row["filepath"]
        ci = int(row["chunk_idx"])
        chunk_global_idx = int(row["chunk_global_idx"])
        # Load audio chunk
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)

        start_sample = ci * CFG["sr"] * self.chunk_stride_sec
        end_sample = start_sample + CHUNK_SAMPLES
        if end_sample > len(wav):
            wav_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            avail = min(CHUNK_SAMPLES, max(0, len(wav) - start_sample))
            if avail > 0:
                wav_chunk[:avail] = wav[start_sample:start_sample + avail]
        else:
            wav_chunk = wav[start_sample:end_sample]

        # Frame-level pseudo (fp16)
        pseudo = self.pseudo[chunk_global_idx]  # (T_teacher, 234) fp16

        return {
            "wav": torch.from_numpy(wav_chunk).float(),
            "clip_target_idx": -1,  # not used for XC
            "frame_pseudo": torch.from_numpy(pseudo),  # (T_teacher, 234) fp16
            "source_flag": 1,
        }


# Build datasets
bc_train_ds = BC2026Dataset(bc_train_df, training=True)
bc_val_ds = BC2026Dataset(bc_val_df, training=False)
xc_ds = XCPseudoDataset(xc_pseudo_index, xc_pseudo_arr)

combined_train_ds = ConcatDataset([bc_train_ds, xc_ds])
print(f"BC train: {len(bc_train_ds)} | XC: {len(xc_ds)} | Combined: {len(combined_train_ds)}")
print(f"Val (BC only): {len(bc_val_ds)}")
"""),

        code_cell("model", """# Cell 9: Model — effv2_b0 + SED head (matches exp020 R2 SED structure)
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"], power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, wav):
        return self.db(self.mel_spec(wav))


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class BirdSEDModelEffv2(nn.Module):
    def __init__(self, backbone_name=CFG["backbone"], num_classes=N_CLASSES,
                 drop_path_rate=CFG["drop_path_rate"], hidden_dim=CFG["hidden_dim"],
                 in_chans=CFG["in_chans"]):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=in_chans,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = CHUNK_SAMPLES // CFG["hop_length"] + 1
            dummy = torch.randn(1, in_chans, CFG["n_mels"], n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)  # (B, T_student, C)
        return clip_logits


mel_extractor = MelSpecTransform().to(DEVICE)
model = BirdSEDModelEffv2().to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"  backbone: {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M")
print(f"  head: {(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.backbone.parameters()))/1e6:.2f}M")

# Determine student T_frames
with torch.no_grad():
    dummy_wav = torch.randn(1, CHUNK_SAMPLES, device=DEVICE)
    dummy_mel = mel_extractor(dummy_wav.unsqueeze(1))
    _, fw = model(dummy_mel, return_framewise=True)
    T_STUDENT = fw.shape[1]
print(f"\\nT_STUDENT (effv2_b0): {T_STUDENT}")
print(f"T_TEACHER (eca_nfnet_l0): {T_TEACHER}")
print(f"Interpolation: T_TEACHER {T_TEACHER} → T_STUDENT {T_STUDENT}")
"""),

        code_cell("aug", """# Cell 10: SpecAugment (no MixUp for simplicity)
def spec_augment(mel, freq_mask=CFG["spec_aug_freq_mask"], time_mask=CFG["spec_aug_time_mask"],
                 n_freq=2, n_time=2):
    B, _, F_, T = mel.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask + 1)
        f0 = np.random.randint(0, max(1, F_ - f))
        mel[:, :, f0:f0+f, :] = 0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask + 1)
        t0 = np.random.randint(0, max(1, T - t))
        mel[:, :, :, t0:t0+t] = 0
    return mel
"""),

        code_cell("dataloader", """# Cell 11: DataLoader + optimizer + scheduler
def collate_fn(batch):
    \"\"\"Combine BC2026 + XC samples into mixed batch.\"\"\"
    wavs = torch.stack([b["wav"] for b in batch])
    clip_targets = torch.tensor([b["clip_target_idx"] for b in batch], dtype=torch.long)
    source_flags = torch.tensor([b["source_flag"] for b in batch], dtype=torch.long)
    # Pseudo: pad to common shape (T_teacher, C)
    # All XC have same shape; BC have placeholder (1,)
    # We'll use a dict-of-tensors approach
    xc_mask = source_flags == 1
    if xc_mask.any():
        xc_pseudos = torch.stack([b["frame_pseudo"] for b in batch if b["source_flag"] == 1])  # (n_xc, T_teacher, C)
    else:
        xc_pseudos = torch.zeros(0, T_TEACHER, N_CLASSES, dtype=torch.float16)
    return {
        "wav": wavs,
        "clip_target": clip_targets,
        "source_flag": source_flags,
        "xc_pseudo": xc_pseudos,  # only for XC samples (compact tensor)
    }


train_loader = DataLoader(combined_train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
                          persistent_workers=True, collate_fn=collate_fn)
val_loader = DataLoader(bc_val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True,
                        persistent_workers=True,
                        collate_fn=lambda b: {
                            "wav": torch.stack([x["wav"] for x in b]),
                            "clip_target": torch.tensor([x["clip_target_idx"] for x in b], dtype=torch.long),
                        })

n_steps = len(train_loader) * CFG["epochs"]
print(f"Steps/epoch: {len(train_loader)}, total: {n_steps}")

optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

def lr_lambda(step):
    if step < CFG["warmup_steps"]:
        return step / max(1, CFG["warmup_steps"])
    progress = (step - CFG["warmup_steps"]) / max(1, n_steps - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
"""),

        code_cell("eval_fn", """# Cell 12: Validation eval (BC2026 val 10%)
@torch.no_grad()
def evaluate():
    model.eval()
    all_logits = []
    all_targets = []
    for batch in val_loader:
        wav = batch["wav"].to(DEVICE, non_blocking=True).unsqueeze(1)  # (B, 1, T)
        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            # normalize per-sample
            mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
            logit = model(mel)  # clip_logits (B, C)
        all_logits.append(logit.float().cpu())
        all_targets.append(batch["clip_target"])
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
    taxo_df = taxo.copy(); taxo_df["idx"] = taxo_df["primary_label"].map(label_to_idx)
    for cls in taxo_df["class_name"].unique():
        idx_list = taxo_df[taxo_df["class_name"] == cls]["idx"].tolist()
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

        code_cell("train_loop", """# Cell 13: Combined training loop (BC clip-level BCE + XC frame-level BCE)
best_val = 0.0
best_path = DRIVE_ROOT / "effv2b0_combined_best.pth"
log_path = DRIVE_ROOT / "effv2b0_combined_train.log"

step = 0
total_t0 = time.time()

for ep in range(CFG["epochs"]):
    t0 = time.time()
    model.train()
    losses = []
    losses_bc = []
    losses_xc = []
    n_bc_total = 0
    n_xc_total = 0

    for batch_i, batch in enumerate(train_loader):
        wav = batch["wav"].to(DEVICE, non_blocking=True).unsqueeze(1)  # (B, 1, T)
        clip_target = batch["clip_target"].to(DEVICE)
        source_flag = batch["source_flag"].to(DEVICE)
        xc_pseudo = batch["xc_pseudo"].to(DEVICE, non_blocking=True).float()  # (n_xc, T_teacher, C)

        bc_mask = source_flag == 0
        xc_mask = source_flag == 1
        n_bc = int(bc_mask.sum())
        n_xc = int(xc_mask.sum())
        n_bc_total += n_bc
        n_xc_total += n_xc

        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
            mel = spec_augment(mel)
            clip_logits, frame_logits = model(mel, return_framewise=True)
            # clip_logits: (B, C)
            # frame_logits: (B, T_student, C)

            loss_total = 0.0
            loss_bc_val = torch.tensor(0.0, device=DEVICE)
            loss_xc_val = torch.tensor(0.0, device=DEVICE)

            # --- BC2026 clip-level BCE ---
            if n_bc > 0:
                bc_clip_logits = clip_logits[bc_mask]
                bc_target_idx = clip_target[bc_mask]
                tgt_onehot = F.one_hot(bc_target_idx, N_CLASSES).float()
                ls = CFG["label_smoothing"]
                tgt_smooth = tgt_onehot * (1 - ls) + ls / N_CLASSES
                loss_bc_val = F.binary_cross_entropy_with_logits(bc_clip_logits, tgt_smooth)
                loss_total = loss_total + loss_bc_val

            # --- XC frame-level BCE (with interpolation to student T) ---
            if n_xc > 0:
                xc_frame_logits = frame_logits[xc_mask]  # (n_xc, T_student, C)
                # interpolate teacher pseudo (T_teacher → T_student)
                xc_pseudo_perm = xc_pseudo.permute(0, 2, 1)  # (n_xc, C, T_teacher)
                xc_pseudo_interp = F.interpolate(xc_pseudo_perm, size=T_STUDENT,
                                                  mode="linear", align_corners=False)
                xc_pseudo_interp = xc_pseudo_interp.permute(0, 2, 1)  # (n_xc, T_student, C)
                # clamp to valid prob range and compute BCE on probs (teacher already sigmoid'd)
                xc_pseudo_interp = xc_pseudo_interp.clamp(min=1e-6, max=1.0 - 1e-6)
                loss_xc_val = F.binary_cross_entropy_with_logits(xc_frame_logits, xc_pseudo_interp)
                loss_total = loss_total + CFG["xc_pseudo_weight"] * loss_xc_val

            # avoid divide-by-zero on rare all-one-source batch
            loss = loss_total / max(1.0, float((n_bc > 0) + (n_xc > 0)) * 0.5)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if n_bc > 0: losses_bc.append(float(loss_bc_val.item()))
        if n_xc > 0: losses_xc.append(float(loss_xc_val.item()))
        step += 1
        if batch_i % 100 == 0:
            print(f"  [ep{ep+1} step {batch_i}/{len(train_loader)}] "
                  f"loss={loss.item():.4f} "
                  f"loss_bc={loss_bc_val.item():.4f} loss_xc={loss_xc_val.item():.4f} "
                  f"n_bc={n_bc} n_xc={n_xc} lr={scheduler.get_last_lr()[0]:.2e}")

    avg_loss = float(np.mean(losses))
    avg_loss_bc = float(np.mean(losses_bc)) if losses_bc else 0.0
    avg_loss_xc = float(np.mean(losses_xc)) if losses_xc else 0.0
    val_macro, val_ns22, taxon_aucs, cstat = evaluate()
    ep_elapsed = time.time() - t0
    total_elapsed = time.time() - total_t0
    cur_lr = scheduler.get_last_lr()[0]

    is_best = val_ns22 > best_val
    best_mark = "BEST" if is_best else ""

    log_l1 = (f"=== Ep {ep+1}/{CFG['epochs']}: loss={avg_loss:.4f} "
              f"(bc={avg_loss_bc:.4f} xc={avg_loss_xc:.4f}) "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} {best_mark} "
              f"lr={cur_lr:.2e} ({ep_elapsed/60:.1f}min, total {total_elapsed/60:.1f}min) "
              f"n_bc={n_bc_total} n_xc={n_xc_total} ===")
    log_l2 = "    taxon: " + " ".join(f"{k}={v:.3f}" if v == v else f"{k}=nan"
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
                    "ep": ep+1, "cfg": CFG}, best_path)
        print(f"    BEST saved val_ns22={val_ns22:.4f}")

print(f"\\nDone. Best val_ns22: {best_val:.4f}")
print(f"Best ckpt: {best_path}")
"""),

        code_cell("upload_kaggle", """# Cell 14: Upload to Kaggle Dataset
import os, json, shutil
from pathlib import Path

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
SLUG = "birdclef2026-exp058-effv2b0-combined"
TITLE = "BirdCLEF2026 exp058 effv2b0 combined"

UPLOAD_DIR = Path("/content/upload_exp058_train")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(best_path, UPLOAD_DIR / "effv2b0_combined_best.pth")
shutil.copy(log_path, UPLOAD_DIR / "effv2b0_combined_train.log")

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset created")
except Exception as e:
    print(f"create_new err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="exp058 combined training (effv2_b0)",
                                    dir_mode="zip", quiet=False)
        print("OK version created")
    except Exception as e2:
        print(f"create_version err: {str(e2)[:200]}")
print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),

        code_cell("summary", """# Cell 15: summary
print(f"=== exp058 NB2 complete ===")
print(f"Best val_ns22: {best_val:.4f}")
print(f"Total epochs: {CFG['epochs']}")
print(f"Dataset: maekeso/{SLUG}")
print(f"\\nNext: NB3 = Kaggle CPU inference + submit")
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
    print(f"    {c['cell_type']:8s} id={c.get('id'):18s} L={n}")
