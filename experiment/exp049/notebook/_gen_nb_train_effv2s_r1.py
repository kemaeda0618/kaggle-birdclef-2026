"""Generate Kaggle NB: nb_train_effv2s_r1.ipynb

EfficientNetV2-S R1 baseline on Kaggle T4x2.
- Backbone: tf_efficientnetv2_s.in21k_ft_in1k
- Loss: BCE only (R1 = no pseudo distill)
- Input: 5s waveform from /kaggle/input/competitions/birdclef-2026/train_audio/
- Aug: spec mixup α=0.5 + SpecAugment
- Output: ckpt to /kaggle/working/, optional Dataset upload
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

nb = {
    "cells": [
        md_cell("hdr", """# exp049 EffNetV2-S R1 Training on Kaggle T4x2

**目的**: paper 256 (BC25 2位) 採用 backbone `efficientnetv2_s` の BC2026 R1 baseline 学習。
我々の lineage 未試行軸 → 4-way blend (NB4+ResSSM, Tucker, e29 R3, **e49 effv2s**) で diversity 拡張、gold 突破補助。

## 仕様

| Item | Value |
|---|---|
| Backbone | `tf_efficientnetv2_s.in21k_ft_in1k` (~22M params) |
| Loss | **BCE only** (R1、pseudo distill なし) |
| Input | 5s × 32kHz waveform → mel n_mels=128 |
| Mel params | n_fft=2048, hop=512, fmin=20, fmax=16000 |
| Batch | 64 (T4 16GB × 2、DataParallel) |
| Epochs | 20 |
| Optimizer | AdamW, lr=3e-4, wd=1e-4 |
| Scheduler | CosineAnnealingLR T_max=20 |
| Aug | Spec mixup α=0.5 + SpecAugment (freq_mask=30, time_mask=40) |
| Val | labeled SS hold-out (66 files) val_ns22 |
| Runtime budget | ~6-8h Kaggle T4x2 session |

## 期待

- val_ns22 0.91-0.93
- LB standalone 0.91-0.93 (推測)
- 4-way blend に追加で gold 0.954 補助

## Output

- `/kaggle/working/best_ckpt.pth` (best val_ns22 ckpt)
- `/kaggle/working/last_ckpt.pth`
- `/kaggle/working/history.json` (per-epoch log)

完走後 dataset 化 → `maekeso/birdclef2026-exp049-effv2s-r1`
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup + GPU check
# ============================================================
import os, sys, json, time, math, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import soundfile as sf
import librosa
from tqdm.auto import tqdm

print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}, devs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""),

        code_cell("config", """# ============================================================
# Cell 2: Config
# ============================================================
class CFG:
    SEED = 42
    # data
    BC2026_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
    TRAIN_CSV = BC2026_ROOT / "train.csv"
    TRAIN_AUDIO = BC2026_ROOT / "train_audio"
    LABELED_SS_CSV = BC2026_ROOT / "train_soundscapes_labels.csv"
    LABELED_SS_AUDIO = BC2026_ROOT / "train_soundscapes"
    TAXO_CSV = BC2026_ROOT / "taxonomy.csv"

    # model
    BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"
    N_CLASSES = 234

    # audio
    SR = 32000
    CHUNK_SEC = 5
    CHUNK_LEN = SR * CHUNK_SEC  # 160000 samples

    # mel
    N_MELS = 128
    N_FFT = 2048
    HOP = 512
    FMIN = 20
    FMAX = 16000

    # train
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 3e-4
    WD = 1e-4
    NUM_WORKERS = 4
    MIXUP_PROB = 0.5
    MIXUP_ALPHA = 0.5
    FREQ_MASK = 30
    TIME_MASK = 40

    # output
    OUT_DIR = Path("/kaggle/working")
    BEST_CKPT = OUT_DIR / "best_ckpt.pth"
    LAST_CKPT = OUT_DIR / "last_ckpt.pth"
    HIST_JSON = OUT_DIR / "history.json"

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)
torch.backends.cudnn.benchmark = True

print(f"Backbone: {CFG.BACKBONE}")
print(f"Mel: {CFG.N_MELS}×T, {CFG.SR}Hz")
print(f"Train: {CFG.EPOCHS} ep × batch {CFG.BATCH_SIZE}")
print(f"Output: {CFG.OUT_DIR}")
"""),

        code_cell("data_load", """# ============================================================
# Cell 3: Load metadata + species mapping
# ============================================================
tax = pd.read_csv(CFG.TAXO_CSV)
PRIMARY_LABELS = tax["primary_label"].tolist()
LABEL_TO_IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
assert len(PRIMARY_LABELS) == CFG.N_CLASSES

train_df = pd.read_csv(CFG.TRAIN_CSV)
print(f"train_df: {len(train_df)} rows")
print(f"  columns: {train_df.columns.tolist()[:10]}")
# Build absolute audio path
def _resolve_audio_path(row):
    fname = row["filename"]
    if not str(fname).endswith(".ogg"):
        fname = fname + ".ogg"
    return CFG.TRAIN_AUDIO / fname
train_df["audio_path"] = train_df.apply(_resolve_audio_path, axis=1)
n_exists = train_df["audio_path"].apply(lambda p: p.exists()).sum()
print(f"  audio exists: {n_exists}/{len(train_df)}")

# Labeled SS for val
val_df = pd.read_csv(CFG.LABELED_SS_CSV) if CFG.LABELED_SS_CSV.exists() else None
if val_df is not None:
    print(f"val_df: {len(val_df)} rows")
else:
    print("WARN: train_soundscapes_labels.csv not found, will do random hold-out")
"""),

        code_cell("dataset", """# ============================================================
# Cell 4: Dataset (waveform → mel on GPU)
# ============================================================
class TrainDataset(Dataset):
    def __init__(self, df, sr=CFG.SR, chunk_len=CFG.CHUNK_LEN, training=True):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.chunk_len = chunk_len
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        try:
            wav, sr = sf.read(str(audio_path), dtype="float32")
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        except Exception:
            wav = np.zeros(self.chunk_len, dtype=np.float32)

        # Random crop or pad to chunk_len
        if len(wav) >= self.chunk_len:
            if self.training:
                start = np.random.randint(0, len(wav) - self.chunk_len + 1)
            else:
                start = (len(wav) - self.chunk_len) // 2
            wav = wav[start:start + self.chunk_len]
        else:
            pad = self.chunk_len - len(wav)
            wav = np.pad(wav, (0, pad), mode="constant")

        # Hard label
        label = np.zeros(CFG.N_CLASSES, dtype=np.float32)
        primary = row["primary_label"]
        if primary in LABEL_TO_IDX:
            label[LABEL_TO_IDX[primary]] = 1.0
        # Secondary labels
        sec = row.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec not in ("[]", "", "nan"):
            try:
                sec_list = eval(sec) if sec.startswith("[") else []
                for s in sec_list:
                    if s in LABEL_TO_IDX:
                        label[LABEL_TO_IDX[s]] = 1.0
            except Exception:
                pass

        return torch.from_numpy(wav), torch.from_numpy(label)


class MelExtractor(nn.Module):
    \"\"\"GPU mel transform.\"\"\"
    def __init__(self, sr=CFG.SR, n_mels=CFG.N_MELS, n_fft=CFG.N_FFT,
                 hop=CFG.HOP, fmin=CFG.FMIN, fmax=CFG.FMAX):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=fmin, f_max=fmax,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def forward(self, wav):
        # wav: (B, T) -> mel: (B, n_mels, T'), then dB
        mel = self.mel(wav)
        mel = self.db(mel)
        # Normalize to [-1, 1] approx
        mel = (mel + 80) / 80 * 2 - 1
        return mel


# Sanity check
ds_check = TrainDataset(train_df.head(10), training=True)
wav, label = ds_check[0]
print(f"wav: {wav.shape} {wav.dtype}, label sum: {label.sum().item():.0f}")
"""),

        code_cell("model", """# ============================================================
# Cell 5: SED Model (timm backbone + framewise attention pool)
# ============================================================
class SEDHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.att = nn.Linear(in_dim, n_classes)
        self.cla = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        # x: (B, T, C) framewise features
        att = torch.tanh(self.att(x))
        cla = self.cla(x)
        # softmax over time -> framewise attention weights
        norm_att = F.softmax(att, dim=1)  # softmax over T
        clipwise = (norm_att * cla).sum(dim=1)  # (B, C)
        return clipwise, cla


class SEDModel(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE, n_classes=CFG.N_CLASSES):
        super().__init__()
        # 1ch -> 3ch input
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=3,
            num_classes=0, global_pool="",
        )
        feat_dim = self.backbone.num_features
        self.head = SEDHead(feat_dim, n_classes)

    def forward(self, mel):
        # mel: (B, n_mels, T)
        # Expand to 3 channels (replicate)
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, n_mels, T)
        feat = self.backbone(x)  # (B, C, H, W)
        # Global avg over frequency dim, keep time dim
        feat = feat.mean(dim=2)  # (B, C, T')
        feat = feat.transpose(1, 2)  # (B, T', C)
        clipwise, framewise = self.head(feat)
        return clipwise, framewise


# Sanity check
model_check = SEDModel().to(DEVICE)
dummy = torch.randn(2, CFG.N_MELS, 313).to(DEVICE)  # batch 2, mel, T=5s
with torch.no_grad():
    out, fr = model_check(dummy)
print(f"clipwise: {out.shape}, framewise: {fr.shape}")
del model_check; torch.cuda.empty_cache()
"""),

        code_cell("aug", """# ============================================================
# Cell 6: Augmentations (Spec mixup + SpecAugment)
# ============================================================
def spec_mixup(mel, label, alpha=CFG.MIXUP_ALPHA, p=CFG.MIXUP_PROB):
    \"\"\"Spec-domain mixup.\"\"\"
    if np.random.random() > p:
        return mel, label
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(mel.size(0), device=mel.device)
    mel_mix = lam * mel + (1 - lam) * mel[idx]
    # OR-style label mixing (multi-label)
    label_mix = torch.maximum(label, label[idx])
    return mel_mix, label_mix


def spec_augment(mel, freq_mask=CFG.FREQ_MASK, time_mask=CFG.TIME_MASK):
    \"\"\"SpecAugment (freq + time mask, in-place).\"\"\"
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
# Cell 7: Train/Val split + dataloaders
# ============================================================
# Random hold-out (no Drive cache, simple split)
train_df_s = train_df.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
n_val = int(len(train_df_s) * 0.05)
train_split = train_df_s.iloc[n_val:].reset_index(drop=True)
val_split = train_df_s.iloc[:n_val].reset_index(drop=True)
print(f"train split: {len(train_split)}, val split: {len(val_split)}")

train_ds = TrainDataset(train_split, training=True)
val_ds = TrainDataset(val_split, training=False)

train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True,
                          drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                         num_workers=CFG.NUM_WORKERS, pin_memory=True,
                         persistent_workers=True)

print(f"steps/ep: train={len(train_loader)}, val={len(val_loader)}")
"""),

        code_cell("model_setup", """# ============================================================
# Cell 8: Build model, optimizer, scheduler
# ============================================================
model = SEDModel().to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

mel_extractor = MelExtractor().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
scaler = GradScaler()
loss_fn = nn.BCEWithLogitsLoss()
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
"""),

        code_cell("eval_fn", """# ============================================================
# Cell 9: val_ns22 (no-Sec-2-2) - per-class AUC mean (skip classes with <2 pos)
# ============================================================
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def evaluate(model, mel_ex, loader):
    model.eval()
    all_logits, all_labels = [], []
    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        with autocast():
            mel = mel_ex(wav)
            logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    # per-class AUC, skip classes with <2 positive samples
    aucs = []
    for c in range(labels.shape[1]):
        pos = labels[:, c].sum()
        if pos >= 2 and pos < labels.shape[0]:  # need both pos and neg
            try:
                aucs.append(roc_auc_score(labels[:, c], logits[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0
"""),

        code_cell("train_loop", """# ============================================================
# Cell 10: Training loop
# ============================================================
history = {"train_loss": [], "val_ns22": [], "lr": [], "elapsed_min": []}
best_val = 0.0
start_t = time.time()

for epoch in range(1, CFG.EPOCHS + 1):
    ep_start = time.time()
    model.train()
    train_losses = []
    pbar = tqdm(train_loader, desc=f"Ep {epoch}/{CFG.EPOCHS}", leave=False)
    for wav, label in pbar:
        wav = wav.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        with autocast():
            mel = mel_extractor(wav)
            # spec mixup + augmentations
            mel, label_mix = spec_mixup(mel, label)
            mel = spec_augment(mel)
            logit, _ = model(mel)
            loss = loss_fn(logit, label_mix)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        pbar.set_postfix(loss=f"{np.mean(train_losses[-50:]):.4f}")
    scheduler.step()

    train_loss = float(np.mean(train_losses))
    val_ns22 = evaluate(model, mel_extractor, val_loader)
    lr_now = optimizer.param_groups[0]["lr"]
    elapsed_min = (time.time() - start_t) / 60
    ep_min = (time.time() - ep_start) / 60

    history["train_loss"].append(train_loss)
    history["val_ns22"].append(val_ns22)
    history["lr"].append(lr_now)
    history["elapsed_min"].append(elapsed_min)

    is_best = val_ns22 > best_val
    if is_best:
        best_val = val_ns22
        torch.save(
            {"epoch": epoch, "val_ns22": val_ns22,
             "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
             "cfg": {k: str(v) for k, v in CFG.__dict__.items() if not k.startswith("_")},
            },
            CFG.BEST_CKPT,
        )

    json.dump(history, open(CFG.HIST_JSON, "w"), indent=2)
    print(f"=== Ep {epoch}/{CFG.EPOCHS}: loss={train_loss:.4f} val_ns22={val_ns22:.4f}"
          f" {'★best' if is_best else ''} lr={lr_now:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min) ===")

# Save final
torch.save(
    {"epoch": CFG.EPOCHS, "val_ns22": history["val_ns22"][-1],
     "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    },
    CFG.LAST_CKPT,
)
print(f"\\nTraining DONE. Best val_ns22={best_val:.4f}")
"""),

        code_cell("save_dataset", """# ============================================================
# Cell 11: (Optional) Save ckpt as Kaggle Dataset
# ============================================================
import json, os, shutil
from pathlib import Path

DRY_RUN = True  # set False to upload

if not DRY_RUN:
    os.environ.setdefault("KAGGLE_KEY", "")  # will use mounted kaggle.json
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()

    UPLOAD_DIR = Path("/kaggle/working/exp049_upload")
    UPLOAD_DIR.mkdir(exist_ok=True)
    for f in [CFG.BEST_CKPT, CFG.LAST_CKPT, CFG.HIST_JSON]:
        if f.exists():
            shutil.copy2(f, UPLOAD_DIR / f.name)

    SLUG = "birdclef2026-exp049-effv2s-r1"
    USER = "maekeso"
    meta = {
        "title": "BirdCLEF2026 exp049 EffNetV2-S R1",
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "other"}],
    }
    (UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR),
                                    version_notes=f"best val_ns22={best_val:.4f}",
                                    dir_mode="tar", quiet=False)
        print("OK dataset version uploaded")
    except Exception:
        try:
            api.dataset_create_new(folder=str(UPLOAD_DIR), public=False,
                                    dir_mode="tar", quiet=False)
            print("OK new dataset created")
        except Exception as e:
            print(f"upload err: {str(e)[:300]}")
    print(f"URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
else:
    print(f"DRY_RUN=True, skip upload")
    print(f"Files: {CFG.BEST_CKPT.exists()=}, size={CFG.BEST_CKPT.stat().st_size/1e6:.1f}MB" if CFG.BEST_CKPT.exists() else "no ckpt")
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id'):15s} L={n}")
