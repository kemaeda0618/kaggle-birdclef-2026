"""Generate exp065 combined NB: SC pseudo gen + R2 training (single Colab Blackwell NB).

Step 1 ablation: SC pseudo 単独効果 (Perch distill なし、MixUp なし)
  - exp058 R1 config そのまま
  - + train_soundscapes pseudo 追加だけ

Flow:
  1. DL data
  2. Teacher (exp020 R2 5-fold) で SC pseudo gen
  3. R2 training: BC2026 hard + XC pseudo (既存) + SC pseudo (新)
  4. R1 ckpt warm-start
  5. Output: maekeso/birdclef2026-exp065-effv2b0-r2-sc

Time: ~4-6h Colab Blackwell
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp065\notebook\nb_train_r2_sc.ipynb")


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
        md_cell("hdr", """# exp065 — R2 SC Pseudo Ablation (Colab Blackwell, single NB)

**Hypothesis**: exp058 R1 失敗の主因が「train_soundscapes 未利用」か検証

**Step 1 ablation**:
  exp058 R1 config そのまま + **SC pseudo 追加だけ**
  - Perch distill 追加なし (NO)
  - MixUp 追加なし (NO)
  - SC pseudo data だけ追加 (YES)

**Pipeline**:
  1. SC pseudo gen: exp020 R2 5-fold teacher で train_soundscapes (~10k files × 6 chunks)
  2. R2 training: BC2026 hard + XC pseudo (既存) + SC pseudo (新)
  3. R1 ckpt warm-start
  4. 15 epochs

**Output**: `maekeso/birdclef2026-exp065-effv2b0-r2-sc`

**Expected**:
  SC が真因なら: LB 0.85-0.90 (R1 0.76 から +0.09-0.14)
  SC で限定なら: LB 0.78-0.83 (Perch も必要)
"""),

        code_cell("install", """!pip install -q timm==1.0.11 soundfile librosa kaggle 2>&1 | tail -1
from google.colab import drive
drive.mount('/content/drive')
import os
from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp065")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Drive: {DRIVE_ROOT}")
"""),

        code_cell("dl_data", """import os, json, time, shutil
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_DIR.mkdir(exist_ok=True)
if not (KAGGLE_DIR / "kaggle.json").exists():
    src_kg = Path("/content/drive/MyDrive/kaggle/kaggle.json")
    if src_kg.exists():
        shutil.copy(src_kg, KAGGLE_DIR / "kaggle.json")
        os.chmod(KAGGLE_DIR / "kaggle.json", 0o600)
_kgat = json.loads((KAGGLE_DIR/"kaggle.json").read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print("auth OK")

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)
t0 = time.time()

DATASETS = [
    "maekeso/birdclef2026-exp058-xc-pseudo",          # XC pseudo (既存)
    "maekeso/birdclef2026-xc-api-dl-part1",
    "maekeso/birdclef2026-xc-api-dl-part2",
    "maekeso/birdclef2026-xc-api-dl-part3",
    "maekeso/birdclef2026-exp020-weights-5fold",      # teacher for SC pseudo
    "maekeso/birdclef2026-exp058-effv2b0-combined",   # R1 ckpt for warm-start
]
for ds in DATASETS:
    name = ds.split("/")[-1]
    dst = LOCAL_DATA / name
    if dst.exists() and any(dst.iterdir()):
        n = sum(1 for _ in dst.rglob("*") if _.is_file())
        if n > 0: print(f"  ✓ {name} ({n} files)"); continue
    dst.mkdir(exist_ok=True)
    print(f"  DL {ds}...")
    api.dataset_download_files(ds, path=str(dst), unzip=True, quiet=False)

BC_DIR = LOCAL_DATA / "birdclef-2026"
if not (BC_DIR / "train.csv").exists():
    BC_DIR.mkdir(exist_ok=True)
    print("DL BC2026...")
    api.competition_download_files("birdclef-2026", path=str(BC_DIR), quiet=False)
    import zipfile
    zp = BC_DIR / "birdclef-2026.zip"
    if zp.exists():
        with zipfile.ZipFile(zp) as zf: zf.extractall(BC_DIR)
        zp.unlink()
print(f"DL: {(time.time()-t0)/60:.1f}min")
"""),

        code_cell("setup", """import sys, gc, re, math, random
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
print(f"Device: {DEVICE}, mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""),

        code_cell("config", """CFG = dict(
    backbone="tf_efficientnetv2_b0.in1k",
    in_chans=1,
    sr=32000, chunk_sec=5,
    n_mels=256, n_fft=2048, hop_length=512,
    fmin=20, fmax=16000,
    epochs=15,
    batch_size=192,
    lr=4e-4,                # ★ R1 lr 8e-4 の half (warm-start から再調整)
    weight_decay=1e-3,
    warmup_steps=300,       # ★ R1 500 → 300 (warm start なので shorter)
    label_smoothing=0.1,
    grad_clip=2.0,
    num_workers=8,
    val_split=0.1, val_seed=42,
    xc_pseudo_weight=0.5,
    sc_pseudo_weight=0.5,
    drop_path_rate=0.1,
    hidden_dim=512,
    use_mixup=False,
    spec_aug_freq_mask=30, spec_aug_time_mask=60,
    sc_max_chunks_per_file=6,  # match XC pattern, ~60k SC chunks (balanced)
)
for k, v in CFG.items(): print(f"  {k}: {v}")
CHUNK_SAMPLES = CFG["sr"] * CFG["chunk_sec"]
N_WINDOWS = 12
"""),

        code_cell("species", f"""__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {{l: i for i, l in enumerate(PRIMARY_LABELS)}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
print(f"BC2026: {{N_CLASSES}} species")
"""),

        code_cell("teacher_load", """# Teacher: exp020 R2 5-fold for SC pseudo gen
E20_DIR = next(Path("/content/data/birdclef2026-exp020-weights-5fold").rglob("r2_fold0_ckpt_best_ns22.pth")).parent
print(f"teacher dir: {E20_DIR}")

class _GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(float(p_init))); self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0); x = x.clamp(min=self.eps).pow(p)
        return x.mean(dim=2).pow(1.0 / p)

class _DistillHead(nn.Module):
    def __init__(self, bd, ed=1536):
        super().__init__(); self.proj = nn.Linear(bd, ed)
    def forward(self, fm): return self.proj(fm.mean(dim=[2,3]))

class _E20SED(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("eca_nfnet_l0", pretrained=False, in_chans=1, num_classes=0, global_pool="", drop_path_rate=0.1)
        with torch.no_grad():
            n_tf = CHUNK_SAMPLES // CFG["hop_length"] + 1
            dummy = torch.randn(1, 1, CFG["n_mels"], n_tf)
            self.backbone_dim = self.backbone(dummy).shape[1]
        self.gem_freq = _GeMFreq(3.0)
        self.dense = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.backbone_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.att = nn.Conv1d(512, N_CLASSES, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(512, N_CLASSES, kernel_size=1, bias=True)
        self.distill_head = _DistillHead(self.backbone_dim, 1536)
    def forward(self, x):
        h = self.backbone(x); h_cls = h.detach()
        h_cls = self.gem_freq(h_cls).permute(0, 2, 1)
        h_cls = self.dense(h_cls).permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        fw = self.cla(h_cls)
        clip = torch.sum(norm_att * fw, dim=2)
        return clip, fw.permute(0, 2, 1)

class _MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"], n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"], power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x): return self.db(self.mel(x))

teacher_mel_tf = _MelTF().to(DEVICE)
teacher_ckpts = sorted(E20_DIR.rglob("r2_fold*_ckpt_best_ns22.pth"))
teacher_models = []
for ck in teacher_ckpts:
    try: st = torch.load(str(ck), map_location="cpu", weights_only=False)
    except TypeError: st = torch.load(str(ck), map_location="cpu")
    m = _E20SED().to(DEVICE); m.load_state_dict(st["model_state"], strict=False); m.eval()
    teacher_models.append(m); del st; gc.collect()
print(f"Loaded {len(teacher_models)} teacher folds")
"""),

        code_cell("sc_pseudo_gen", """# Generate SC pseudo with teacher (5-fold avg)
from scipy.ndimage import gaussian_filter1d
SC_DIR = Path("/content/data/birdclef-2026/train_soundscapes")
sc_files = sorted(SC_DIR.glob("*.ogg"))
print(f"SC files: {len(sc_files)}")

# Sample SC chunks: random sc_max_chunks_per_file per file (deterministic seed)
SC_RNG = np.random.RandomState(42)
sc_entries = []  # (file_idx, chunk_idx)
for fi in range(len(sc_files)):
    chunks = SC_RNG.choice(N_WINDOWS, size=min(N_WINDOWS, CFG["sc_max_chunks_per_file"]), replace=False)
    for ci in sorted(chunks):
        sc_entries.append((fi, int(ci)))
print(f"SC entries (chunks to process): {len(sc_entries)}")

class _SCChunkDataset(Dataset):
    def __init__(self, files, entries):
        self.files = files; self.entries = entries
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        fi, ci = self.entries[idx]
        fp = self.files[fi]
        try:
            wav, sr = sf.read(str(fp), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(60 * CFG["sr"], dtype=np.float32)
        target_len = 60 * CFG["sr"]
        if len(wav) < target_len: wav = np.pad(wav, (0, target_len - len(wav)))
        else: wav = wav[:target_len]
        chunk = wav[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
        return torch.from_numpy(chunk).float(), fi, ci

sc_ds_gen = _SCChunkDataset(sc_files, sc_entries)
sc_loader_gen = DataLoader(sc_ds_gen, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# Determine T_frames
with torch.no_grad():
    w, _, _ = next(iter(sc_loader_gen))
    w = w.to(DEVICE).unsqueeze(1)
    m = teacher_mel_tf(w); m = (m - m.mean()) / (m.std() + 1e-6)
    _, fw = teacher_models[0](m)
    T_TEACHER = fw.shape[1]
print(f"T_TEACHER: {T_TEACHER}")

sc_pseudo_arr = np.zeros((len(sc_ds_gen), T_TEACHER, N_CLASSES), dtype=np.float16)
print(f"SC pseudo array size: {sc_pseudo_arr.nbytes/1e9:.2f}GB")

t0 = time.time()
chunk_g = 0
for bi, (wavs, fis, cis) in enumerate(sc_loader_gen):
    wavs = wavs.to(DEVICE, non_blocking=True).unsqueeze(1)
    with torch.no_grad():
        mel = teacher_mel_tf(wavs)
        mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
        accum = None
        for m in teacher_models:
            _, fw = m(mel)
            fw_prob = torch.sigmoid(fw)
            accum = fw_prob if accum is None else accum + fw_prob
        accum = (accum / len(teacher_models)).float().cpu().numpy().astype(np.float16)
    bsz = accum.shape[0]
    sc_pseudo_arr[chunk_g:chunk_g + bsz] = accum
    chunk_g += bsz
    if (bi + 1) % 50 == 0:
        el = (time.time() - t0) / 60
        eta = el / (bi + 1) * (len(sc_loader_gen) - bi - 1)
        print(f"  SC pseudo [{bi+1}/{len(sc_loader_gen)}] el={el:.1f}min eta={eta:.1f}min")

print(f"SC pseudo done: {sc_pseudo_arr.shape}, mean={sc_pseudo_arr.mean():.4f} in {(time.time()-t0)/60:.1f}min")

# Free teacher
del teacher_models, teacher_mel_tf
gc.collect()
torch.cuda.empty_cache()
"""),

        code_cell("xc_pseudo_load", """# Load existing XC pseudo
PSEUDO_DIR = Path("/content/data/birdclef2026-exp058-xc-pseudo")
xc_npz = next(PSEUDO_DIR.rglob("xc_pseudo.npz"))
xc_idx_csv = next(PSEUDO_DIR.rglob("xc_pseudo_index.csv"))
print(f"loading XC: {xc_npz}")
xc_pseudo_arr = np.load(xc_npz)["pseudo"]
xc_index = pd.read_csv(xc_idx_csv)
print(f"XC pseudo: {xc_pseudo_arr.shape}, index: {len(xc_index)} rows")
"""),

        code_cell("bc_meta", """# BC2026 train_audio metadata + val split
BC_DIR = Path("/content/data/birdclef-2026")
train_csv = pd.read_csv(BC_DIR / "train.csv")
print(f"train.csv: {len(train_csv)} rows")

train_audio_dir = BC_DIR / "train_audio"
bc_records = []
for _, r in train_csv.iterrows():
    pl = str(r["primary_label"])
    if pl not in label_to_idx: continue
    fp = train_audio_dir / str(r["filename"])
    if fp.exists(): bc_records.append((str(fp), pl))

bc_df = pd.DataFrame(bc_records, columns=["filepath", "primary_label"])
bc_df["target_idx"] = bc_df["primary_label"].map(label_to_idx)

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
print(f"BC train: {len(bc_train_df)} | val: {len(bc_val_df)}")
"""),

        code_cell("datasets", """class BC2026Dataset(Dataset):
    def __init__(self, df, training=True):
        self.df = df.reset_index(drop=True); self.training = training
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; fp = row["filepath"]; target_idx = int(row["target_idx"])
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        if len(wav) < CHUNK_SAMPLES: wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
        else:
            start = np.random.randint(0, len(wav) - CHUNK_SAMPLES + 1) if self.training else (len(wav) - CHUNK_SAMPLES) // 2
            wav = wav[start:start + CHUNK_SAMPLES]
        return {
            "wav": torch.from_numpy(wav).float(),
            "clip_target_idx": target_idx,
            "frame_pseudo": torch.zeros(1, dtype=torch.float16),
            "source_flag": 0,  # BC
        }


class XCPseudoDataset(Dataset):
    def __init__(self, idx_df, pseudo_arr):
        self.idx_df = idx_df.reset_index(drop=True); self.pseudo = pseudo_arr
        self.chunk_stride_sec = 5
    def __len__(self): return len(self.idx_df)
    def __getitem__(self, idx):
        row = self.idx_df.iloc[idx]; fp = row["filepath"]; ci = int(row["chunk_idx"])
        gidx = int(row["chunk_global_idx"])
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        ss = ci * CFG["sr"] * self.chunk_stride_sec; es = ss + CHUNK_SAMPLES
        if es > len(wav):
            wav_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            avail = min(CHUNK_SAMPLES, max(0, len(wav) - ss))
            if avail > 0: wav_chunk[:avail] = wav[ss:ss + avail]
        else:
            wav_chunk = wav[ss:es]
        pseudo = self.pseudo[gidx]
        return {
            "wav": torch.from_numpy(wav_chunk).float(),
            "clip_target_idx": -1,
            "frame_pseudo": torch.from_numpy(pseudo),
            "source_flag": 1,  # XC
        }


class SCPseudoDataset(Dataset):
    def __init__(self, files, entries, pseudo_arr):
        self.files = files; self.entries = entries; self.pseudo = pseudo_arr
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        fi, ci = self.entries[idx]; fp = self.files[fi]
        try:
            wav, sr = sf.read(str(fp), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(60 * CFG["sr"], dtype=np.float32)
        target_len = 60 * CFG["sr"]
        if len(wav) < target_len: wav = np.pad(wav, (0, target_len - len(wav)))
        else: wav = wav[:target_len]
        chunk = wav[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
        return {
            "wav": torch.from_numpy(chunk).float(),
            "clip_target_idx": -1,
            "frame_pseudo": torch.from_numpy(self.pseudo[idx]),
            "source_flag": 2,  # SC
        }


bc_train_ds = BC2026Dataset(bc_train_df, training=True)
bc_val_ds = BC2026Dataset(bc_val_df, training=False)
xc_ds = XCPseudoDataset(xc_index, xc_pseudo_arr)
sc_ds = SCPseudoDataset(sc_files, sc_entries, sc_pseudo_arr)

combined_train_ds = ConcatDataset([bc_train_ds, xc_ds, sc_ds])
print(f"BC train: {len(bc_train_ds)} | XC: {len(xc_ds)} | SC: {len(sc_ds)} | combined: {len(combined_train_ds)}")
print(f"Val (BC only): {len(bc_val_ds)}")
"""),

        code_cell("model", """class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"], n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"], power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x): return self.db(self.mel_spec(x))


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(float(p_init))); self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0); x = x.clamp(min=self.eps).pow(p)
        return x.mean(dim=2).pow(1.0 / p)


class BirdSEDModelEffv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG["backbone"], pretrained=True, in_chans=CFG["in_chans"], num_classes=0, global_pool="", drop_path_rate=CFG["drop_path_rate"])
        with torch.no_grad():
            n_tf = CHUNK_SAMPLES // CFG["hop_length"] + 1
            dummy = torch.randn(1, CFG["in_chans"], CFG["n_mels"], n_tf)
            self.backbone_dim = self.backbone(dummy).shape[1]
        self.gem_freq = GeMFreqPool(3.0)
        self.dense = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.backbone_dim, CFG["hidden_dim"]), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.att = nn.Conv1d(CFG["hidden_dim"], N_CLASSES, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(CFG["hidden_dim"], N_CLASSES, kernel_size=1, bias=True)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h).permute(0, 2, 1)
        h_cls = self.dense(h_cls).permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        fw = self.cla(h_cls)
        clip = torch.sum(norm_att * fw, dim=2)
        if return_framewise: return clip, fw.permute(0, 2, 1)
        return clip


mel_extractor = MelSpecTransform().to(DEVICE)
model = BirdSEDModelEffv2().to(DEVICE)

# WARM-START from exp058 R1
R1_CKPT_PATH = next(Path("/content/data/birdclef2026-exp058-effv2b0-combined").rglob("effv2b0_combined_best.pth"))
print(f"R1 ckpt: {R1_CKPT_PATH}")
try: r1_state = torch.load(str(R1_CKPT_PATH), map_location="cpu", weights_only=False)
except TypeError: r1_state = torch.load(str(R1_CKPT_PATH), map_location="cpu")
r1_sd = r1_state.get("state_dict", r1_state)
miss, unexp = model.load_state_dict(r1_sd, strict=False)
print(f"R1 warm-start: missing={len(miss)}, unexpected={len(unexp)}")
print(f"R1 val_ns22={r1_state.get('val_ns22'):.4f}, val_macro={r1_state.get('val_macro'):.4f}")
print(f"Student params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Determine T_student
with torch.no_grad():
    dw = torch.randn(1, CHUNK_SAMPLES, device=DEVICE)
    dm = mel_extractor(dw.unsqueeze(1))
    _, df_ = model(dm, return_framewise=True)
    T_STUDENT = df_.shape[1]
print(f"T_STUDENT: {T_STUDENT}")
"""),

        code_cell("aug", """def spec_augment(mel, freq_mask=CFG["spec_aug_freq_mask"], time_mask=CFG["spec_aug_time_mask"], n_freq=2, n_time=2):
    B, _, F_, T = mel.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask + 1); f0 = np.random.randint(0, max(1, F_ - f))
        mel[:, :, f0:f0+f, :] = 0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask + 1); t0 = np.random.randint(0, max(1, T - t))
        mel[:, :, :, t0:t0+t] = 0
    return mel
"""),

        code_cell("dataloader", """T_TEACHER = sc_pseudo_arr.shape[1]
assert T_TEACHER == xc_pseudo_arr.shape[1], f"T mismatch: SC={T_TEACHER} XC={xc_pseudo_arr.shape[1]}"
print(f"T_TEACHER: {T_TEACHER}, T_STUDENT: {T_STUDENT}")


def collate_fn(batch):
    wavs = torch.stack([b["wav"] for b in batch])
    clip_targets = torch.tensor([b["clip_target_idx"] for b in batch], dtype=torch.long)
    source_flags = torch.tensor([b["source_flag"] for b in batch], dtype=torch.long)
    pseudo_mask = source_flags > 0  # XC or SC
    if pseudo_mask.any():
        pseudos = torch.stack([b["frame_pseudo"] for b in batch if b["source_flag"] > 0])
    else:
        pseudos = torch.zeros(0, T_TEACHER, N_CLASSES, dtype=torch.float16)
    return {"wav": wavs, "clip_target": clip_targets, "source_flag": source_flags, "pseudos": pseudos}


train_loader = DataLoader(combined_train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
                          persistent_workers=True, collate_fn=collate_fn)
val_loader = DataLoader(bc_val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True,
                        collate_fn=lambda b: {"wav": torch.stack([x["wav"] for x in b]),
                                              "clip_target": torch.tensor([x["clip_target_idx"] for x in b], dtype=torch.long)})

n_steps = len(train_loader) * CFG["epochs"]
print(f"Steps/epoch: {len(train_loader)}, total: {n_steps}")

optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

def lr_lambda(step):
    if step < CFG["warmup_steps"]: return step / max(1, CFG["warmup_steps"])
    progress = (step - CFG["warmup_steps"]) / max(1, n_steps - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
"""),

        code_cell("eval_fn", """@torch.no_grad()
def evaluate():
    model.eval()
    all_logits = []; all_targets = []
    for batch in val_loader:
        wav = batch["wav"].to(DEVICE, non_blocking=True).unsqueeze(1)
        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
            logit = model(mel)
        all_logits.append(logit.float().cpu()); all_targets.append(batch["clip_target"])
    all_logits = torch.cat(all_logits); all_targets = torch.cat(all_targets)
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(all_logits).numpy()
    targets_onehot = np.eye(N_CLASSES)[all_targets.numpy()]
    aucs = []
    for c in range(N_CLASSES):
        if targets_onehot[:, c].sum() < 2: continue
        try: aucs.append(roc_auc_score(targets_onehot[:, c], probs[:, c]))
        except: pass
    aucs_arr = np.array(aucs)
    val_macro = float(np.mean(aucs_arr)) if len(aucs_arr) else 0.0
    ns_aucs = aucs_arr[aucs_arr < 1.0]
    val_ns22 = float(np.sort(ns_aucs)[:22].mean()) if len(ns_aucs) >= 22 else float(np.mean(ns_aucs)) if len(ns_aucs) > 0 else val_macro

    taxon_aucs = {}
    sp_df = species_df.copy(); sp_df["idx"] = sp_df["primary_label"].map(label_to_idx)
    for cls in sp_df["class_name"].unique():
        idx_list = sp_df[sp_df["class_name"] == cls]["idx"].tolist()
        cls_aucs = []
        for c in idx_list:
            if targets_onehot[:, c].sum() < 2: continue
            try: cls_aucs.append(roc_auc_score(targets_onehot[:, c], probs[:, c]))
            except: pass
        taxon_aucs[cls] = float(np.mean(cls_aucs)) if cls_aucs else float("nan")

    cstat = {
        "n": len(aucs_arr),
        "median": float(np.median(aucs_arr)) if len(aucs_arr) else 0.0,
        "p25": float(np.percentile(aucs_arr, 25)) if len(aucs_arr) else 0.0,
        "p75": float(np.percentile(aucs_arr, 75)) if len(aucs_arr) else 0.0,
        "n_gt05": int((aucs_arr > 0.5).sum()),
        "n_gt07": int((aucs_arr > 0.7).sum()),
        "n_gt09": int((aucs_arr > 0.9).sum()),
        "n_perfect": int((aucs_arr == 1.0).sum()),
    }
    return val_macro, val_ns22, taxon_aucs, cstat
"""),

        code_cell("train_loop", """best_val = 0.0
best_path = DRIVE_ROOT / "effv2b0_r2_sc_best.pth"
log_path = DRIVE_ROOT / "effv2b0_r2_sc_train.log"

step = 0
total_t0 = time.time()
for ep in range(CFG["epochs"]):
    t0 = time.time()
    model.train()
    losses = []; losses_bc = []; losses_pseudo = []
    n_bc_total = n_xc_total = n_sc_total = 0

    for bi, batch in enumerate(train_loader):
        wav = batch["wav"].to(DEVICE, non_blocking=True).unsqueeze(1)
        clip_target = batch["clip_target"].to(DEVICE)
        source_flag = batch["source_flag"].to(DEVICE)
        pseudos = batch["pseudos"].to(DEVICE, non_blocking=True).float()

        bc_mask = source_flag == 0
        xc_mask = source_flag == 1
        sc_mask = source_flag == 2
        n_bc = int(bc_mask.sum()); n_xc = int(xc_mask.sum()); n_sc = int(sc_mask.sum())
        n_bc_total += n_bc; n_xc_total += n_xc; n_sc_total += n_sc

        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
            mel = spec_augment(mel)
            clip_logits, frame_logits = model(mel, return_framewise=True)

            loss_total = 0.0
            loss_bc_val = torch.tensor(0.0, device=DEVICE)
            loss_pseudo_val = torch.tensor(0.0, device=DEVICE)

            # BC2026: clip-level BCE on hard label
            if n_bc > 0:
                bc_clip = clip_logits[bc_mask]
                bc_tgt_idx = clip_target[bc_mask]
                onh = F.one_hot(bc_tgt_idx, N_CLASSES).float()
                ls = CFG["label_smoothing"]
                smooth = onh * (1 - ls) + ls / N_CLASSES
                loss_bc_val = F.binary_cross_entropy_with_logits(bc_clip, smooth)
                loss_total = loss_total + loss_bc_val

            # XC + SC: frame-level BCE on pseudo (interpolate teacher T → student T)
            pseudo_mask = source_flag > 0
            n_pseudo = int(pseudo_mask.sum())
            if n_pseudo > 0:
                p_frame = frame_logits[pseudo_mask]
                pp = pseudos.permute(0, 2, 1)
                pp = F.interpolate(pp, size=T_STUDENT, mode="linear", align_corners=False)
                pp = pp.permute(0, 2, 1).clamp(min=1e-6, max=1.0 - 1e-6)
                # XC weight 0.5、SC weight 0.5 (uniform on pseudo samples)
                w_pseudo = CFG["xc_pseudo_weight"]  # = sc_pseudo_weight same value 0.5
                loss_pseudo_val = F.binary_cross_entropy_with_logits(p_frame, pp)
                loss_total = loss_total + w_pseudo * loss_pseudo_val

            loss = loss_total / max(1.0, float((n_bc > 0) + (n_pseudo > 0)) * 0.5)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if n_bc > 0: losses_bc.append(float(loss_bc_val.item()))
        if n_pseudo > 0: losses_pseudo.append(float(loss_pseudo_val.item()))
        step += 1
        if bi % 100 == 0:
            print(f"  [ep{ep+1} step {bi}/{len(train_loader)}] loss={loss.item():.4f} bc={loss_bc_val.item():.4f} pseudo={loss_pseudo_val.item():.4f} n_bc={n_bc} n_xc={n_xc} n_sc={n_sc} lr={scheduler.get_last_lr()[0]:.2e}")

    avg_loss = float(np.mean(losses))
    avg_bc = float(np.mean(losses_bc)) if losses_bc else 0.0
    avg_pseudo = float(np.mean(losses_pseudo)) if losses_pseudo else 0.0
    val_macro, val_ns22, taxon_aucs, cstat = evaluate()
    ep_el = time.time() - t0
    tot_el = time.time() - total_t0
    cur_lr = scheduler.get_last_lr()[0]
    is_best = val_ns22 > best_val
    best_mark = "BEST" if is_best else ""

    l1 = (f"=== Ep {ep+1}/{CFG['epochs']}: loss={avg_loss:.4f} (bc={avg_bc:.4f} pseudo={avg_pseudo:.4f}) "
          f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} {best_mark} lr={cur_lr:.2e} "
          f"({ep_el/60:.1f}min, total {tot_el/60:.1f}min) n_bc={n_bc_total} n_xc={n_xc_total} n_sc={n_sc_total} ===")
    l2 = "    taxon: " + " ".join(f"{k}={v:.3f}" if v == v else f"{k}=nan" for k, v in taxon_aucs.items())
    l3 = (f"    class: n={cstat['n']} median={cstat['median']:.3f} p25={cstat['p25']:.3f} p75={cstat['p75']:.3f} "
          f"#>0.5={cstat['n_gt05']} #>0.7={cstat['n_gt07']} #>0.9={cstat['n_gt09']} #perfect={cstat['n_perfect']}")
    print(l1); print(l2); print(l3)
    with open(log_path, "a") as f:
        f.write(l1 + "\\n" + l2 + "\\n" + l3 + "\\n")

    if is_best:
        best_val = val_ns22
        torch.save({"state_dict": model.state_dict(), "val_ns22": val_ns22, "val_macro": val_macro, "ep": ep+1, "cfg": CFG}, best_path)
        print(f"    BEST saved val_ns22={val_ns22:.4f}")

print(f"\\nDone. Best val_ns22: {best_val:.4f}")
print(f"Best ckpt: {best_path}")
"""),

        code_cell("upload", """import os, json, shutil
USER = "maekeso"
SLUG = "birdclef2026-exp065-effv2b0-r2-sc"
TITLE = "BirdCLEF2026 exp065 effv2b0 R2 SC"

UPLOAD_DIR = Path("/content/upload_exp065")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(best_path, UPLOAD_DIR / "effv2b0_r2_sc_best.pth")
shutil.copy(log_path, UPLOAD_DIR / "effv2b0_r2_sc_train.log")

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name":"other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset")
except Exception as e:
    print(f"create_new err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="exp065 R2 SC", dir_mode="zip", quiet=False)
        print("OK version")
    except Exception as e2:
        print(f"err: {str(e2)[:200]}")
print(f"URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),

        code_cell("disconnect", """from google.colab import runtime
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
    cid = c.get("id", "?")
    src = "".join(c["source"])
    print(f"    {c['cell_type']:8s} id={cid:18s} L={len(src.splitlines())}")
