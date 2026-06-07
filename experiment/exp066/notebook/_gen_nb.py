"""exp066: Tucker SED full reproduction (R1 = LB 0.90+ 目標).

戦略: Tucker SED の構成要素を全部入れる
  1. backbone: tf_efficientnet_b0.ns_jft_in1k (ns+jft+in1k pretrain)
  2. Perch v2 distill (ALPHA=1.0)
  3. SpecAug 弱 (10/10)
  4. MixUp focal-focal α=0.4
  5. Audio aug (gain ±6dB, noise SNR 10-30dB)
  6. labeled SC (66 files) を hard label として追加
  7. secondary_labels も活用
  8. MIN_SAMPLE=20 rare class upsample

Single NB on Colab Blackwell:
  - Pre-compute Perch embeddings (~30-60 min)
  - Train (~4-5h)

Output: maekeso/birdclef2026-exp066-tucker-repro
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp066\notebook\nb_train.ipynb")


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
        md_cell("hdr", """# exp066 — Tucker SED Full Reproduction (R1 target LB 0.90+)

**Goal**: Tucker SED (public LB 0.94+) の構成要素を全部入れて R1 から LB 0.90+

**Key features**:
  1. tf_efficientnet_b0.ns_jft_in1k (NS + JFT + ImageNet pretrain)
  2. Perch v2 distill (ALPHA = 1.0, MSE loss)
  3. SpecAug 10/10 (我々過去 30/60 を弱める)
  4. MixUp focal-focal α=0.4 (hard union labels)
  5. Audio aug: gain ±6dB, noise SNR 10-30dB
  6. labeled SC (66 files、train_soundscapes_labels.csv) hard label
  7. secondary_labels 活用 (train.csv の副 label)
  8. MIN_SAMPLE=20 rare class upsample

**Single NB** (Colab Blackwell):
  1. Pre-compute Perch embeddings (~30-60 min, GPU ONNX)
  2. Train 25 ep (~4-5h)
  3. Upload to Kaggle

**Output**: `maekeso/birdclef2026-exp066-tucker-repro`

**Expected LB**: 0.90-0.94 (Tucker と同水準狙い)
"""),

        code_cell("install", """!pip install -q timm==1.0.11 soundfile librosa kaggle onnxruntime-gpu 2>&1 | tail -1
from google.colab import drive
drive.mount('/content/drive')
import os
from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp066")
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
    "rishikeshjani/perch-onnx-for-birdclef-2026",
    # XC audio + filtered metadata (exp058 NB1 output で recordist cap 適用済)
    "maekeso/birdclef2026-xc-api-dl-part1",
    "maekeso/birdclef2026-xc-api-dl-part2",
    "maekeso/birdclef2026-xc-api-dl-part3",
    "maekeso/birdclef2026-exp058-xc-pseudo",  # filtered_metadata.csv 使う
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
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchaudio
import soundfile as sf
import librosa
import timm
import onnxruntime as ort
from tqdm.auto import tqdm
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
print(f"Device: {DEVICE}, GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""),

        code_cell("config", """CFG = dict(
    # Backbone (★ ns_jft_in1k 重要 ★)
    backbone="eca_nfnet_l0",  # ★ proven with Perch distill (exp020 R2 LB 0.915) ★
    in_chans=1,
    # Mel
    sr=32000, chunk_sec=5,
    n_mels=256, n_fft=2048, hop_length=512,
    fmin=20, fmax=16000,
    # Train (Blackwell 速度 up + A+C 短縮)
    epochs=15,                # A: 25 → 15 (Tucker 25 だが best ckpt 早期 plateau)
    batch_size=192,           # C: 128 → 192 (Blackwell 余裕)
    lr=1e-3, min_lr=1e-5, weight_decay=1e-4,  # C: 8e-4 → 1e-3 (linear scale)
    warmup_epochs=2,          # 15 ep に合わせて短縮
    # XC hard label (Q2 X1)
    use_xc=True,
    xc_max_chunks_per_file=1,  # 1 random crop per file per epoch (BC2026 same pattern)
    # Perch distill (★)
    use_perch_distill=True,
    perch_embed_dim=1536,
    alpha_distill=1.0,
    # SpecAug (Tucker mild 10/10)
    spec_aug_freq_mask=10, spec_aug_time_mask=10,
    n_freq_masks=1, n_time_masks=2,
    # MixUp
    use_focal_mixup=True, mixup_prob=0.5, mixup_alpha=0.4, mixup_hard=True,
    # Audio aug
    aug_prob=0.5,
    aug_gain_db_range=(-6.0, 6.0),
    aug_noise_snr_db_range=(10.0, 30.0),
    # Rare class
    min_sample=20,
    # Data sources
    use_focal_secondary=True,
    use_labeled_sc=True,
    val_split=0.1, val_seed=42,
    grad_clip=2.0, num_workers=8,
    label_smoothing=0.0,  # Tucker doesn't use
    hidden_dim=512, drop_path_rate=0.1,
)
for k, v in CFG.items(): print(f"  {k}: {v}")
CHUNK_SAMPLES = CFG["sr"] * CFG["chunk_sec"]
N_CLASSES = 234
N_WINDOWS = 12
"""),

        code_cell("species", f"""__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
label_to_idx = {{l: i for i, l in enumerate(PRIMARY_LABELS)}}
LABEL_TO_CLASS = dict(zip(species_df["primary_label"], species_df["class_name"]))
print(f"BC2026: {{len(PRIMARY_LABELS)}} species")
"""),

        code_cell("data_meta", """# Build metadata: BC2026 train_audio (focal) + labeled SC (66 files)
BC_DIR = Path("/content/data/birdclef-2026")

# 1. Focal (BC2026 train_audio) with primary + secondary labels
train_csv = pd.read_csv(BC_DIR / "train.csv")
print(f"train.csv: {len(train_csv)} rows, columns: {train_csv.columns.tolist()}")

focal_records = []
for _, r in train_csv.iterrows():
    pl = str(r["primary_label"])
    if pl not in label_to_idx: continue
    fp = BC_DIR / "train_audio" / str(r["filename"])
    if not fp.exists(): continue
    # Parse secondary_labels (string repr of list)
    sec_labels = []
    if CFG["use_focal_secondary"]:
        sec_raw = r.get("secondary_labels", "[]")
        if isinstance(sec_raw, str):
            try:
                sec_list = eval(sec_raw) if sec_raw.strip().startswith("[") else []
                sec_labels = [s for s in sec_list if s in label_to_idx]
            except Exception:
                sec_labels = []
    focal_records.append({
        "filepath": str(fp),
        "primary_label": pl,
        "secondary_labels": sec_labels,
        "source": "focal",
    })

print(f"Focal records: {len(focal_records)}")
n_with_sec = sum(1 for r in focal_records if r["secondary_labels"])
print(f"  with secondary: {n_with_sec}")

# 2. Labeled SC (train_soundscapes_labels.csv)
sc_labels_path = BC_DIR / "train_soundscapes_labels.csv"
sc_labels = pd.read_csv(sc_labels_path)
print(f"\\nSC labels: {len(sc_labels)} rows, columns: {sc_labels.columns.tolist()[:8]}")

sc_records = []
if CFG["use_labeled_sc"]:
    # SC labels schema: filename, start, end, primary_label (annotation format)
    # 1478 annotations、複数 annotation が同 chunk に重なる場合 multi-label に集約
    SC_AUDIO_DIR = BC_DIR / "train_soundscapes"
    # Parse "HH:MM:SS" or "M:SS" or float seconds → seconds
    def _to_sec(t):
        s = str(t).strip()
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    # Consolidate annotations by (filename, chunk_idx)
    sc_chunks_by_key = {}  # (filename, chunk_idx) → list of primary_labels
    for _, r in sc_labels.iterrows():
        fn = str(r["filename"])
        start = _to_sec(r["start"])
        pl = str(r["primary_label"])
        if pl not in label_to_idx: continue
        # chunk_idx: start time falls into chunk i if i*5 <= start < (i+1)*5
        chunk_idx = int(start // 5)
        if chunk_idx < 0 or chunk_idx >= N_WINDOWS: continue
        key = (fn, chunk_idx)
        if key not in sc_chunks_by_key:
            sc_chunks_by_key[key] = []
        sc_chunks_by_key[key].append(pl)
    print(f"  Unique (file, chunk) keys with annotations: {len(sc_chunks_by_key)}")
    for (fn, ci), labels in sc_chunks_by_key.items():
        fp = SC_AUDIO_DIR / fn
        if not fp.exists(): continue
        # multi-label per chunk (dedup)
        active = list(set(labels))
        sc_records.append({
            "filepath": str(fp),
            "chunk_idx": ci,
            "primary_label": active[0],
            "active_labels": active,
            "source": "sc",
        })
print(f"SC records (annotated chunks): {len(sc_records)}")

# Stratified val split on focal (10%)
np.random.seed(CFG["val_seed"])
focal_df = pd.DataFrame(focal_records)
val_mask = np.zeros(len(focal_df), dtype=bool)
for sp in focal_df["primary_label"].unique():
    idx = focal_df.index[focal_df["primary_label"] == sp].tolist()
    n_val = max(2, int(len(idx) * CFG["val_split"]))
    pick = np.random.choice(idx, size=min(n_val, len(idx)), replace=False)
    val_mask[pick] = True
focal_train_df = focal_df[~val_mask].reset_index(drop=True)
focal_val_df = focal_df[val_mask].reset_index(drop=True)
print(f"Focal train: {len(focal_train_df)}, val: {len(focal_val_df)}")

# MIN_SAMPLE upsampling for rare classes
species_counts = focal_train_df["primary_label"].value_counts()
rare_species = species_counts[species_counts < CFG["min_sample"]].index.tolist()
print(f"\\nRare species (<{CFG['min_sample']}): {len(rare_species)}")
upsampled_dfs = [focal_train_df]
for sp in rare_species:
    sp_df = focal_train_df[focal_train_df["primary_label"] == sp]
    n_needed = CFG["min_sample"] - len(sp_df)
    if n_needed > 0 and len(sp_df) > 0:
        # repeat with replacement
        upsampled = sp_df.sample(n=n_needed, replace=True, random_state=42)
        upsampled_dfs.append(upsampled)
focal_train_df = pd.concat(upsampled_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"After upsample focal_train: {len(focal_train_df)}")

sc_df = pd.DataFrame(sc_records) if sc_records else pd.DataFrame()
print(f"SC train: {len(sc_df)}")

# === XC HARD LABEL DATA (X1 plan) ===
xc_records = []
if CFG.get("use_xc", False):
    # Load filtered metadata (recordist cap N=10 applied in exp058 NB1)
    PSEUDO_DIR = Path("/content/data/birdclef2026-exp058-xc-pseudo")
    filtered_meta_path = next(PSEUDO_DIR.rglob("filtered_metadata.csv"), None)
    assert filtered_meta_path is not None, "exp058 filtered_metadata.csv not found"
    xc_meta = pd.read_csv(filtered_meta_path)
    print(f"\\nXC filtered metadata: {len(xc_meta)} rows, columns: {xc_meta.columns.tolist()[:10]}")

    # Use primary_label as hard label
    # filepath from exp058 (absolute /content/data/...) → fallback to filename + xc_root reconstruction
    XC_ROOTS = [
        Path("/content/data/birdclef2026-xc-api-dl-part1"),
        Path("/content/data/birdclef2026-xc-api-dl-part2"),
        Path("/content/data/birdclef2026-xc-api-dl-part3"),
    ]
    for _, r in xc_meta.iterrows():
        pl = str(r.get("primary_label", ""))
        if pl not in label_to_idx: continue
        fp = str(r.get("filepath", ""))
        if not Path(fp).exists():
            # fallback: reconstruct from filename + XC roots
            fn = str(r.get("filename", ""))
            for root in XC_ROOTS:
                cand = root / fn
                if cand.exists():
                    fp = str(cand); break
                # try with deeper rglob (xc_api/ subdir)
                cand2 = next(root.rglob(fn), None)
                if cand2 is not None and cand2.exists():
                    fp = str(cand2); break
            if not Path(fp).exists(): continue
        xc_records.append({
            "filepath": fp,
            "primary_label": pl,
            "secondary_labels": [],  # XC metadata doesn't have secondary in our schema
            "source": "xc",
        })
    print(f"XC records: {len(xc_records)}")
xc_df = pd.DataFrame(xc_records) if xc_records else pd.DataFrame()
print(f"XC train: {len(xc_df)}")
"""),

        code_cell("perch_setup", """# Perch v2 ONNX setup
PERCH_ONNX_PATH = next(Path("/content/data/perch-onnx-for-birdclef-2026").rglob("*.onnx"))
print(f"Perch ONNX: {PERCH_ONNX_PATH}")

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
perch_sess = ort.InferenceSession(str(PERCH_ONNX_PATH), providers=providers)
print(f"Providers: {perch_sess.get_providers()}")

PERCH_INPUT_NAME = perch_sess.get_inputs()[0].name
print(f"Perch input: {PERCH_INPUT_NAME}, shape: {perch_sess.get_inputs()[0].shape}")

# Find embedding output (1536-dim)
dummy = np.random.randn(1, CHUNK_SAMPLES).astype(np.float32)
outs = perch_sess.run(None, {PERCH_INPUT_NAME: dummy})
PERCH_EMBED_OUT_NAME = None
for o, info in zip(outs, perch_sess.get_outputs()):
    if o.shape[-1] == CFG["perch_embed_dim"]:
        PERCH_EMBED_OUT_NAME = info.name; break
assert PERCH_EMBED_OUT_NAME is not None, "embedding output not found"
print(f"Embed output: {PERCH_EMBED_OUT_NAME}")
"""),

        code_cell("perch_precompute", """# Pre-compute Perch embeddings for all train data (focal + SC)
# Per chunk: 5s center crop or fixed offset
# Store as numpy array, indexed by row index in df

# Combine focal_train + focal_val + sc into single list for embedding
# Use deterministic crop (center for focal, fixed chunk_idx for SC)

all_train_for_emb = []
# Focal: (filepath, "center", file_idx_in_focal)
for i, r in focal_train_df.iterrows():
    all_train_for_emb.append((r["filepath"], "focal_center", i, None))
for i, r in focal_val_df.iterrows():
    all_train_for_emb.append((r["filepath"], "focal_center", -(i+1), None))  # negative = val
# SC: (filepath, "sc_chunk", sc_idx, chunk_idx)
for i, r in sc_df.iterrows():
    all_train_for_emb.append((r["filepath"], "sc_chunk", i, int(r["chunk_idx"])))
# XC: (filepath, "xc_center", xc_idx, None)
for i, r in xc_df.iterrows():
    all_train_for_emb.append((r["filepath"], "xc_center", 1000000 + i, None))  # offset for routing
print(f"Total embeds to compute: {len(all_train_for_emb)}")

# DataLoader for embedding gen
class EmbedDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        fp, kind, did, ci = self.entries[idx]
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        if kind in ("focal_center", "xc_center"):
            if len(wav) < CHUNK_SAMPLES: wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
            else:
                start = (len(wav) - CHUNK_SAMPLES) // 2
                wav = wav[start:start + CHUNK_SAMPLES]
        else:  # sc_chunk
            target_len = 60 * CFG["sr"]
            if len(wav) < target_len: wav = np.pad(wav, (0, target_len - len(wav)))
            else: wav = wav[:target_len]
            wav = wav[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
        return wav.astype(np.float32), idx

emb_ds = EmbedDataset(all_train_for_emb)
emb_loader = DataLoader(emb_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                         collate_fn=lambda b: (np.stack([x[0] for x in b]), np.array([x[1] for x in b])))

# Allocate
n_focal_train = len(focal_train_df)
n_focal_val = len(focal_val_df)
n_sc = len(sc_df)
n_xc = len(xc_df)
emb_focal_train = np.zeros((n_focal_train, CFG["perch_embed_dim"]), dtype=np.float16)
emb_focal_val = np.zeros((n_focal_val, CFG["perch_embed_dim"]), dtype=np.float16)
emb_sc = np.zeros((n_sc, CFG["perch_embed_dim"]), dtype=np.float16)
emb_xc = np.zeros((n_xc, CFG["perch_embed_dim"]), dtype=np.float16)

t0 = time.time()
for bi, (wavs_np, idxs) in enumerate(emb_loader):
    out = perch_sess.run([PERCH_EMBED_OUT_NAME], {PERCH_INPUT_NAME: wavs_np.astype(np.float32)})[0]
    if out.ndim == 3:
        out = out.mean(axis=1)
    out_f16 = out.astype(np.float16)
    for k, eidx in enumerate(idxs):
        fp, kind, did, ci = all_train_for_emb[eidx]
        if kind == "focal_center":
            if did >= 0:
                emb_focal_train[did] = out_f16[k]
            else:
                emb_focal_val[-(did+1)] = out_f16[k]
        elif kind == "xc_center":
            xc_idx = did - 1000000
            emb_xc[xc_idx] = out_f16[k]
        else:
            emb_sc[did] = out_f16[k]
    if (bi + 1) % 100 == 0:
        el = (time.time() - t0) / 60
        eta = el / (bi + 1) * (len(emb_loader) - bi - 1)
        print(f"  emb [{bi+1}/{len(emb_loader)}] el={el:.1f}min eta={eta:.1f}min")

print(f"\\nPerch embeds done in {(time.time()-t0)/60:.1f}min")
print(f"  focal_train: {emb_focal_train.shape}, focal_val: {emb_focal_val.shape}, sc: {emb_sc.shape}, xc: {emb_xc.shape}")
del perch_sess; gc.collect()
"""),

        code_cell("datasets", """class BC2026FocalDataset(Dataset):
    def __init__(self, df, embeds, training=True):
        self.df = df.reset_index(drop=True)
        self.embeds = embeds
        self.training = training
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row["filepath"]; pl = row["primary_label"]
        sec = row.get("secondary_labels", [])
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
        # Audio aug (training only)
        if self.training and np.random.rand() < CFG["aug_prob"]:
            gain_db = np.random.uniform(*CFG["aug_gain_db_range"])
            wav = wav * (10 ** (gain_db / 20.0))
        if self.training and np.random.rand() < CFG["aug_prob"]:
            snr_db = np.random.uniform(*CFG["aug_noise_snr_db_range"])
            wav_power = np.mean(wav ** 2) + 1e-12
            noise_power = wav_power / (10 ** (snr_db / 10.0))
            noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(noise_power)
            wav = wav + noise
        # Multi-label target (primary + secondary)
        target = np.zeros(N_CLASSES, dtype=np.float32)
        target[label_to_idx[pl]] = 1.0
        for s in sec:
            if s in label_to_idx:
                target[label_to_idx[s]] = 1.0
        return {
            "wav": torch.from_numpy(wav).float(),
            "target": torch.from_numpy(target),
            "perch_embed": torch.from_numpy(self.embeds[idx].astype(np.float32)),
            "source": 0,  # focal
        }


class SCDataset(Dataset):
    def __init__(self, df, embeds):
        self.df = df.reset_index(drop=True)
        self.embeds = embeds
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row["filepath"]; ci = int(row["chunk_idx"])
        active = row["active_labels"]
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != CFG["sr"]:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG["sr"])
        except Exception:
            wav = np.zeros(60 * CFG["sr"], dtype=np.float32)
        target_len = 60 * CFG["sr"]
        if len(wav) < target_len: wav = np.pad(wav, (0, target_len - len(wav)))
        else: wav = wav[:target_len]
        wav = wav[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
        # Audio aug
        if np.random.rand() < CFG["aug_prob"]:
            gain_db = np.random.uniform(*CFG["aug_gain_db_range"])
            wav = wav * (10 ** (gain_db / 20.0))
        if np.random.rand() < CFG["aug_prob"]:
            snr_db = np.random.uniform(*CFG["aug_noise_snr_db_range"])
            wav_power = np.mean(wav ** 2) + 1e-12
            noise_power = wav_power / (10 ** (snr_db / 10.0))
            noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(noise_power)
            wav = wav + noise
        # Multi-label target (empty for negative chunks → all zeros)
        target = np.zeros(N_CLASSES, dtype=np.float32)
        for l in active:
            if l in label_to_idx:
                target[label_to_idx[l]] = 1.0
        return {
            "wav": torch.from_numpy(wav).float(),
            "target": torch.from_numpy(target),  # all zeros for negative chunks (no birds)
            "perch_embed": torch.from_numpy(self.embeds[idx].astype(np.float32)),
            "source": 1,  # sc
        }


focal_train_ds = BC2026FocalDataset(focal_train_df, emb_focal_train, training=True)
focal_val_ds = BC2026FocalDataset(focal_val_df, emb_focal_val, training=False)
sc_train_ds = SCDataset(sc_df, emb_sc) if len(sc_df) > 0 else None
# XC train uses same BC2026FocalDataset class (XC has hard label + filepath same structure)
xc_train_ds = BC2026FocalDataset(xc_df, emb_xc, training=True) if len(xc_df) > 0 else None

print(f"focal_train: {len(focal_train_ds)}, focal_val: {len(focal_val_ds)}, sc_train: {len(sc_train_ds) if sc_train_ds else 0}, xc_train: {len(xc_train_ds) if xc_train_ds else 0}")

# Combined train (concat focal + SC + XC)
from torch.utils.data import ConcatDataset
parts = [focal_train_ds]
if sc_train_ds is not None: parts.append(sc_train_ds)
if xc_train_ds is not None: parts.append(xc_train_ds)
combined_train = ConcatDataset(parts) if len(parts) > 1 else focal_train_ds
print(f"combined_train: {len(combined_train)}")
"""),

        code_cell("model", """class MelSpecTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"], n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"], power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x): return self.db(self.mel(x))


class GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(float(p_init))); self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0); x = x.clamp(min=self.eps).pow(p)
        return x.mean(dim=2).pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, bd, ed=1536):
        super().__init__(); self.proj = nn.Linear(bd, ed)
    def forward(self, fm): return self.proj(fm.mean(dim=[2,3]))


class SEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG["backbone"], pretrained=True, in_chans=CFG["in_chans"], num_classes=0, global_pool="", drop_path_rate=CFG["drop_path_rate"])
        with torch.no_grad():
            n_tf = CHUNK_SAMPLES // CFG["hop_length"] + 1
            dummy = torch.randn(1, CFG["in_chans"], CFG["n_mels"], n_tf)
            self.backbone_dim = self.backbone(dummy).shape[1]
        self.gem_freq = GeMFreq(3.0)
        self.dense = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.backbone_dim, CFG["hidden_dim"]), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.att = nn.Conv1d(CFG["hidden_dim"], N_CLASSES, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(CFG["hidden_dim"], N_CLASSES, kernel_size=1, bias=True)
        if CFG["use_perch_distill"]:
            self.distill_head = DistillHead(self.backbone_dim, CFG["perch_embed_dim"])
    def forward(self, x, return_distill=False):
        h = self.backbone(x)  # (B, C, H, W)
        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)
        h_cls = h.detach() if CFG["use_perch_distill"] else h
        h_cls = self.gem_freq(h_cls).permute(0, 2, 1)
        h_cls = self.dense(h_cls).permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        fw = self.cla(h_cls)
        clip = torch.sum(norm_att * fw, dim=2)
        if return_distill: return clip, fw.permute(0, 2, 1), distill_emb
        return clip


mel_extractor = MelSpecTF().to(DEVICE)
model = SEDModel().to(DEVICE)
print(f"Backbone: {CFG['backbone']}")
print(f"Backbone dim: {model.backbone_dim}")
print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
"""),

        code_cell("aug_mixup", """def spec_augment(mel, freq_mask=CFG["spec_aug_freq_mask"], time_mask=CFG["spec_aug_time_mask"], n_freq=CFG["n_freq_masks"], n_time=CFG["n_time_masks"]):
    B, _, F_, T = mel.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask + 1); f0 = np.random.randint(0, max(1, F_ - f))
        mel[:, :, f0:f0+f, :] = 0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask + 1); t0 = np.random.randint(0, max(1, T - t))
        mel[:, :, :, t0:t0+t] = 0
    return mel


def mixup_batch(wavs, targets, perch_embeds, alpha=0.4, hard=True):
    bsz = wavs.size(0)
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(bsz, device=wavs.device)
    wavs_mixed = lam * wavs + (1 - lam) * wavs[perm]
    perch_mixed = lam * perch_embeds + (1 - lam) * perch_embeds[perm]
    if hard:
        targets_mixed = torch.clamp(targets + targets[perm], 0, 1)
    else:
        targets_mixed = lam * targets + (1 - lam) * targets[perm]
    return wavs_mixed, targets_mixed, perch_mixed
"""),

        code_cell("dataloader", """def collate_fn(batch):
    wavs = torch.stack([b["wav"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    perch_embeds = torch.stack([b["perch_embed"] for b in batch])
    sources = torch.tensor([b["source"] for b in batch], dtype=torch.long)
    return {"wav": wavs, "target": targets, "perch_embed": perch_embeds, "source": sources}


# WeightedRandomSampler for Tucker-style ratio (focal 60% / SC 10% / XC 30%)
from torch.utils.data import WeightedRandomSampler
RATIO = {"focal": 0.6, "sc": 0.1, "xc": 0.3}
n_focal_t = len(focal_train_ds)
n_sc_t = len(sc_train_ds) if sc_train_ds else 0
n_xc_t = len(xc_train_ds) if xc_train_ds else 0
# Per-sample weight = source_ratio / source_count
weights = []
weights.extend([RATIO["focal"] / n_focal_t] * n_focal_t)
if sc_train_ds is not None:
    weights.extend([RATIO["sc"] / n_sc_t] * n_sc_t)
if xc_train_ds is not None:
    weights.extend([RATIO["xc"] / n_xc_t] * n_xc_t)
weights = torch.tensor(weights, dtype=torch.double)
print(f"Sample weights: focal={RATIO['focal']/n_focal_t:.2e}, sc={RATIO['sc']/n_sc_t if n_sc_t else 0:.2e}, xc={RATIO['xc']/n_xc_t if n_xc_t else 0:.2e}")
print(f"Effective ratios: focal={RATIO['focal']:.1f}, sc={RATIO['sc']:.1f}, xc={RATIO['xc']:.1f}")
print(f"Total weight sum: {weights.sum():.4f} (should be ~1.0)")

# num_samples = original combined size (1 epoch = 1 pass through all data conceptually)
sampler = WeightedRandomSampler(weights, num_samples=len(combined_train), replacement=True)

train_loader = DataLoader(combined_train, batch_size=CFG["batch_size"],
                          sampler=sampler,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
                          persistent_workers=True, collate_fn=collate_fn)
val_loader = DataLoader(focal_val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True,
                        collate_fn=collate_fn)

n_steps = len(train_loader) * CFG["epochs"]
warmup_steps = len(train_loader) * CFG["warmup_epochs"]
print(f"Steps/epoch: {len(train_loader)}, total: {n_steps}, warmup: {warmup_steps}")

optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
def lr_lambda(step):
    if step < warmup_steps: return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return cos * (1 - CFG["min_lr"]/CFG["lr"]) + CFG["min_lr"]/CFG["lr"]
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
        all_logits.append(logit.float().cpu()); all_targets.append(batch["target"])
    all_logits = torch.cat(all_logits); all_targets = torch.cat(all_targets)
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(all_logits).numpy()
    tgt = all_targets.numpy()
    aucs = []
    for c in range(N_CLASSES):
        if tgt[:, c].sum() < 2: continue
        try: aucs.append(roc_auc_score(tgt[:, c], probs[:, c]))
        except: pass
    aucs_arr = np.array(aucs)
    val_macro = float(np.mean(aucs_arr)) if len(aucs_arr) else 0.0
    ns_aucs = aucs_arr[aucs_arr < 1.0]
    val_ns22 = float(np.sort(ns_aucs)[:22].mean()) if len(ns_aucs) >= 22 else float(np.mean(ns_aucs)) if len(ns_aucs) > 0 else val_macro
    taxon = {}
    sp = species_df.copy(); sp["idx"] = sp["primary_label"].map(label_to_idx)
    for cls in sp["class_name"].unique():
        idx_list = sp[sp["class_name"] == cls]["idx"].tolist()
        cls_aucs = []
        for c in idx_list:
            if tgt[:, c].sum() < 2: continue
            try: cls_aucs.append(roc_auc_score(tgt[:, c], probs[:, c]))
            except: pass
        taxon[cls] = float(np.mean(cls_aucs)) if cls_aucs else float("nan")
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
    return val_macro, val_ns22, taxon, cstat
"""),

        code_cell("train_loop", """best_val = 0.0
best_path = DRIVE_ROOT / "exp066_best.pth"
log_path = DRIVE_ROOT / "exp066_train.log"

step = 0
total_t0 = time.time()
for ep in range(CFG["epochs"]):
    t0 = time.time()
    model.train()
    losses = []; losses_bce = []; losses_distill = []

    for bi, batch in enumerate(train_loader):
        wav = batch["wav"].to(DEVICE, non_blocking=True)  # (B, T)
        target = batch["target"].to(DEVICE)  # (B, 234) multi-label
        perch_emb = batch["perch_embed"].to(DEVICE)  # (B, 1536)

        # MixUp on whole batch (focal + sc mixed)
        if CFG["use_focal_mixup"] and np.random.rand() < CFG["mixup_prob"]:
            wav, target, perch_emb = mixup_batch(wav, target, perch_emb, CFG["mixup_alpha"], CFG["mixup_hard"])

        wav = wav.unsqueeze(1)  # (B, 1, T)
        with autocast('cuda', dtype=torch.bfloat16):
            mel = mel_extractor(wav)
            mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
            mel = spec_augment(mel)
            clip_logits, frame_logits, distill_emb = model(mel, return_distill=True)

            loss_bce = F.binary_cross_entropy_with_logits(clip_logits, target)
            loss_distill = F.mse_loss(distill_emb, perch_emb)
            loss = loss_bce + CFG["alpha_distill"] * loss_distill

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        losses_bce.append(float(loss_bce.item()))
        losses_distill.append(float(loss_distill.item()))
        step += 1
        if bi % 100 == 0:
            print(f"  [ep{ep+1} step {bi}/{len(train_loader)}] loss={loss.item():.4f} bce={loss_bce.item():.4f} distill={loss_distill.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    avg_loss = float(np.mean(losses))
    avg_bce = float(np.mean(losses_bce))
    avg_distill = float(np.mean(losses_distill))
    val_macro, val_ns22, taxon_aucs, cstat = evaluate()
    ep_el = time.time() - t0; tot_el = time.time() - total_t0
    cur_lr = scheduler.get_last_lr()[0]
    is_best = val_ns22 > best_val
    best_mark = "BEST" if is_best else ""

    l1 = (f"=== Ep {ep+1}/{CFG['epochs']}: loss={avg_loss:.4f} (bce={avg_bce:.4f} distill={avg_distill:.4f}) "
          f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} {best_mark} lr={cur_lr:.2e} "
          f"({ep_el/60:.1f}min, total {tot_el/60:.1f}min) ===")
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
SLUG = "birdclef2026-exp066-tucker-repro"
TITLE = "BirdCLEF2026 exp066 Tucker repro"

UPLOAD_DIR = Path("/content/upload_exp066")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(best_path, UPLOAD_DIR / "exp066_best.pth")
shutil.copy(log_path, UPLOAD_DIR / "exp066_train.log")
meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name":"other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset")
except Exception as e:
    print(f"err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="exp066 Tucker repro", dir_mode="zip", quiet=False)
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
