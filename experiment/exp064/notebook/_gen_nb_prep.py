"""Generate exp064 NB1: SC pseudo + Perch embed gen (Colab Blackwell).

Outputs (1 Kaggle Dataset):
  maekeso/birdclef2026-exp064-prep
    sc_pseudo.npz: (N_chunks_sc, T_frames, 234) fp16 pseudo on train_soundscapes
    sc_pseudo_index.csv: chunk → (file, chunk_idx)
    perch_embeds_bc.npz: (N_bc_files, 1536) fp16 — Perch embed for BC2026 center 5s
    perch_embeds_bc_index.csv: file → row_idx
    perch_embeds_sc.npz: (N_chunks_sc, 1536) fp16 — Perch embed per SC chunk
    perch_embeds_sc_index.csv: chunk → row_idx
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp064\notebook\nb_prep.ipynb")


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# exp064 NB1 — SC Pseudo + Perch Embed Gen (Colab Blackwell)

**Pipeline**:
1. exp020 R2 5-fold で train_soundscapes (10,592 files) に pseudo frame-level
2. Perch v2 ONNX で BC2026 train_audio (center 5s) embedding (~35k files)
3. Perch v2 ONNX で train_soundscapes per-chunk embedding (~120k chunks)
4. Kaggle Dataset upload

**Inputs**:
- maekeso/birdclef2026-exp020-weights-5fold (R2 ckpts, teacher for pseudo)
- rishikeshjani/perch-onnx-for-birdclef-2026 (Perch v2 ONNX)
- birdclef-2026 (train_audio + train_soundscapes)

**Output**: maekeso/birdclef2026-exp064-prep (~1.5 GB)

**Time**: ~2-3h on Colab Blackwell
"""),

        code_cell("install", """!pip install -q timm==1.0.11 soundfile librosa kaggle onnxruntime-gpu 2>&1 | tail -1
from google.colab import drive
drive.mount('/content/drive')
import os
from pathlib import Path
DRIVE_ROOT = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp064")
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

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)
t0 = time.time()

DATASETS = [
    "maekeso/birdclef2026-exp020-weights-5fold",
    "rishikeshjani/perch-onnx-for-birdclef-2026",
]
for ds in DATASETS:
    name = ds.split("/")[-1]
    dst = LOCAL_DATA / name
    if dst.exists() and any(dst.iterdir()):
        print(f"  ✓ {name} exists, skip"); continue
    dst.mkdir(exist_ok=True)
    print(f"  DL {ds}...")
    api.dataset_download_files(ds, path=str(dst), unzip=True, quiet=False)

# BC2026 competition
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

        code_cell("setup", """import sys, gc, re, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import librosa
import timm
import onnxruntime as ort
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
"""),

        code_cell("config", """SR = 32000
CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC
N_MELS = 256; N_FFT = 2048; HOP = 512; FMIN = 20; FMAX = 16000
N_CLASSES = 234
PERCH_EMBED_DIM = 1536
N_WINDOWS = 12
"""),

        code_cell("species", """BC_DIR = Path("/content/data/birdclef-2026")
taxo = pd.read_csv(BC_DIR / "taxonomy.csv")
PRIMARY_LABELS = taxo["primary_label"].tolist()
print(f"N_CLASSES: {len(PRIMARY_LABELS)}")
"""),

        code_cell("teacher_model", """# Teacher: exp020 R2 5-fold ONNX export & load
E20_DIR = next(Path("/content/data/birdclef2026-exp020-weights-5fold").rglob("r2_fold0_ckpt_best_ns22.pth")).parent
print(f"teacher ckpt dir: {E20_DIR}")

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
            n_tf = CHUNK_SAMPLES // HOP + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            self.backbone_dim = self.backbone(dummy).shape[1]
        self.gem_freq = _GeMFreq(3.0)
        self.dense = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.backbone_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.att = nn.Conv1d(512, N_CLASSES, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(512, N_CLASSES, kernel_size=1, bias=True)
        self.distill_head = _DistillHead(self.backbone_dim, PERCH_EMBED_DIM)
    def forward(self, x):
        h = self.backbone(x)
        h_cls = h.detach()
        h_cls = self.gem_freq(h_cls).permute(0, 2, 1)
        h_cls = self.dense(h_cls).permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        fw = self.cla(h_cls)
        clip = torch.sum(norm_att * fw, dim=2)
        return clip, fw.permute(0, 2, 1)

class _MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x): return self.db(self.mel(x))

teacher_mel_tf = _MelTF().to(DEVICE)
teacher_ckpts = sorted(E20_DIR.rglob("r2_fold*_ckpt_best_ns22.pth"))
print(f"Found {len(teacher_ckpts)} teacher folds")
teacher_models = []
for ck in teacher_ckpts:
    try: st = torch.load(str(ck), map_location="cpu", weights_only=False)
    except TypeError: st = torch.load(str(ck), map_location="cpu")
    m = _E20SED().to(DEVICE); m.load_state_dict(st["model_state"], strict=False); m.eval()
    teacher_models.append(m); del st; gc.collect()
print(f"5-fold teacher loaded")
"""),

        code_cell("perch_setup", """# Perch v2 ONNX setup (input: raw waveform 5s at 32kHz)
PERCH_ONNX_PATH = next(Path("/content/data/perch-onnx-for-birdclef-2026").rglob("*.onnx"))
print(f"Perch ONNX: {PERCH_ONNX_PATH}")

# Use GPU provider if available
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
perch_sess = ort.InferenceSession(str(PERCH_ONNX_PATH), providers=providers)
print(f"Perch providers: {perch_sess.get_providers()}")
for inp in perch_sess.get_inputs():
    print(f"  input: {inp.name} shape={inp.shape} dtype={inp.type}")
for out in perch_sess.get_outputs():
    print(f"  output: {out.name} shape={out.shape}")
"""),

        code_cell("perch_check", """# Verify Perch v2: input 1D waveform 32k * 5sec = 160000 samples, output embedding 1536
import numpy as np
dummy_wav = np.random.randn(1, 5 * SR).astype(np.float32)
inp_name = perch_sess.get_inputs()[0].name
out_names = [o.name for o in perch_sess.get_outputs()]
print(f"running Perch test with input shape {dummy_wav.shape}")
outs = perch_sess.run(None, {inp_name: dummy_wav})
for n, o in zip(out_names, outs):
    print(f"  out {n}: shape={o.shape}, dtype={o.dtype}")
# We want embedding output (1536-dim). Find which one
EMBED_OUT_NAME = None
for n, o in zip(out_names, outs):
    if o.shape[-1] == 1536:
        EMBED_OUT_NAME = n; break
print(f"EMBED_OUT_NAME: {EMBED_OUT_NAME}")
assert EMBED_OUT_NAME is not None
"""),

        code_cell("sc_pseudo_gen", """# SC pseudo gen with exp020 R2 5-fold teacher (train_soundscapes)
import time
from scipy.ndimage import gaussian_filter1d

SC_DIR = BC_DIR / "train_soundscapes"
sc_files = sorted(SC_DIR.glob("*.ogg"))
print(f"SC files: {len(sc_files)}")

# Each file: 60s -> 12 chunks of 5s
class _SCChunkDataset(Dataset):
    def __init__(self, files):
        self.entries = []  # (fi, ci)
        self.files = files
        for fi, fp in enumerate(files):
            for ci in range(N_WINDOWS):
                self.entries.append((fi, ci))
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        fi, ci = self.entries[idx]
        fp = self.files[fi]
        try:
            wav, sr = sf.read(str(fp), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        except Exception:
            wav = np.zeros(60 * SR, dtype=np.float32)
        target_len = 60 * SR
        if len(wav) < target_len: wav = np.pad(wav, (0, target_len - len(wav)))
        else: wav = wav[:target_len]
        chunk = wav[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
        return torch.from_numpy(chunk).float(), fi, ci

sc_ds = _SCChunkDataset(sc_files)
print(f"SC chunks: {len(sc_ds)}")
sc_loader = DataLoader(sc_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# Determine T_frames
with torch.no_grad():
    w, _, _ = next(iter(sc_loader))
    w = w.to(DEVICE).unsqueeze(1)
    m = teacher_mel_tf(w)
    m = (m - m.mean()) / (m.std() + 1e-6)
    _, fw = teacher_models[0](m)
    T_FRAMES = fw.shape[1]
print(f"T_FRAMES: {T_FRAMES}")

# Allocate
sc_pseudo = np.zeros((len(sc_ds), T_FRAMES, N_CLASSES), dtype=np.float16)
sc_index = []
t0 = time.time()
chunk_g = 0
for bi, (wavs, fis, cis) in enumerate(sc_loader):
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
    sc_pseudo[chunk_g:chunk_g + bsz] = accum
    for j in range(bsz):
        sc_index.append((chunk_g + j, int(fis[j]), int(cis[j])))
    chunk_g += bsz
    if (bi + 1) % 50 == 0:
        el = (time.time() - t0) / 60
        eta = el / (bi + 1) * (len(sc_loader) - bi - 1)
        print(f"  [{bi+1}/{len(sc_loader)}] el={el:.1f}min eta={eta:.1f}min")
print(f"SC pseudo done: {sc_pseudo.shape} in {(time.time()-t0)/60:.1f}min")

# Save
sc_index_df = pd.DataFrame(sc_index, columns=["chunk_idx_global", "file_idx", "chunk_idx"])
sc_index_df["filename"] = sc_index_df["file_idx"].map(lambda i: sc_files[i].name)
np.savez_compressed(DRIVE_ROOT / "sc_pseudo.npz", pseudo=sc_pseudo)
sc_index_df.to_csv(DRIVE_ROOT / "sc_pseudo_index.csv", index=False)
print(f"saved sc_pseudo.npz + sc_pseudo_index.csv")
del teacher_models
gc.collect()
"""),

        code_cell("perch_bc_gen", """# Perch embed gen for BC2026 train_audio (center 5s per file)
train_csv = pd.read_csv(BC_DIR / "train.csv")
print(f"train.csv: {len(train_csv)} rows")
BC_AUDIO_DIR = BC_DIR / "train_audio"

# Collect existing files
bc_records = []
for _, r in train_csv.iterrows():
    pl = str(r["primary_label"])
    fp = BC_AUDIO_DIR / str(r["filename"])
    if fp.exists():
        bc_records.append((str(fp), pl, str(r["filename"])))
print(f"BC files found: {len(bc_records)}")

class _BCChunkDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        fp, pl, fn = self.records[idx]
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        if len(wav) < CHUNK_SAMPLES:
            wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
        else:
            start = (len(wav) - CHUNK_SAMPLES) // 2  # center crop
            wav = wav[start:start + CHUNK_SAMPLES]
        return wav, idx

bc_ds = _BCChunkDataset(bc_records)
bc_loader = DataLoader(bc_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
                       collate_fn=lambda b: (np.stack([x[0] for x in b]), np.array([x[1] for x in b])))

# Allocate
bc_embeds = np.zeros((len(bc_ds), PERCH_EMBED_DIM), dtype=np.float16)
t0 = time.time()
inp_name = perch_sess.get_inputs()[0].name
for bi, (wavs_np, idxs) in enumerate(bc_loader):
    wavs_np = wavs_np.astype(np.float32)
    out = perch_sess.run([EMBED_OUT_NAME], {inp_name: wavs_np})[0]  # (B, ..., 1536)
    if out.ndim == 3:
        # (B, T, C) → mean over T
        out = out.mean(axis=1)
    bc_embeds[idxs] = out.astype(np.float16)
    if (bi + 1) % 50 == 0:
        el = (time.time() - t0) / 60
        eta = el / (bi + 1) * (len(bc_loader) - bi - 1)
        print(f"  BC [{bi+1}/{len(bc_loader)}] el={el:.1f}min eta={eta:.1f}min")

print(f"BC Perch done: {bc_embeds.shape} in {(time.time()-t0)/60:.1f}min")

# Save
bc_index_df = pd.DataFrame(bc_records, columns=["filepath", "primary_label", "filename"])
np.savez_compressed(DRIVE_ROOT / "perch_embeds_bc.npz", embeds=bc_embeds)
bc_index_df.to_csv(DRIVE_ROOT / "perch_embeds_bc_index.csv", index=False)
print(f"saved perch_embeds_bc.npz + index.csv")
"""),

        code_cell("perch_sc_gen", """# Perch embed gen for train_soundscapes (per chunk)
sc_embeds = np.zeros((len(sc_ds), PERCH_EMBED_DIM), dtype=np.float16)
t0 = time.time()
for bi, (wavs, fis, cis) in enumerate(sc_loader):
    wavs_np = wavs.numpy().astype(np.float32)
    out = perch_sess.run([EMBED_OUT_NAME], {inp_name: wavs_np})[0]
    if out.ndim == 3:
        out = out.mean(axis=1)
    bsz = out.shape[0]
    g_start = bi * 64
    sc_embeds[g_start:g_start + bsz] = out.astype(np.float16)
    if (bi + 1) % 50 == 0:
        el = (time.time() - t0) / 60
        eta = el / (bi + 1) * (len(sc_loader) - bi - 1)
        print(f"  SC [{bi+1}/{len(sc_loader)}] el={el:.1f}min eta={eta:.1f}min")

print(f"SC Perch done: {sc_embeds.shape} in {(time.time()-t0)/60:.1f}min")
np.savez_compressed(DRIVE_ROOT / "perch_embeds_sc.npz", embeds=sc_embeds)
print(f"saved perch_embeds_sc.npz")
del perch_sess
gc.collect()
"""),

        code_cell("upload", """# Upload to Kaggle Dataset
import os, json, shutil
USER = "maekeso"
SLUG = "birdclef2026-exp064-prep"
TITLE = "BirdCLEF2026 exp064 prep"

UPLOAD_DIR = Path("/content/upload_exp064_prep")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
for f in ["sc_pseudo.npz", "sc_pseudo_index.csv",
          "perch_embeds_bc.npz", "perch_embeds_bc_index.csv",
          "perch_embeds_sc.npz"]:
    shutil.copy(DRIVE_ROOT / f, UPLOAD_DIR / f)

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
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="exp064 prep", dir_mode="zip", quiet=False)
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
