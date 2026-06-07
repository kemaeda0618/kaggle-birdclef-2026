"""Generate exp058 NB1: XC metadata refresh + filter + pseudo gen (combined).

Path: OverfitOracle 1-stage combined training の前段。
1. XC metadata 再 fetch (recordist 追加)
2. Filter: quality A/B + per-recordist cap N=10
3. exp020 R2 5-fold で filtered XC chunks に per-frame pseudo gen
4. Kaggle Dataset upload

Inputs (6 Kaggle Datasets):
- maekeso/birdclef2026-xc-api-dl-part1 (76 Aves audio + metadata)
- maekeso/birdclef2026-xc-api-dl-part2 (83 Aves audio + metadata)
- maekeso/birdclef2026-xc-api-dl-part3 (27 non-Aves audio + metadata)
- maekeso/birdclef2026-exp020-weights-5fold (R2 5-fold ckpts)
- birdclef-2026 (taxonomy.csv のみ使用)

Output:
- maekeso/birdclef2026-exp058-xc-pseudo (Kaggle Dataset)
  - filtered_metadata.csv: filter 後の XC file list (recordist 込み)
  - xc_pseudo.npz: per-frame pseudo predictions (B, T_frames, 234) fp16
  - xc_pseudo_index.csv: each pseudo entry の (xc_id, file_path, n_chunks)

Target: Colab Blackwell (RTX PRO 6000 96GB)
Time: ~3-5h (metadata 10min + DL data + filter + 5-fold inference)
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp058\notebook\nb_pseudo.ipynb")


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# exp058 NB1 — XC Metadata Refresh + Filter + Pseudo Gen (Colab Blackwell)

**Path**: OverfitOracle 1-stage combined training の前段準備

**Pipeline**:
1. XC metadata 再 fetch (XC API、recordist 追加、~10 min)
2. Filter: quality A/B + per-recordist cap N=10
3. exp020 R2 (5-fold eca_nfnet_l0 SED) を teacher として filtered XC で推論
4. Per-frame pseudo (5-fold 平均、fp16) を Kaggle Dataset として upload

**Teacher**: exp020 R2 (5-fold eca_nfnet_l0)、val avg 0.9358、LB 0.915 (5-fold ensemble)

**Output**:
- `maekeso/birdclef2026-exp058-xc-pseudo`
  - `filtered_metadata.csv`: filter 後の XC file list
  - `xc_pseudo.npz`: (n_chunks, T_frames, 234) per-frame pseudo predictions fp16
  - `xc_pseudo_index.csv`: each chunk → (xc_id, file_path, chunk_idx, scientific_name)

**Expected time**: 3-5h on Colab Blackwell (depending on filtered file count)
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

        code_cell("dl_data", """# Cell 2: Download XC datasets + exp020 R2 ckpts + BC2026 (auth + DL self-contained)
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
        print("  ✓ Copied kaggle.json from Drive")
_kgat = json.loads((KAGGLE_DIR / "kaggle.json").read_text())["key"]
os.environ["KAGGLE_API_TOKEN"] = _kgat
print(f"  KAGGLE_API_TOKEN set")

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

LOCAL_DATA = Path("/content/data")
LOCAL_DATA.mkdir(exist_ok=True)
t0 = time.time()

DATASETS = [
    "maekeso/birdclef2026-xc-api-dl-part1",
    "maekeso/birdclef2026-xc-api-dl-part2",
    "maekeso/birdclef2026-xc-api-dl-part3",
    "maekeso/birdclef2026-exp020-weights-5fold",
]

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

# BC2026 competition (taxonomy.csv のみ必要)
BC_DIR = LOCAL_DATA / "birdclef-2026"
if not (BC_DIR / "taxonomy.csv").exists():
    BC_DIR.mkdir(exist_ok=True)
    print(f"  DL BC2026 taxonomy...")
    try:
        api.competition_download_file("birdclef-2026", "taxonomy.csv", path=str(BC_DIR), quiet=False)
        # may come as .zip
        import zipfile
        zp = BC_DIR / "taxonomy.csv.zip"
        if zp.exists():
            with zipfile.ZipFile(zp) as zf: zf.extractall(BC_DIR)
            zp.unlink()
        print(f"    ✓ taxonomy.csv")
    except Exception as e:
        print(f"    ✗ err: {str(e)[:200]}")

print(f"\\nDL total: {(time.time()-t0)/60:.1f} min")
"""),

        code_cell("setup", """# Cell 3: imports + GPU info
import os, sys, json, time, math, gc, re
from pathlib import Path
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
import requests
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")
"""),

        code_cell("species_taxo", """# Cell 4: BC2026 species → primary_label mapping (taxonomy.csv)
import pandas as pd

BC_DIR = Path("/content/data/birdclef-2026")
taxo = pd.read_csv(BC_DIR / "taxonomy.csv")
print(f"BC2026 taxonomy: {len(taxo)} species")
print(f"  columns: {taxo.columns.tolist()}")
print(f"  by class: {taxo['class_name'].value_counts().to_dict()}")

PRIMARY_LABELS = taxo["primary_label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}

# scientific_name → primary_label mapping (lowercase 統一)
SCI_TO_LABEL = {str(s).lower(): l for s, l in zip(taxo["scientific_name"], taxo["primary_label"])}
print(f"sci_to_label: {len(SCI_TO_LABEL)} entries")
"""),

        code_cell("load_existing_meta", """# Cell 5: Load existing XC metadata (from Part 1/2/3)
LOCAL_DATA = Path("/content/data")

XC_DIRS = [
    ("birdclef2026-xc-api-dl-part1", "part1"),
    ("birdclef2026-xc-api-dl-part2", "part2"),
    ("birdclef2026-xc-api-dl-part3", "part3"),
]

meta_rows = []
for ds_name, part in XC_DIRS:
    root = LOCAL_DATA / ds_name
    if not root.exists():
        print(f"⚠ {ds_name} not found, skip")
        continue
    # Find _metadata.csv (may be in xc_api/ or xc_api_part2/ subdir)
    meta_csv = next(root.rglob("_metadata.csv"), None)
    if meta_csv is None:
        print(f"⚠ no _metadata.csv in {ds_name}")
        continue
    df = pd.read_csv(meta_csv)
    df["part"] = part
    df["_metadata_csv"] = str(meta_csv)
    meta_rows.append(df)
    print(f"  {part}: {len(df)} rows ({meta_csv.relative_to(root)})")

meta_existing = pd.concat(meta_rows, ignore_index=True)
print(f"\\nTotal existing metadata: {len(meta_existing)} rows")
print(f"  unique xc_id: {meta_existing['xc_id'].nunique()}")
print(f"  unique scientific_name: {meta_existing['scientific_name'].nunique()}")
print(f"  columns: {meta_existing.columns.tolist()}")
"""),

        code_cell("scan_audio", """# Cell 6: Scan XC audio files (locate by xc_id)
import re

LOCAL_DATA = Path("/content/data")

# Audio dir 構造: <root>/audio/<species_label>/XC<id>.mp3 or .ogg
def find_audio_files(part_root):
    audio_files = []
    for ext in ["*.mp3", "*.ogg", "*.wav", "*.flac", "*.MP3", "*.OGG"]:
        audio_files.extend(list(part_root.rglob(ext)))
    return audio_files

xc_id_to_path = {}
for ds_name, part in [("birdclef2026-xc-api-dl-part1", "part1"),
                       ("birdclef2026-xc-api-dl-part2", "part2"),
                       ("birdclef2026-xc-api-dl-part3", "part3")]:
    root = LOCAL_DATA / ds_name
    if not root.exists():
        continue
    audio_files = find_audio_files(root)
    for p in audio_files:
        # extract xc_id from filename (e.g., XC123456.mp3)
        m = re.search(r"XC(\d+)", p.name, re.IGNORECASE)
        if m:
            xc_id = int(m.group(1))
            if xc_id not in xc_id_to_path:
                xc_id_to_path[xc_id] = str(p)
    print(f"  {part}: {len(audio_files)} audio files found")

print(f"\\nUnique xc_id with audio: {len(xc_id_to_path)}")

# Join with metadata
meta_existing["xc_id_int"] = pd.to_numeric(meta_existing["xc_id"], errors="coerce").astype("Int64")
meta_existing["filepath"] = meta_existing["xc_id_int"].map(lambda x: xc_id_to_path.get(int(x)) if pd.notna(x) else None)
meta_existing_with_audio = meta_existing[meta_existing["filepath"].notna()].copy()
print(f"  metadata rows with audio file: {len(meta_existing_with_audio)}")
print(f"  metadata rows without audio file: {len(meta_existing) - len(meta_existing_with_audio)}")
"""),

        code_cell("xc_meta_refresh", """# Cell 7: XC API metadata refresh (add recordist field)
# Only fetch (xc_id, recordist) pairs; reuse existing fields for everything else.

XC_API_KEY = "1eef1b2770d218e80cea18fc28621e8dcad01f6a"  # PRIVATE NB
XC_API_URL = "https://xeno-canto.org/api/3/recordings"
QUERY_DELAY_SEC = 1.5
REQUEST_TIMEOUT = 30

USER_AGENT = "BirdCLEF2026-research/1.0 (Kaggle research project; contact: kaggle.com/maekeso)"

unique_species = meta_existing_with_audio["scientific_name"].dropna().unique().tolist()
print(f"Fetching metadata for {len(unique_species)} species (recordist field)")

species_records = []
errors = []
t0 = time.time()
for idx, sci_name in enumerate(tqdm(unique_species)):
    page = 1
    while True:
        params = {"query": f'sp:"{sci_name}"', "page": page, "key": XC_API_KEY}
        try:
            r = requests.get(XC_API_URL, params=params, timeout=REQUEST_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code != 200:
                errors.append((sci_name, page, f"HTTP {r.status_code}"))
                break
            data = r.json()
        except Exception as e:
            errors.append((sci_name, page, str(e)[:100]))
            break
        recs = data.get("recordings", [])
        if not recs:
            break
        for rec in recs:
            species_records.append({
                "xc_id": rec.get("id"),
                "scientific_name": sci_name,
                "recordist": rec.get("rec"),  # ★ 録音者名 ★
                "quality": rec.get("q"),       # 既存と一致確認用
            })
        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1
        time.sleep(QUERY_DELAY_SEC)
    time.sleep(QUERY_DELAY_SEC)
    if (idx + 1) % 20 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"  [{idx+1}/{len(unique_species)}] {len(species_records)} records, {len(errors)} errors, {elapsed:.1f} min")

meta_refresh = pd.DataFrame(species_records)
print(f"\\nXC API refresh: {len(meta_refresh)} records, {len(errors)} errors")
print(f"  time: {(time.time()-t0)/60:.1f} min")
print(f"  recordists: {meta_refresh['recordist'].nunique()} unique")
if errors[:3]: print(f"  first errors: {errors[:3]}")

# Save refresh result
meta_refresh.to_csv(DRIVE_ROOT / "xc_metadata_refresh.csv", index=False)
print(f"  Saved: {DRIVE_ROOT / 'xc_metadata_refresh.csv'}")
"""),

        code_cell("filter", """# Cell 8: Apply filters (quality + per-recordist cap)
# Filter 1: quality in [A, B]
# Filter 2: per-recordist cap N=10

# Merge existing metadata (has audio) with refresh (has recordist)
meta_refresh["xc_id_int"] = pd.to_numeric(meta_refresh["xc_id"], errors="coerce").astype("Int64")
merged = meta_existing_with_audio.merge(
    meta_refresh[["xc_id_int", "recordist"]],
    on="xc_id_int",
    how="left",
)
print(f"After merge: {len(merged)} rows")
print(f"  has recordist: {merged['recordist'].notna().sum()}")

# Filter 1: quality (use existing quality field which is the original from DL time)
QUALITY_OK = ["A", "B"]
n_before = len(merged)
merged_q = merged[merged["quality"].isin(QUALITY_OK)].copy()
print(f"\\nFilter 1 (quality in {QUALITY_OK}): {n_before} → {len(merged_q)}")
print(f"  quality breakdown (before filter): {merged['quality'].value_counts().to_dict()}")

# Filter 2: per-recordist cap N=10 per (species, recordist)
RECORDIST_CAP_N = 10
n_before = len(merged_q)
filtered = (merged_q
            .sort_values(["scientific_name", "recordist", "xc_id_int"])
            .groupby(["scientific_name", "recordist"], dropna=False)
            .head(RECORDIST_CAP_N)
            .reset_index(drop=True))
print(f"\\nFilter 2 (per-recordist cap N={RECORDIST_CAP_N}): {n_before} → {len(filtered)}")

# Map scientific_name → primary_label
filtered["scientific_name_lower"] = filtered["scientific_name"].astype(str).str.lower()
filtered["primary_label"] = filtered["scientific_name_lower"].map(SCI_TO_LABEL)
n_before = len(filtered)
filtered = filtered[filtered["primary_label"].notna()].copy()
print(f"\\nFilter 3 (BC2026 species mapping): {n_before} → {len(filtered)}")

# Summary
print(f"\\n=== Filtered XC dataset ===")
print(f"  files: {len(filtered)}")
print(f"  unique species (primary_label): {filtered['primary_label'].nunique()} / {N_CLASSES}")
print(f"  unique recordists: {filtered['recordist'].nunique()}")
print(f"  per-species file count: mean={filtered.groupby('primary_label').size().mean():.1f}, median={filtered.groupby('primary_label').size().median():.0f}")

# Save filtered metadata
filtered.to_csv(DRIVE_ROOT / "filtered_metadata.csv", index=False)
print(f"\\nSaved: {DRIVE_ROOT / 'filtered_metadata.csv'}")
"""),

        code_cell("model_def", """# Cell 9: BirdSEDModel definition (must match exp020 R2 training)
SR = 32000
TRAIN_DURATION = 5
TRAIN_SAMPLES = SR * TRAIN_DURATION
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000
BACKBONE = "eca_nfnet_l0"
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536


class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, wav):
        return self.db_transform(self.mel_spec(wav))


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


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
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
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits

print("OK model def ready")
"""),

        code_cell("load_ckpts", """# Cell 10: Load 5 fold ckpts → GPU
STATE_DIR = Path("/content/data/birdclef2026-exp020-weights-5fold")
ckpt_files = sorted(STATE_DIR.rglob("r2_fold*_ckpt_best_ns22.pth"))
assert len(ckpt_files) >= 1, f"No R2 ckpts in {STATE_DIR}"
print(f"Found {len(ckpt_files)} R2 fold ckpts:")
for f in ckpt_files:
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")

mel_extractor = MelSpecTransform().to(DEVICE)

models = []
for ckpt_path in ckpt_files:
    fold_id = int(re.search(r"fold(\\d+)", ckpt_path.name).group(1))
    print(f"\\n[Fold {fold_id}] Loading...")
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    print(f"  best_ns22={state.get('best_ns22', float('nan')):.4f}, "
          f"best_macro={state.get('best_macro', float('nan')):.4f}")
    model = BirdSEDModel().to(DEVICE)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    models.append((fold_id, model))
    del state
    gc.collect()
print(f"\\nLoaded {len(models)} fold models on {DEVICE}")
"""),

        code_cell("pseudo_dataset", """# Cell 11: Dataset for XC pseudo gen (5s chunks with stride)
PSEUDO_STRIDE_SEC = 5  # non-overlapping 5s chunks
PSEUDO_MAX_CHUNKS_PER_FILE = 6  # cap chunks per file to limit data size
CHUNK_SAMPLES = SR * TRAIN_DURATION


class XCChunkDataset(Dataset):
    \"\"\"Process XC audio into 5s chunks. Each sample = (wav_chunk, chunk_info).\"\"\"
    def __init__(self, df):
        # Pre-compute chunk index: list of (row_idx, chunk_idx, start_sample)
        # Iterate XC files; for each file, generate chunks based on file length
        self.df = df.reset_index(drop=True)
        self.entries = []  # (row_idx, chunk_idx, start_sample)
        self._build_chunks()

    def _build_chunks(self):
        for ridx, row in self.df.iterrows():
            fp = row["filepath"]
            try:
                # quick read header to get length
                info = sf.info(fp)
                duration_s = info.duration if info.duration else 60
            except Exception:
                duration_s = 60  # fallback
            chunk_stride = SR * PSEUDO_STRIDE_SEC
            n_chunks = int(duration_s / PSEUDO_STRIDE_SEC)
            n_chunks = max(1, min(n_chunks, PSEUDO_MAX_CHUNKS_PER_FILE))
            for ci in range(n_chunks):
                start_sample = ci * chunk_stride
                self.entries.append((ridx, ci, start_sample))

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        ridx, ci, start_sample = self.entries[idx]
        row = self.df.iloc[ridx]
        fp = row["filepath"]
        try:
            wav, sr = sf.read(fp, dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        except Exception:
            wav = np.zeros(CHUNK_SAMPLES, dtype=np.float32)

        # Extract chunk
        end_sample = start_sample + CHUNK_SAMPLES
        if end_sample > len(wav):
            # pad if needed
            wav_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            avail = min(CHUNK_SAMPLES, len(wav) - start_sample)
            if avail > 0:
                wav_chunk[:avail] = wav[start_sample:start_sample + avail]
        else:
            wav_chunk = wav[start_sample:end_sample]

        return torch.from_numpy(wav_chunk).float(), ridx, ci

print(f"Building chunk dataset...")
chunk_ds = XCChunkDataset(filtered)
print(f"  total chunks: {len(chunk_ds)}")
print(f"  files: {len(filtered)}")
print(f"  avg chunks/file: {len(chunk_ds)/len(filtered):.2f}")

# DataLoader
BATCH_PSEUDO = 64
chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_PSEUDO, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
print(f"  batches: {len(chunk_loader)}")
"""),

        code_cell("pseudo_gen", """# Cell 12: Generate per-frame pseudo (5-fold average, fp16)
# Output: list of (T_frames, 234) per chunk, stored as numpy fp16

# Determine T_frames by running 1 batch
with torch.no_grad():
    sample_wav, _, _ = next(iter(chunk_loader))
    sample_wav = sample_wav.to(DEVICE).unsqueeze(1)
    mel = mel_extractor(sample_wav)
    # normalize (matches exp020 R2 inference)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    fold_id, m = models[0]
    _, fw = m(mel, return_framewise=True)
    T_FRAMES = fw.shape[1]
    print(f"T_frames: {T_FRAMES}, C: {fw.shape[2]}")
    del sample_wav, mel, fw

# Allocate output array
N_CHUNKS = len(chunk_ds)
print(f"\\nAllocating pseudo array: ({N_CHUNKS}, {T_FRAMES}, {N_CLASSES}) fp16")
print(f"  size: {N_CHUNKS * T_FRAMES * N_CLASSES * 2 / 1e9:.2f} GB")
pseudo_arr = np.zeros((N_CHUNKS, T_FRAMES, N_CLASSES), dtype=np.float16)
pseudo_index = []  # list of (chunk_global_idx, ridx, ci)

t0 = time.time()
chunk_global = 0
for batch_idx, (wavs, ridxs, cis) in enumerate(chunk_loader):
    wavs = wavs.to(DEVICE, non_blocking=True).unsqueeze(1)  # (B, 1, T)
    with torch.no_grad():
        mel = mel_extractor(wavs)
        mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / (mel.std(dim=(2, 3), keepdim=True) + 1e-6)
        # ensemble 5-fold framewise
        accum = None
        for fold_id, model in models:
            _, fw = model(mel, return_framewise=True)  # (B, T, C)
            fw_prob = torch.sigmoid(fw)  # convert logit → prob
            if accum is None:
                accum = fw_prob
            else:
                accum = accum + fw_prob
        accum = accum / len(models)  # mean
        accum = accum.float().cpu().numpy().astype(np.float16)
    # store
    bsz = accum.shape[0]
    pseudo_arr[chunk_global:chunk_global + bsz] = accum
    for j in range(bsz):
        pseudo_index.append((chunk_global + j, int(ridxs[j]), int(cis[j])))
    chunk_global += bsz
    if (batch_idx + 1) % 50 == 0:
        elapsed = (time.time() - t0) / 60
        eta = elapsed / (batch_idx + 1) * (len(chunk_loader) - batch_idx - 1)
        print(f"  [{batch_idx+1}/{len(chunk_loader)}] elapsed={elapsed:.1f}min eta={eta:.1f}min")

print(f"\\nPseudo gen complete: {(time.time()-t0)/60:.1f} min")
print(f"  chunks processed: {chunk_global}")
print(f"  array shape: {pseudo_arr.shape}, dtype: {pseudo_arr.dtype}")
print(f"  pseudo mean: {pseudo_arr.mean():.4f}, max: {pseudo_arr.max():.4f}")
"""),

        code_cell("save_pseudo", """# Cell 13: Save pseudo + index (compressed npz + csv)
import numpy as np

# Build pseudo_index dataframe
pseudo_index_df = pd.DataFrame(pseudo_index, columns=["chunk_global_idx", "row_idx", "chunk_idx"])
# Join with filtered metadata
pseudo_index_df = pseudo_index_df.merge(
    filtered[["scientific_name", "xc_id", "filepath", "primary_label", "recordist"]].reset_index().rename(columns={"index": "row_idx"}),
    on="row_idx",
    how="left",
)
print(f"pseudo_index_df: {len(pseudo_index_df)} rows")

# Save
PSEUDO_NPZ = DRIVE_ROOT / "xc_pseudo.npz"
PSEUDO_INDEX_CSV = DRIVE_ROOT / "xc_pseudo_index.csv"

np.savez_compressed(PSEUDO_NPZ, pseudo=pseudo_arr)
pseudo_index_df.to_csv(PSEUDO_INDEX_CSV, index=False)

print(f"  Saved: {PSEUDO_NPZ} ({PSEUDO_NPZ.stat().st_size/1e9:.2f} GB)")
print(f"  Saved: {PSEUDO_INDEX_CSV} ({PSEUDO_INDEX_CSV.stat().st_size/1e6:.2f} MB)")
"""),

        code_cell("upload_kaggle", """# Cell 14: Upload to Kaggle Dataset (self-contained auth)
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
SLUG = "birdclef2026-exp058-xc-pseudo"
TITLE = "BirdCLEF2026 exp058 XC pseudo"

UPLOAD_DIR = Path("/content/upload_exp058_pseudo")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(DRIVE_ROOT / "xc_pseudo.npz", UPLOAD_DIR / "xc_pseudo.npz")
shutil.copy(DRIVE_ROOT / "xc_pseudo_index.csv", UPLOAD_DIR / "xc_pseudo_index.csv")
shutil.copy(DRIVE_ROOT / "filtered_metadata.csv", UPLOAD_DIR / "filtered_metadata.csv")
shutil.copy(DRIVE_ROOT / "xc_metadata_refresh.csv", UPLOAD_DIR / "xc_metadata_refresh.csv")

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

try:
    api.dataset_create_new(folder=str(UPLOAD_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset created")
except Exception as e:
    print(f"create_new err: {str(e)[:200]}")
    try:
        api.dataset_create_version(folder=str(UPLOAD_DIR), version_notes="exp058 XC pseudo (5-fold exp020 R2 teacher)",
                                    dir_mode="zip", quiet=False)
        print("OK version created")
    except Exception as e2:
        print(f"create_version err: {str(e2)[:200]}")
print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),

        code_cell("summary", """# Cell 15: summary
print(f"=== exp058 NB1 complete ===")
print(f"Total chunks pseudo'd: {N_CHUNKS}")
print(f"Filtered XC files: {len(filtered)}")
print(f"Species coverage: {filtered['primary_label'].nunique()} / {N_CLASSES}")
print(f"Output dataset: maekeso/{SLUG}")
print(f"\\nNext: NB2 = Combined training (BC2026 hard + XC pseudo)")
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
    print(f"    {c['cell_type']:8s} id={c.get('id'):20s} L={n}")
