"""Generate Kaggle inference NB for Stage 2A model.

Environment: Kaggle Notebook CPU (Kernels Only, no internet)
Inputs: birdclef-2026 competition + birdclef2026-exp053-stage2a-effv2s Dataset
Output: submission.csv

Approach: pytorch CPU inference with batching (Stage 2A is 22M params, fits CPU)
PP: TopN N=1 (paper 256 spec, also exp048 0.950 で実証)
"""
import json
from pathlib import Path

OUT_DIR = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp053\notebook")
OUT = OUT_DIR / "nb_infer.ipynb"

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

all_species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species.txt").read_text(encoding="utf-8")
all_species_lines = []
for line in all_species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        all_species_lines.append("    " + s)
all_species_block = "\n".join(all_species_lines)

nb = {
    "cells": [
        md_cell("hdr", """# exp053 Inference NB (Kaggle CPU): Stage 2A → submission.csv

**Input**:
- birdclef-2026 (competition)
- maekeso/birdclef2026-exp053-stage2a-effv2s (Stage 2A ckpt)

**Output**: submission.csv

**Approach**:
- pytorch CPU inference (no ONNX、Stage 2A 22M params で CPU 内収まる)
- batch 32 chunks 同時推論
- TopN N=1 PP (paper 256 spec、exp048 0.950 で実証)

**Expected runtime**: 40-60 min (CPU 90 min 制限内)
**Expected LB**: 0.88-0.92 standalone
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup
# ============================================================
import os, sys, json, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
import timm
from tqdm.auto import tqdm

print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()} (expected False for submission NB)")
DEVICE = "cpu"

torch.set_num_threads(4)
torch.set_grad_enabled(False)
print(f"torch.num_threads: {torch.get_num_threads()}")

# Paths
COMP_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_ROOT.exists():
    COMP_ROOT = Path("/kaggle/input/birdclef-2026")
assert COMP_ROOT.exists(), f"BC2026 not mounted at {COMP_ROOT}"

# Stage 2A ckpt - try multiple paths
CKPT_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp053-stage2a-effv2s/stage2a_best.pth"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp053-stage2a-effv2s/stage2a_best.pth"),
]
CKPT_PATH = next((p for p in CKPT_CANDIDATES if p.exists()), None)
assert CKPT_PATH is not None, f"Stage 2A ckpt not found in: {CKPT_CANDIDATES}"
print(f"Stage 2A ckpt: {CKPT_PATH} ({CKPT_PATH.stat().st_size/1e6:.1f} MB)")

TEST_DIR = COMP_ROOT / "test_soundscapes"
print(f"test_soundscapes: {TEST_DIR}")
"""),

        code_cell("species", """# ============================================================
# Cell 2: 234 species list (embedded)
# ============================================================
__BC2026_SPECIES = [
__ALL_SPECIES_PLACEHOLDER__
]
PRIMARY_LABELS = [r[1] for r in __BC2026_SPECIES]
N_CLASSES = len(PRIMARY_LABELS)
print(f"BC2026 species: {N_CLASSES}")
"""),

        code_cell("config", """# ============================================================
# Cell 3: Config
# ============================================================
class CFG:
    BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"
    N_CLASSES = N_CLASSES

    SR = 32000
    CHUNK_SEC = 5
    CHUNK_LEN = SR * CHUNK_SEC  # 160000

    N_MELS = 128
    N_FFT = 2048
    HOP = 512
    FMIN = 20
    FMAX = 16000

    BATCH_SIZE = 32  # CPU batch

    # PP
    FCS_TOP_K = 1     # paper 256 spec
    FCS_POWER = 1.0   # exp048 0.950 で実証

print(f"Mel: {CFG.N_MELS}×T, {CFG.SR}Hz")
print(f"PP: TopN N=1, POWER=1.0 (paper 256)")
"""),

        code_cell("model", """# ============================================================
# Cell 4: Model definition + load Stage 2A ckpt
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
        return clipwise


class SEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.BACKBONE, pretrained=False, in_chans=3,
            num_classes=0, global_pool="",
        )
        feat_dim = self.backbone.num_features
        self.head = SEDHead(feat_dim, CFG.N_CLASSES)

    def forward(self, mel):
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x)
        feat = feat.mean(dim=2)
        feat = feat.transpose(1, 2)
        return self.head(feat)


# Load Stage 2A ckpt
model = SEDModel()
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
# Remove "module." prefix if DataParallel was used
state = {k.replace("module.", ""): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
print(f"Stage 2A val_ns22: {ckpt.get('val_ns22', 'n/a')}")
print(f"Stage 2A val_macro: {ckpt.get('val_macro', 'n/a')}")
model.eval()
model = model.to(DEVICE)
mel_extractor = MelExtractor().to(DEVICE)
print(f"Model loaded, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
"""),

        code_cell("test_files", """# ============================================================
# Cell 5: Discover test_soundscapes
# ============================================================
test_files = sorted(list(TEST_DIR.glob("*.ogg")))
print(f"Test soundscapes: {len(test_files)}")
if test_files:
    print(f"  Sample: {test_files[0].name}")
    # Check duration of first file
    info = sf.info(str(test_files[0]))
    print(f"  Duration: {info.duration:.1f}s, SR: {info.samplerate}, Channels: {info.channels}")
"""),

        code_cell("infer_fn", """# ============================================================
# Cell 6: Inference function (per file, returns per-segment predictions)
# ============================================================
@torch.no_grad()
def predict_file(file_path, model, mel_ex):
    \"\"\"Process one test_soundscape file → list of (row_id, 234 probs)\"\"\"
    file_id = Path(file_path).stem

    # Load audio
    try:
        wav, sr = sf.read(str(file_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != CFG.SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=CFG.SR)
    except Exception as e:
        print(f"  [read err] {file_path}: {str(e)[:80]}")
        return []

    # Segment into 5s chunks (BC2026 standard: 60s file → 12 segments)
    n_full_chunks = len(wav) // CFG.CHUNK_LEN
    if n_full_chunks == 0:
        # Pad to single chunk
        wav = np.pad(wav, (0, CFG.CHUNK_LEN - len(wav)))
        n_full_chunks = 1
        chunks = [wav]
    else:
        chunks = [wav[i*CFG.CHUNK_LEN:(i+1)*CFG.CHUNK_LEN] for i in range(n_full_chunks)]

    # Build batch tensor
    batch_wav = torch.from_numpy(np.stack(chunks)).float()

    # Batched inference (handle large batch via chunking)
    all_preds = []
    for i in range(0, len(batch_wav), CFG.BATCH_SIZE):
        sub_batch = batch_wav[i:i+CFG.BATCH_SIZE]
        mel = mel_ex(sub_batch)
        logit = model(mel)
        prob = torch.sigmoid(logit).numpy()
        all_preds.append(prob)
    probs = np.concatenate(all_preds, axis=0)  # (n_chunks, 234)

    # Build row_ids: {file_id}_{end_time_seconds}
    rows = []
    for i, p in enumerate(probs):
        end_sec = (i + 1) * CFG.CHUNK_SEC
        row_id = f"{file_id}_{end_sec}"
        row = {"row_id": row_id}
        for j, lbl in enumerate(PRIMARY_LABELS):
            row[lbl] = float(p[j])
        rows.append(row)
    return rows
"""),

        code_cell("infer_loop", """# ============================================================
# Cell 7: Run inference on all test files
# ============================================================
print(f"Starting inference on {len(test_files)} files...")
start_t = time.time()

all_rows = []
for i, f in enumerate(tqdm(test_files, desc="infer")):
    rows = predict_file(f, model, mel_extractor)
    all_rows.extend(rows)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_t
        rate = (i + 1) / elapsed * 60
        eta = (len(test_files) - i - 1) / rate * 60
        print(f"  [{i+1}/{len(test_files)}] {len(all_rows)} rows, {rate:.1f}/min, ETA {eta/60:.1f}min")

elapsed = time.time() - start_t
print(f"\\nInference DONE in {elapsed/60:.1f}min")
print(f"Total rows: {len(all_rows)}")

# Build DataFrame
sub_df = pd.DataFrame(all_rows)
print(f"\\nsub_df shape: {sub_df.shape}")
print(f"  columns: {sub_df.columns[:5].tolist()} ... {sub_df.columns[-3:].tolist()}")
"""),

        code_cell("pp", """# ============================================================
# Cell 8: PP (TopN N=1, paper 256 spec)
# ============================================================
print(f"Applying TopN N=1 PP (FCS_TOP_K={CFG.FCS_TOP_K}, FCS_POWER={CFG.FCS_POWER})...")

# Extract file_id from row_id
sub_df["_file_id"] = sub_df["row_id"].apply(lambda x: x.rsplit("_", 1)[0])

# For each file × class: compute top-K max → scale
prob_cols = [c for c in sub_df.columns if c in PRIMARY_LABELS]
preds_array = sub_df[prob_cols].values.astype(np.float32)
file_ids = sub_df["_file_id"].values

unique_files = pd.unique(file_ids)
file_to_idx = {f: i for i, f in enumerate(unique_files)}
file_idx_arr = np.array([file_to_idx[f] for f in file_ids])

# For each class, compute per-file top-K mean
n = len(preds_array)
c = preds_array.shape[1]

# group by file, find top-K per class
for f_idx in range(len(unique_files)):
    mask = (file_idx_arr == f_idx)
    file_preds = preds_array[mask]  # (12, 234) typically
    # Top-K per class
    if CFG.FCS_TOP_K == 1:
        file_max = file_preds.max(axis=0)  # (234,)
    else:
        sorted_preds = np.sort(file_preds, axis=0)
        file_max = sorted_preds[-CFG.FCS_TOP_K:].mean(axis=0)
    scale = np.power(file_max, CFG.FCS_POWER)
    preds_array[mask] = file_preds * scale[np.newaxis, :]

# Write back
for j, col in enumerate(prob_cols):
    sub_df[col] = preds_array[:, j]

sub_df = sub_df.drop(columns=["_file_id"])
print(f"PP done. preds range: [{preds_array.min():.4f}, {preds_array.max():.4f}]")
"""),

        code_cell("save", """# ============================================================
# Cell 9: Save submission.csv
# ============================================================
# Ensure column order: row_id + 234 species
sample_sub_path = COMP_ROOT / "sample_submission.csv"
if sample_sub_path.exists():
    sample = pd.read_csv(sample_sub_path)
    expected_cols = sample.columns.tolist()
    # Reorder sub_df to match
    sub_df = sub_df[expected_cols]
    print(f"Column order matched to sample_submission.csv ({len(expected_cols)} cols)")

OUT_PATH = Path("/kaggle/working/submission.csv")
sub_df.to_csv(OUT_PATH, index=False)
print(f"\\nSaved: {OUT_PATH}")
print(f"  shape: {sub_df.shape}")
print(f"  size: {OUT_PATH.stat().st_size/1e6:.1f} MB")
print(f"\\nFirst 3 rows:")
print(sub_df.head(3).iloc[:, :6])
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
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
