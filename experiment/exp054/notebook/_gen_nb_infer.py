"""Generate Stage 2 v2 inference NB for Kaggle CPU.

Inputs:
  - Stage 2 v2 ckpt: maekeso/birdclef2026-exp054-stage2-effv2s
  - birdclef-2026 competition data (test_soundscapes)

Output: submission.csv (TopN N=1 PP applied)
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp054\notebook\nb_infer.ipynb")

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
        md_cell("hdr", """# exp054 Stage 2 v2 Inference (Kaggle CPU)

**Purpose**: Stage 2 v2 ckpt で test_soundscapes 推論 → submission.csv

**Input**:
- `maekeso/birdclef2026-exp054-stage2-effv2s` (Stage 2 v2 ckpt)
- birdclef-2026 competition (test_soundscapes auto-mounted)

**Output**: submission.csv with TopN N=1 PP (paper 256 spec)

**Pipeline**:
1. Load Stage 2 v2 model (EffNetV2-S + SED head)
2. Read each test_soundscape file (60s)
3. Chunk to 12 × 5s segments
4. Forward pass → sigmoid probabilities
5. TopN N=1 PP scaling
6. Build submission.csv

**Expected runtime**: 60-70 min (Kaggle CPU 90 min cap)
**Expected LB**: 0.82-0.88 (standalone)
"""),

        code_cell("setup", """# Cell 1: imports + paths
import os, sys, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import librosa
import timm
from tqdm.auto import tqdm

# Paths
CKPT_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp054-stage2-effv2s/stage2_v2_best.pth"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp054-stage2-effv2s/stage2_v2_best.pth"),
]
CKPT_PATH = next((p for p in CKPT_CANDIDATES if p.exists()), None)
assert CKPT_PATH is not None, f"Stage 2 v2 ckpt not found: {CKPT_CANDIDATES}"
print(f"CKPT: {CKPT_PATH}")

TEST_CANDIDATES = [
    Path("/kaggle/input/birdclef-2026/test_soundscapes"),
    Path("/kaggle/input/competitions/birdclef-2026/test_soundscapes"),
]
TEST_DIR = next((p for p in TEST_CANDIDATES if p.exists()), None)
print(f"TEST_DIR: {TEST_DIR}")

# Sample submission for column order
SAMPLE_SUB_CANDIDATES = [
    Path("/kaggle/input/birdclef-2026/sample_submission.csv"),
    Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv"),
]
SAMPLE_SUB = next((p for p in SAMPLE_SUB_CANDIDATES if p.exists()), None)
sample = pd.read_csv(SAMPLE_SUB)
print(f"Sample sub: {len(sample)} rows, {len(sample.columns)} cols (incl row_id)")

DEVICE = torch.device("cpu")
torch.set_num_threads(4)
"""),

        code_cell("species", f"""# Cell 2: BC2026 species list (234)
__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
PRIMARY_LABELS = species_df["primary_label"].tolist()
N_CLASSES = len(PRIMARY_LABELS)
print(f"BC2026: {{N_CLASSES}} species")
"""),

        code_cell("model", """# Cell 3: Model definition + ckpt load
SR = 32000
CHUNK_LEN = SR * 5  # 5s
N_WINDOWS = 12      # 60s / 5s
N_MELS = 128

class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=N_MELS, f_min=20, f_max=16000,
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
            "tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, in_chans=3,
            num_classes=0, global_pool="",
            drop_path_rate=0.0,    # eval: no stochastic depth
        )
        self.head = SEDHead(self.backbone.num_features, N_CLASSES)
    def forward(self, mel):
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x).mean(dim=2).transpose(1, 2)
        return self.head(feat)

# Load ckpt
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = Model()
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
mel_extractor = MelExtractor()
print(f"Stage 2 v2 ckpt loaded:")
print(f"  val_ns22: {ckpt.get('val_ns22', '?')}")
print(f"  val_macro: {ckpt.get('val_macro', '?')}")
print(f"  ep: {ckpt.get('ep', '?')}")
print(f"  Stage 1 v2 parent val_ns22: {ckpt.get('stage1_v2_val_ns22', '?')}")
print(f"\\nModel params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
"""),

        code_cell("test_files", """# Cell 4: Enumerate test files
if TEST_DIR is None or not TEST_DIR.exists():
    print("⚠ test_soundscapes not mounted, fallback to empty (Save Version mode)")
    test_files = []
else:
    test_files = sorted(TEST_DIR.glob("*.ogg"))
    if not test_files:
        # Try other formats
        test_files = sorted(TEST_DIR.glob("*.wav"))
    print(f"Test files: {len(test_files)}")
    if test_files:
        print(f"  Sample: {test_files[0].name}")
"""),

        code_cell("infer_fn", """# Cell 5: Inference functions
@torch.no_grad()
def infer_file(file_path):
    \"\"\"Read a test file, chunk to 12×5s, predict probabilities.

    Returns: (N_WINDOWS, N_CLASSES) array of sigmoid probabilities.
    \"\"\"
    try:
        wav, sr = sf.read(str(file_path), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    except Exception as e:
        print(f"  [read err] {file_path.name}: {str(e)[:80]}")
        return np.full((N_WINDOWS, N_CLASSES), 0.0, dtype=np.float32)

    # Pad / chunk to N_WINDOWS × CHUNK_LEN
    target_len = N_WINDOWS * CHUNK_LEN
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    chunks = wav[:target_len].reshape(N_WINDOWS, CHUNK_LEN)

    # Batch inference
    batch = torch.from_numpy(chunks).float()  # (N_WINDOWS, CHUNK_LEN)
    mel = mel_extractor(batch)                # (N_WINDOWS, n_mels, T)
    logit = model(mel)                        # (N_WINDOWS, N_CLASSES)
    prob = torch.sigmoid(logit).numpy()
    return prob
"""),

        code_cell("infer_loop", """# Cell 6: Run inference on all test files
if not test_files:
    print("No test files, skip inference (Save Version mode)")
    all_rows = []
else:
    all_rows = []
    t0 = time.time()
    for fi, fp in enumerate(test_files):
        probs = infer_file(fp)   # (N_WINDOWS, N_CLASSES)
        stem = fp.stem
        for wi in range(N_WINDOWS):
            end_sec = (wi + 1) * 5
            row_id = f"{stem}_{end_sec}"
            row = {"row_id": row_id}
            for ci, lbl in enumerate(PRIMARY_LABELS):
                row[lbl] = float(probs[wi, ci])
            all_rows.append(row)
        if (fi + 1) % 50 == 0 or fi == len(test_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(test_files) - fi - 1) / rate
            print(f"  [{fi+1}/{len(test_files)}] {rate*60:.1f}/min ETA {eta/60:.1f}min")
    print(f"\\nInference done in {(time.time()-t0)/60:.1f}min")
"""),

        code_cell("pp_topn", """# Cell 7: TopN N=1 PP (paper 256 spec, exp048 と同)
cols = ["row_id"] + PRIMARY_LABELS
if all_rows:
    sub_df = pd.DataFrame(all_rows)[cols]
else:
    # Save Version: use sample_submission as template
    sub_df = sample.copy()

print(f"Submission shape: {sub_df.shape}")

if all_rows:
    # TopN N=1 PP: per-file × per-class top-K max scaling
    FCS_TOP_K = 1
    FCS_POWER = 1.0

    preds_array = sub_df[PRIMARY_LABELS].values.astype(np.float32)
    _n, _c = preds_array.shape
    _view = preds_array.reshape(-1, N_WINDOWS, _c)
    _sorted = np.sort(_view, axis=1)
    _topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
    _scale = np.power(_topk_mean, FCS_POWER)
    preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)

    sub_df[PRIMARY_LABELS] = preds_array
    print(f"TopN N=1 PP applied (K={FCS_TOP_K}, POW={FCS_POWER})")
    print(f"  pred mean: {preds_array.mean():.4f}, max: {preds_array.max():.4f}")
"""),

        code_cell("save", """# Cell 8: Save submission.csv
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print(f"Saved {out_path}")
print(f"  shape: {sub_df.shape}")
print(f"  head:")
print(sub_df.head(3).to_string())
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
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
