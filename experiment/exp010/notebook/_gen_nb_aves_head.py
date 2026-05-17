"""Generate exp010 AVES head training NB.

Trains MLP head × 5 seeds on AVES (wav2vec2-base) embeddings + labeled SS 66.
Also saves wav2vec2-base model files for blend NB reuse.

Output:
- /kaggle/working/aves_head_seed{42,123,777,2024,9999}.pt
- /kaggle/working/wav2vec2-base/  (HF model snapshot for offline use)

Runs on T4 GPU, ~30-60 minutes.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from _gen_nb4_blend import code_cell, md_cell

cells = []

cells.append(md_cell("hdr",
    "# exp010 AVES head training\n"
    "\n"
    "Train MLP head × 5 seeds on AVES (wav2vec2-base, 768d) embeddings + labeled SS 66 files.\n"
    "Also caches the HuggingFace wav2vec2-base model for offline use in blend NB.\n"
    "\n"
    "Inputs: maekeso/nb1f-aves-embed (AVES embeddings), birdclef-2026 (labels)\n"
    "Output: head weights + model snapshot, consumed by blend NB v5+ (4-way blend)."))

cells.append(code_cell("install",
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'transformers'], check=False)\n"
    "print('Install attempted')"))

cells.append(code_cell("imports", r"""import os, gc, re, time, random, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__}, device={DEVICE}")

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)"""))

cells.append(code_cell("config", r"""# CONFIG
SR_BC = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12

BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")
TAXONOMY_CSV = BASE / "taxonomy.csv"
SC_LABELS_CSV = BASE / "train_soundscapes_labels.csv"

# AVES embeddings (from NB1f)
AVES_DIR = None
for cand in [
    Path("/kaggle/input/datasets/maekeso/nb1f-aves-embed"),
    Path("/kaggle/input/nb1f-aves-embed"),
]:
    if cand.exists():
        AVES_DIR = cand
        break
if AVES_DIR is None:
    for p in Path("/kaggle/input").rglob("aves_sc_embeddings.npz"):
        AVES_DIR = p.parent
        break
assert AVES_DIR is not None, "AVES dataset not attached"
print(f"AVES_DIR: {AVES_DIR}")

# Hyperparams (mirror NB4 head training)
EMB_DIM_AVES = 768
MLP_HIDDEN = 256
DROPOUT = 0.1
N_SITES = 32
META_DIM = 8
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
USE_SWA = True
SWA_START_FRAC = 0.65
LAMBDA_KD = 0.15
LAMBDA_PRIOR = 0.3
PRIOR_STRENGTH_SITE = 8.0
PRIOR_STRENGTH_HOUR = 8.0
PRIOR_STRENGTH_SH = 4.0
SEEDS = [42, 123, 777, 2024, 9999]

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PAT = re.compile(r"_S(\d{2})_(\d{8})_(\d{2})\d{4}")

print(f"BASE={BASE}, AVES_DIR={AVES_DIR}")"""))

cells.append(code_cell("taxonomy", r"""taxonomy = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
print(f"Species: {N_CLASSES}")"""))

cells.append(code_cell("load", r"""# Load AVES SS embeddings + meta
sc_data = np.load(AVES_DIR / "aves_sc_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)
sc_meta = pd.read_parquet(AVES_DIR / "aves_sc_meta.parquet")
print(f"AVES SS: {sc_emb.shape}, meta={sc_meta.shape}")

# Pseudo-Perch logits (zeros — no Perch teacher for AVES head)
# We could load actual Perch logits from NB1 for KD, but keep it simple
sc_scores = np.zeros((sc_emb.shape[0], N_CLASSES), dtype=np.float32)

sc_labels_df = pd.read_csv(SC_LABELS_CSV)
labeled_files = set(sc_labels_df["filename"].unique())
label_map = {}
for _, r in sc_labels_df.iterrows():
    fn = r["filename"]
    end_sec = int(pd.Timedelta(r["end"]).total_seconds())
    rid = f"{Path(fn).stem}_{end_sec}"
    labels_str = str(r["primary_label"]).split(";")
    y = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in labels_str:
        lbl = lbl.strip()
        if lbl in label_to_idx:
            y[label_to_idx[lbl]] = 1.0
    label_map[rid] = y

is_labeled = sc_meta["filename"].isin(labeled_files).values

def reshape_to_files(arr, meta):
    fnames = meta["filename"].values
    unique = list(dict.fromkeys(fnames))
    n_files = len(unique)
    D = arr.shape[1]
    out = np.zeros((n_files, N_WINDOWS, D), dtype=arr.dtype)
    file_to_idx = {f: i for i, f in enumerate(unique)}
    counters = np.zeros(n_files, dtype=int)
    for ri, fn in enumerate(fnames):
        fi = file_to_idx[fn]
        wi = counters[fi]
        if wi < N_WINDOWS:
            out[fi, wi] = arr[ri]
            counters[fi] += 1
    return out, unique


def parse_meta(fname):
    m = META_PAT.search(fname)
    if m is None:
        return 0, 0
    return int(m.group(1)), int(m.group(3))


lab_meta = sc_meta[is_labeled].reset_index(drop=True)
lab_emb_flat = sc_emb[is_labeled]
lab_scores_flat = sc_scores[is_labeled]
lab_emb_files, lab_file_list = reshape_to_files(lab_emb_flat, lab_meta)
lab_scores_files, _ = reshape_to_files(lab_scores_flat, lab_meta)
lab_site_ids = np.array([parse_meta(fn)[0] for fn in lab_file_list], dtype=np.int64)
lab_hours = np.array([parse_meta(fn)[1] for fn in lab_file_list], dtype=np.int64)
lab_labels_files = np.zeros((len(lab_file_list), N_WINDOWS, N_CLASSES), dtype=np.float32)
for fi, fn in enumerate(lab_file_list):
    stem = Path(fn).stem
    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        rid = f"{stem}_{end_sec}"
        if rid in label_map:
            lab_labels_files[fi, wi] = label_map[rid]
print(f"Labeled: {lab_emb_files.shape[0]} files, {lab_emb_files.shape}")

# Prior tables
file_labels = (lab_labels_files.sum(axis=1) > 0).astype(np.float32)
global_p = file_labels.mean(axis=0).astype(np.float32)

prior_site_ids = sorted(set(int(s) for s in lab_site_ids))
site_to_pi = {s: i for i, s in enumerate(prior_site_ids)}
site_n = np.zeros(len(prior_site_ids), dtype=np.float32)
site_p = np.zeros((len(prior_site_ids), N_CLASSES), dtype=np.float32)
for s in prior_site_ids:
    m = (lab_site_ids == s)
    site_n[site_to_pi[s]] = m.sum()
    site_p[site_to_pi[s]] = file_labels[m].mean(axis=0)

prior_hours = sorted(set(int(h) for h in lab_hours))
hour_to_pi = {h: i for i, h in enumerate(prior_hours)}
hour_n = np.zeros(len(prior_hours), dtype=np.float32)
hour_p = np.zeros((len(prior_hours), N_CLASSES), dtype=np.float32)
for h in prior_hours:
    m = (lab_hours == h)
    hour_n[hour_to_pi[h]] = m.sum()
    hour_p[hour_to_pi[h]] = file_labels[m].mean(axis=0)

sh_to_pi = {}; sh_n_list, sh_p_list = [], []
for s in prior_site_ids:
    for h in prior_hours:
        m = (lab_site_ids == s) & (lab_hours == h)
        if m.sum() > 0:
            sh_to_pi[(s, h)] = len(sh_n_list)
            sh_n_list.append(float(m.sum()))
            sh_p_list.append(file_labels[m].mean(axis=0))
sh_n = np.array(sh_n_list, dtype=np.float32) if sh_n_list else np.zeros(0, dtype=np.float32)
sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, N_CLASSES), dtype=np.float32)


def compute_prior_logit(site_id, hour, eps=1e-4):
    p = global_p.astype(np.float32).copy()
    h_i = hour_to_pi.get(int(hour), -1)
    if h_i >= 0:
        nh = hour_n[h_i]; wh = nh / (nh + PRIOR_STRENGTH_HOUR)
        p = wh * hour_p[h_i] + (1 - wh) * p
    s_i = site_to_pi.get(int(site_id), -1)
    if s_i >= 0:
        ns = site_n[s_i]; ws = ns / (ns + PRIOR_STRENGTH_SITE)
        p = ws * site_p[s_i] + (1 - ws) * p
    sh_i = sh_to_pi.get((int(site_id), int(hour)), -1)
    if sh_i >= 0:
        nsh = sh_n[sh_i]; wsh = nsh / (nsh + PRIOR_STRENGTH_SH)
        p = wsh * sh_p[sh_i] + (1 - wsh) * p
    p = np.clip(p, eps, 1 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


lab_prior_files = np.stack(
    [compute_prior_logit(s, h) for s, h in zip(lab_site_ids, lab_hours)]
).astype(np.float32)
print(f"Prior tables: sites={len(prior_site_ids)}, hours={len(prior_hours)}, sh={len(sh_to_pi)}")"""))

cells.append(code_cell("model", r"""# AVES MLP head (mirror NB4 MLPHead but d_input=768)
class AVESHead(nn.Module):
    def __init__(self, d_input=EMB_DIM_AVES, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                 dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # No alpha blending here — AVES has no Perch logit, raw head output
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, emb, site_ids=None, hours=None,
                prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb)
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        h = self.mlp(x) * self.temperature + self.bias
        if prior_logit is not None and lambda_prior > 0:
            h = h + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(h)


print("AVESHead defined.")"""))

cells.append(code_cell("train", r"""# Train AVESHead × 5 seeds
train_emb = torch.tensor(lab_emb_files, dtype=torch.float32).to(DEVICE)
train_labels = torch.tensor(lab_labels_files, dtype=torch.float32).to(DEVICE)
train_site = torch.tensor(lab_site_ids, dtype=torch.long).to(DEVICE)
train_hour = torch.tensor(lab_hours, dtype=torch.long).to(DEVICE)
train_prior = torch.tensor(lab_prior_files, dtype=torch.float32).to(DEVICE)

pos_counts = train_labels.sum(dim=(0, 1)).clamp(min=1)
neg_counts = train_labels.shape[0] * train_labels.shape[1] - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(max=30.0)


def train_one(seed):
    seed_everything(seed)
    model = AVESHead().to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS)
    swa_model = AveragedModel(model) if USE_SWA else None
    swa_start = int(N_EPOCHS * SWA_START_FRAC); swa_n = 0
    best_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(N_EPOCHS):
        model.train()
        out = model(train_emb, site_ids=train_site, hours=train_hour,
                    prior_logit=train_prior, lambda_prior=LAMBDA_PRIOR)
        bce = F.binary_cross_entropy(out, train_labels, reduction="none")
        loss = (bce * pos_weight.unsqueeze(0).unsqueeze(0)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if USE_SWA and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1
    if USE_SWA and swa_n >= 1:
        model.load_state_dict(swa_model.module.state_dict())
    else:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


t0 = time.time()
for seed in SEEDS:
    t_seed = time.time()
    m, bl = train_one(seed)
    torch.save(m.state_dict(), OUT_DIR / f"aves_head_seed{seed}.pt")
    print(f"  seed {seed}: best_loss={bl:.4f}, {time.time()-t_seed:.0f}s")
print(f"Total train: {time.time()-t0:.0f}s")"""))

cells.append(code_cell("save-aves-model", r"""# Cache wav2vec2-base model for offline use in blend NB
from transformers import AutoModel, AutoFeatureExtractor

# Try real AVES first, fallback to wav2vec2-base
fallbacks = ["earthspecies/aves-base-bio", "facebook/wav2vec2-base"]
loaded_name = None
for name in fallbacks:
    try:
        fe = AutoFeatureExtractor.from_pretrained(name)
        m = AutoModel.from_pretrained(name)
        loaded_name = name
        print(f"Caching: {name}, SR={fe.sampling_rate}")
        break
    except Exception as e:
        print(f"  {name} failed: {type(e).__name__}")

assert loaded_name is not None, "no aves model loadable"

local_dir = OUT_DIR / "aves_model"
local_dir.mkdir(parents=True, exist_ok=True)
m.save_pretrained(str(local_dir))
fe.save_pretrained(str(local_dir))
# Save metadata for blend NB
meta_info = {"model_name": loaded_name, "sample_rate": fe.sampling_rate}
(local_dir / "_aves_meta.json").write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
print(f"Saved aves model to {local_dir}")
print(f"Files:")
for p in sorted(local_dir.iterdir()):
    print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")"""))

cells.append(code_cell("summary", r"""# Verify outputs
print("\nOutput files:")
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")
    elif p.is_dir():
        print(f"  {p.name}/  ({sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/1e6:.1f} MB)")"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = HERE / "nb_aves_head.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out} ({len(cells)} cells)")
