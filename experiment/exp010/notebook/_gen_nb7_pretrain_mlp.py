"""Generate exp010 NB7: MLP head pretrain on train_audio + fine-tune on labeled SS.

Stage 1 (pretrain):  265k focal recording windows (single-label) → MLP head × 5 seeds
Stage 2 (fine-tune): 66 labeled SS files (multi-label, 12-window context) → load pretrain → fine-tune

Output: /kaggle/working/mlp_weights/finetuned_seed{S}.pt × 5 (consumed by NB8)

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
    "# exp010 NB7: MLP head pretrain on train_audio + fine-tune on SS\n"
    "\n"
    "**Stage 1**: pretrain MLPHead × 5 seeds on 265k focal recording windows (single-label)\n"
    "**Stage 2**: fine-tune × 5 seeds on 66 labeled SS files (multi-label, 12-window context)\n"
    "\n"
    "Output: `/kaggle/working/mlp_weights/finetuned_seed{S}.pt` × 5\n"
    "consumed by NB8 (inference replaces MLP training with weight loading)."))

cells.append(code_cell("imports", r"""import os, gc, re, time, random, warnings
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
print(f"torch {torch.__version__}, device={DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}, "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)"""))

cells.append(code_cell("config", r"""# CONFIG
SR = 32_000; WINDOW_SEC = 5; N_WINDOWS = 12

BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")
EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")
TAXONOMY_CSV = BASE / "taxonomy.csv"
SC_LABELS_CSV = BASE / "train_soundscapes_labels.csv"

# MLP head config (same as NB4 v10)
MLP_HIDDEN = 256
DROPOUT = 0.1
N_SITES = 32
META_DIM = 8

# Pretrain hyperparams (per-window training on 265k focal windows)
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH = 4096
PRETRAIN_LR = 1e-3
PRETRAIN_WD = 1e-4

# Fine-tune hyperparams (multi-label SS, 12-window context)
# v2: lower LR + fewer epochs + SWA off to prevent catastrophic forgetting of pretrain
FINETUNE_EPOCHS = 30
FINETUNE_BATCH = 64
FINETUNE_LR = 1e-4
FINETUNE_WD = 1e-4

# SWA / KD (mirror NB4)
USE_SWA_PRETRAIN = False
USE_SWA_FINETUNE = False   # v2: SWA off — averaging across fine-tune epochs blurs pretrain signal
SWA_START_FRAC = 0.65
LAMBDA_KD = 0.15
LAMBDA_PRIOR = 0.3
PRIOR_STRENGTH_SITE = 8.0
PRIOR_STRENGTH_HOUR = 8.0
PRIOR_STRENGTH_SH = 4.0

SEEDS = [42, 123, 777, 2024, 9999]

WEIGHTS_DIR = Path("/kaggle/working/mlp_weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

META_PAT = re.compile(r"_S(\d{2})_(\d{8})_(\d{2})\d{4}")

print(f"BASE: {BASE}")
print(f"EMB_DIR: {EMB_DIR}")"""))

cells.append(code_cell("taxonomy", r"""# TAXONOMY
taxonomy = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
print(f"Species: {N_CLASSES}")"""))

# ── Load data ──
LOAD = r"""# ─── Load train_audio embeddings (pretrain data) ───
ta_data = np.load(EMB_DIR / "trainaudio_embeddings.npz")
ta_emb = ta_data["embeddings"].astype(np.float32)         # (N, 1536)
ta_scores = ta_data["scores"].astype(np.float32)          # (N, 234) Perch mapped logits
ta_meta = pd.read_parquet(EMB_DIR / "trainaudio_meta.parquet")
print(f"train_audio: {ta_emb.shape[0]} windows, {ta_meta['filename'].nunique()} files")

# v3: soft target = Perch sigmoid + hard 1 override on primary + secondary positives
# Avoids the "all non-primary species absent" assumption that broke v1/v2.
import ast
train_df = pd.read_csv(BASE / "train.csv")
fname_to_secondary = {}
for _, r in train_df.iterrows():
    sec_raw = r.get("secondary_labels", "")
    sec_list = []
    if isinstance(sec_raw, str) and sec_raw.strip() and sec_raw.strip() != "[]":
        try:
            sec_list = ast.literal_eval(sec_raw)
            if not isinstance(sec_list, list):
                sec_list = []
        except Exception:
            sec_list = [s.strip() for s in str(sec_raw).split(";") if s.strip()]
    fname_to_secondary[str(r["filename"])] = [str(s) for s in sec_list]

# Start from Perch sigmoid as base (soft target), then override known positives to 1
ta_labels = (1.0 / (1.0 + np.exp(-ta_scores))).clip(1e-7, 1 - 1e-7).astype(np.float32)
ta_label_idx = -np.ones(len(ta_meta), dtype=np.int64)
n_secondary_added = 0
for i, row in enumerate(ta_meta.itertuples(index=False)):
    primary = row.primary_label
    if primary not in label_to_idx:
        continue
    j = label_to_idx[primary]
    ta_labels[i, j] = 1.0
    ta_label_idx[i] = j
    fn = row.filename if hasattr(row, "filename") else None
    if fn is not None:
        for s in fname_to_secondary.get(str(fn), []):
            if s in label_to_idx:
                ta_labels[i, label_to_idx[s]] = 1.0
                n_secondary_added += 1

print(f"Secondary positives added: {n_secondary_added}")
mask_valid = ta_label_idx >= 0
ta_emb = ta_emb[mask_valid]
ta_scores = ta_scores[mask_valid]
ta_labels = ta_labels[mask_valid]
ta_label_idx = ta_label_idx[mask_valid]
print(f"After label filter: {ta_emb.shape[0]} windows ({mask_valid.mean()*100:.1f}% valid)")
print(f"Soft target stats: mean={ta_labels.mean():.4f}, max={ta_labels.max():.4f}, "
      f"hard-1 fraction={(ta_labels == 1.0).mean():.6f}")
del ta_data, ta_meta, train_df; gc.collect()

# pos_weight from class distribution in focal
_class_count = np.bincount(ta_label_idx, minlength=N_CLASSES).astype(np.float32)
_class_count = np.maximum(_class_count, 1.0)
_neg_count = ta_emb.shape[0] - _class_count
ta_pos_weight = np.minimum(_neg_count / _class_count, 30.0).astype(np.float32)
print(f"Active classes in focal: {(_class_count > 1).sum()}/{N_CLASSES}")

# v3: target IS Perch sigmoid + hard overrides → no separate KD teacher needed


# ─── Load labeled SS data (fine-tune data) ───
sc_data = np.load(EMB_DIR / "soundscape_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)
sc_scores = sc_data["scores"].astype(np.float32)
sc_meta = pd.read_parquet(EMB_DIR / "soundscape_meta.parquet")

sc_labels_df = pd.read_csv(SC_LABELS_CSV)
labeled_files = set(sc_labels_df["filename"].unique())

label_map = {}
for _, r in sc_labels_df.iterrows():
    fn = r["filename"]
    end_sec = int(pd.Timedelta(r["end"]).total_seconds())
    row_id = f"{Path(fn).stem}_{end_sec}"
    labels_str = str(r["primary_label"]).split(";")
    y = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in labels_str:
        lbl = lbl.strip()
        if lbl in label_to_idx:
            y[label_to_idx[lbl]] = 1.0
    label_map[row_id] = y

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
lab_emb_files, lab_file_list = reshape_to_files(lab_emb_flat, lab_meta)
lab_scores_files, _ = reshape_to_files(sc_scores[is_labeled], lab_meta)
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
print(f"Labeled SS: {lab_emb_files.shape[0]} files")

# Prior tables (computed from labeled set)
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

lab_prior_files = np.stack([compute_prior_logit(s, h) for s, h in zip(lab_site_ids, lab_hours)]).astype(np.float32)
print(f"Prior tables: sites={len(prior_site_ids)}, hours={len(prior_hours)}, sh={len(sh_to_pi)}")

# pos_weight for SS fine-tune
_lab_lbl_t = torch.from_numpy(lab_labels_files)
_pc = _lab_lbl_t.sum(dim=(0, 1)).clamp(min=1)
_nc = _lab_lbl_t.shape[0] * _lab_lbl_t.shape[1] - _pc
ss_pos_weight = (_nc / _pc).clamp(max=30.0).numpy().astype(np.float32)
del _lab_lbl_t

# Perch teacher for SS
ss_teacher = (1.0 / (1.0 + np.exp(-lab_scores_files))).clip(1e-7, 1 - 1e-7).astype(np.float32)

del sc_data, sc_emb, sc_scores, sc_meta; gc.collect()
print(f"\nReady: pretrain={ta_emb.shape[0]} windows, finetune={lab_emb_files.shape[0]} files")"""
cells.append(code_cell("load", LOAD))

# ── MLP Head model ──
MODEL = r"""# === MLP Head (NB4 v10 互換) ===
class MLPHead(nn.Module):
    def __init__(self, d_input, d_hidden, n_classes, dropout=0.1,
                 n_sites=32, meta_dim=8):
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
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None,
                prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb)
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        h = self.mlp(x) * self.temperature
        alpha = torch.sigmoid(self.alpha)
        out = alpha * h + (1 - alpha) * logits
        if prior_logit is not None and lambda_prior > 0:
            out = out + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(out)


print("MLPHead defined.")"""
cells.append(code_cell("model", MODEL))

# ── Stage 1 pretrain ──
PRETRAIN = r"""# ─── Stage 1: PRETRAIN on train_audio (per-window, single-label) ───
# Each train_audio window: shape (1, 1536), label = one-hot of primary_label
# Use BCE+pos_weight + KD vs Perch sigmoid

ta_pos_weight_t = torch.from_numpy(ta_pos_weight).to(DEVICE)


def pretrain_one_seed(seed):
    seed_everything(seed)
    model = MLPHead(d_input=1536, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                    dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM).to(DEVICE)
    opt = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
    sched = CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    swa_model = AveragedModel(model) if USE_SWA_PRETRAIN else None
    swa_start = int(PRETRAIN_EPOCHS * SWA_START_FRAC)
    swa_n = 0

    N = ta_emb.shape[0]
    pw = ta_pos_weight_t.unsqueeze(0).unsqueeze(0)   # (1, 1, C)
    best_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        ep_losses = []
        t_ep = time.time()
        for i in range(0, N, PRETRAIN_BATCH):
            idx = perm[i:i + PRETRAIN_BATCH]
            e   = torch.from_numpy(ta_emb[idx]).unsqueeze(1).to(DEVICE, non_blocking=True)   # (B, 1, 1536)
            lg  = torch.from_numpy(ta_scores[idx]).unsqueeze(1).to(DEVICE, non_blocking=True) # (B, 1, C)
            lbl = torch.from_numpy(ta_labels[idx]).unsqueeze(1).to(DEVICE, non_blocking=True) # (B, 1, C) soft target
            # Site/hour: zero defaults (focal recordings have no site_id)
            st  = torch.zeros(idx.shape[0], dtype=torch.long, device=DEVICE)
            hr  = torch.zeros(idx.shape[0], dtype=torch.long, device=DEVICE)

            out = model(e, lg, site_ids=st, hours=hr, prior_logit=None, lambda_prior=0.0)
            # v3: BCE against soft target (= Perch sigmoid + hard overrides), no separate KD
            loss = (F.binary_cross_entropy(out, lbl, reduction="none") * pw).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
            del e, lg, lbl, st, hr, out, loss
        sched.step()
        ep_loss = float(np.mean(ep_losses))
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [seed {seed}] ep {epoch+1}/{PRETRAIN_EPOCHS}: loss={ep_loss:.4f} ({time.time()-t_ep:.1f}s)")

    model.load_state_dict(best_state)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, best_loss


print(f"Stage 1: pretrain on {ta_emb.shape[0]} focal windows, "
      f"BATCH={PRETRAIN_BATCH}, EPOCHS={PRETRAIN_EPOCHS}, SEEDS={SEEDS}")
pretrain_models = {}
t_all = time.time()
for s in SEEDS:
    t_seed = time.time()
    m, bl = pretrain_one_seed(s)
    pretrain_models[s] = m
    torch.save(m.state_dict(), WEIGHTS_DIR / f"pretrain_seed{s}.pt")
    print(f"PRETRAIN seed {s}: best_loss={bl:.4f}, {time.time()-t_seed:.0f}s")
print(f"\nStage 1 total: {(time.time()-t_all)/60:.1f} min")"""
cells.append(code_cell("pretrain", PRETRAIN))

# ── Stage 2 fine-tune ──
FINETUNE = r"""# ─── Stage 2: FINE-TUNE on labeled SS (12-window context, multi-label) ───
ss_pos_weight_t = torch.from_numpy(ss_pos_weight).to(DEVICE)


def finetune_one_seed(seed, init_state):
    seed_everything(seed)
    model = MLPHead(d_input=1536, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                    dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in init_state.items()})
    opt = AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=FINETUNE_WD)
    sched = CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    swa_model = AveragedModel(model) if USE_SWA_FINETUNE else None
    swa_start = int(FINETUNE_EPOCHS * SWA_START_FRAC)
    swa_n = 0
    pw = ss_pos_weight_t.unsqueeze(0).unsqueeze(0)
    best_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    N = lab_emb_files.shape[0]
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        ep_losses = []
        for i in range(0, N, FINETUNE_BATCH):
            idx = perm[i:i + FINETUNE_BATCH]
            e   = torch.from_numpy(lab_emb_files[idx]).to(DEVICE, non_blocking=True)
            lg  = torch.from_numpy(lab_scores_files[idx]).to(DEVICE, non_blocking=True)
            lbl = torch.from_numpy(lab_labels_files[idx]).to(DEVICE, non_blocking=True)
            st  = torch.from_numpy(lab_site_ids[idx]).long().to(DEVICE, non_blocking=True)
            hr  = torch.from_numpy(lab_hours[idx]).long().to(DEVICE, non_blocking=True)
            pr  = torch.from_numpy(lab_prior_files[idx]).to(DEVICE, non_blocking=True)
            tp  = torch.from_numpy(ss_teacher[idx]).to(DEVICE, non_blocking=True)

            out = model(e, lg, site_ids=st, hours=hr, prior_logit=pr, lambda_prior=LAMBDA_PRIOR)
            bce = F.binary_cross_entropy(out, lbl, reduction="none")
            loss_main = (bce * pw).mean()
            loss_kd = F.binary_cross_entropy(out, tp, reduction="mean")
            loss = loss_main + LAMBDA_KD * loss_kd

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        ep_loss = float(np.mean(ep_losses))
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if USE_SWA_FINETUNE and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1

    if USE_SWA_FINETUNE and swa_n >= 1:
        model.load_state_dict(swa_model.module.state_dict())
    else:
        model.load_state_dict(best_state)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, best_loss


print(f"\nStage 2: fine-tune on {lab_emb_files.shape[0]} labeled SS files, "
      f"EPOCHS={FINETUNE_EPOCHS}, LR={FINETUNE_LR}")
t_all = time.time()
for s in SEEDS:
    t_seed = time.time()
    init_state = {k: v.clone() for k, v in pretrain_models[s].state_dict().items()}
    m, bl = finetune_one_seed(s, init_state)
    torch.save(m.state_dict(), WEIGHTS_DIR / f"finetuned_seed{s}.pt")
    print(f"FINETUNE seed {s}: best_loss={bl:.4f}, {time.time()-t_seed:.0f}s")
print(f"\nStage 2 total: {(time.time()-t_all)/60:.1f} min")"""
cells.append(code_cell("finetune", FINETUNE))

# ── Verify ──
VERIFY = r"""# ─── Verify saved weights ───
saved = sorted(WEIGHTS_DIR.glob("*.pt"))
print(f"Total weight files: {len(saved)}")
for p in saved:
    print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")"""
cells.append(code_cell("verify", VERIFY))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb7_pretrain_mlp.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
