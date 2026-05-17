"""Generate exp010 NB5 Colab edition: Noisy Student training on A100.

Differs from Kaggle NB5:
- Runs on Colab A100 (DEVICE='cuda', BATCH_FILES=512)
- Pulls NB1 outputs + BC2026 small files via Kaggle API
- Saves weights to Google Drive (mounted)
- Final zip for manual upload to Kaggle Dataset

Output notebook: nb5_colab.ipynb (open directly in Colab).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from _gen_nb4_blend import code_cell, md_cell, MODEL

cells = []

cells.append(md_cell("hdr",
    "# exp010 NB5 (Colab edition): Noisy Student Multi-Round Training\n"
    "\n"
    "Train ProtoSSM + MLP head over 5 rounds of pseudo-labeling on the 10,592 unlabeled\n"
    "soundscapes, using A100 GPU on Colab Pro.\n"
    "\n"
    "**Prerequisite**:\n"
    "- Colab A100 runtime selected (Runtime → Change runtime type → A100 GPU)\n"
    "- Kaggle API token (`kaggle.json` or `KGAT_` Bearer token) ready\n"
    "\n"
    "Output: `~/ns_weights/round{r}_*.pt` × 50 files. Saved to Google Drive at\n"
    "`MyDrive/birdclef2026/ns_weights/`. Upload to Kaggle Dataset as\n"
    "`maekeso/birdclef2026-exp010-ns-weights` for NB6 to consume."))

# ── Cell: Kaggle credentials ──
KAGGLE_AUTH = r"""# ─── Kaggle credentials ───
# Option A: paste KGAT_ token directly (this project uses KGAT_ Bearer in kaggle.json's "key" field)
KAGGLE_API_TOKEN = "PASTE_KGAT_TOKEN_HERE"   # ← e.g. "KGAT_xxxxxxxxxxxxxxxx"

# Option B: upload kaggle.json from local machine
# from google.colab import files
# files.upload()  # select kaggle.json
# !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

import os, json
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
if KAGGLE_API_TOKEN.startswith("KGAT_"):
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    # Write a stub kaggle.json with the token in 'key' so SDK is happy
    kj_path = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kj_path, "w") as f:
        json.dump({"username": "maekeso", "key": KAGGLE_API_TOKEN}, f)
    os.chmod(kj_path, 0o600)
print("Kaggle auth ready.")"""
cells.append(code_cell("kaggle-auth", KAGGLE_AUTH))

# ── Cell: Install + Download data ──
DOWNLOAD = r"""# ─── Install + download data from Kaggle ───
!pip install -q kaggle

from pathlib import Path
NB1_DIR = Path("/content/nb1_output")
BC_DIR  = Path("/content/bc2026")
NB1_DIR.mkdir(parents=True, exist_ok=True)
BC_DIR.mkdir(parents=True, exist_ok=True)

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# NB1 output (Perch embeddings of all SS files)
print("Downloading NB1 output...")
api.kernels_output("maekeso/birdclef2026-exp010-nb1-embedding", path=str(NB1_DIR))

# BC2026 small files
print("Downloading BC2026 metadata...")
for fname in ["train_soundscapes_labels.csv", "taxonomy.csv"]:
    api.competition_download_file("birdclef-2026", fname, path=str(BC_DIR))

# Unzip if needed
import zipfile
for zp in BC_DIR.glob("*.zip"):
    with zipfile.ZipFile(zp) as z:
        z.extractall(BC_DIR)
    zp.unlink()

print("\\n=== Files downloaded ===")
for p in sorted(NB1_DIR.glob("*")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")
for p in sorted(BC_DIR.glob("*")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.3f} MB")"""
cells.append(code_cell("download", DOWNLOAD))

# ── Cell: Mount Google Drive ──
DRIVE = r"""# ─── Mount Google Drive (for persisting weights) ───
from google.colab import drive
drive.mount('/content/drive')

DRIVE_OUT = Path("/content/drive/MyDrive/birdclef2026/ns_weights")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
print(f"Drive output: {DRIVE_OUT}")"""
cells.append(code_cell("drive", DRIVE))

# ── Cell: Imports + Config ──
IMPORTS_CFG = r"""# ─── Imports + config ───
import gc, re, warnings, time, random, os
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
seed_everything(SEED)

# Hyper-params (mirror NB4 v10)
SR = 32_000; WINDOW_SEC = 5; N_WINDOWS = 12
D_MODEL = 128; D_STATE = 16; N_SSM_LAYERS = 2
MLP_HIDDEN = 256
DROPOUT = 0.1
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
N_SITES = 32; META_DIM = 8
SEEDS = [42, 123, 777, 2024, 9999]
USE_SWA = True; SWA_START_FRAC = 0.65
LAMBDA_KD = 0.15
LAMBDA_PRIOR = 0.3
PRIOR_STRENGTH_SITE = 8.0
PRIOR_STRENGTH_HOUR = 8.0
PRIOR_STRENGTH_SH = 4.0

# A100-friendly batch (40-80GB VRAM allows large batches → fewer SSM Python-loop calls)
BATCH_FILES = 512
PRED_BATCH  = 512

# Noisy-student
N_ROUNDS = 5
PSEUDO_WEIGHT = 0.3

# Paths
NB1_DIR = Path("/content/nb1_output")
BC_DIR  = Path("/content/bc2026")
WEIGHTS_DIR = Path("/content/ns_weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
TAXONOMY_CSV = BC_DIR / "taxonomy.csv"
SC_LABELS_CSV = BC_DIR / "train_soundscapes_labels.csv"

META_PAT = re.compile(r"_S(\d{2})_(\d{8})_(\d{2})\d{4}")"""
cells.append(code_cell("config", IMPORTS_CFG))

# ── Cell: Taxonomy ──
TAX = r"""# ─── Taxonomy ───
taxonomy = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
print(f"Species: {N_CLASSES}")"""
cells.append(code_cell("taxonomy", TAX))

# ── Cell: Load data ──
LOAD = r"""# ─── Load embeddings + labels + prior tables ───
sc_data = np.load(NB1_DIR / "soundscape_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)
sc_scores = sc_data["scores"].astype(np.float32)
sc_meta = pd.read_parquet(NB1_DIR / "soundscape_meta.parquet")
print(f"All SS: {sc_emb.shape[0]} windows, {sc_meta['filename'].nunique()} files")

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

# Labeled (66 files)
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
print(f"Labeled: {lab_emb_files.shape[0]} files")

# Unlabeled
unlab_meta = sc_meta[~is_labeled].reset_index(drop=True)
unlab_emb_flat = sc_emb[~is_labeled]
unlab_emb_files, unlab_file_list = reshape_to_files(unlab_emb_flat, unlab_meta)
unlab_scores_files, _ = reshape_to_files(sc_scores[~is_labeled], unlab_meta)
unlab_site_ids = np.array([parse_meta(fn)[0] for fn in unlab_file_list], dtype=np.int64)
unlab_hours = np.array([parse_meta(fn)[1] for fn in unlab_file_list], dtype=np.int64)
print(f"Unlabeled: {unlab_emb_files.shape[0]} files")
del sc_emb, sc_scores, sc_meta; gc.collect()

# Prior tables (from labeled)
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
unlab_prior_files = np.stack([compute_prior_logit(s, h) for s, h in zip(unlab_site_ids, unlab_hours)]).astype(np.float32)
print(f"Prior built: sites={len(prior_site_ids)}, hours={len(prior_hours)}, sh={len(sh_to_pi)}")

# Flat tensors for ProtoSSM init (labeled only)
lab_emb_flat_t = torch.tensor(lab_emb_flat, dtype=torch.float32).to(DEVICE)
lab_flat_labels_t = torch.zeros(len(lab_meta), N_CLASSES, dtype=torch.float32).to(DEVICE)
for i, rid in enumerate(lab_meta["row_id"]):
    if rid in label_map:
        lab_flat_labels_t[i] = torch.tensor(label_map[rid]).to(DEVICE)"""
cells.append(code_cell("load", LOAD))

# ── Cell: Model defs (reuse from NB4) ──
cells.append(code_cell("model", MODEL))

# ── Cell: Training helpers ──
HELPERS = r"""# ─── Training & pseudo-label helpers ───
def train_one_seed(model_factory, seed, train_emb_np, train_logits_np, train_labels_np,
                   train_site_np, train_hour_np, train_prior_np, sample_weights_np,
                   teacher_prob_np, pos_weight_t):
    seed_everything(seed)
    model = model_factory().to(DEVICE)
    if hasattr(model, "init_prototypes"):
        model.init_prototypes(lab_emb_flat_t, lab_flat_labels_t)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    swa_model = AveragedModel(model) if USE_SWA else None
    swa_start = int(N_EPOCHS * SWA_START_FRAC)
    swa_n = 0
    best_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    pw = pos_weight_t.unsqueeze(0).unsqueeze(0).to(DEVICE)
    N = train_emb_np.shape[0]
    for epoch in range(N_EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        ep_losses = []
        for i in range(0, N, BATCH_FILES):
            idx = perm[i:i + BATCH_FILES]
            e   = torch.from_numpy(train_emb_np[idx]).to(DEVICE, non_blocking=True)
            lg  = torch.from_numpy(train_logits_np[idx]).to(DEVICE, non_blocking=True)
            lbl = torch.from_numpy(train_labels_np[idx]).to(DEVICE, non_blocking=True)
            st  = torch.from_numpy(train_site_np[idx]).long().to(DEVICE, non_blocking=True)
            hr  = torch.from_numpy(train_hour_np[idx]).long().to(DEVICE, non_blocking=True)
            pr  = torch.from_numpy(train_prior_np[idx]).to(DEVICE, non_blocking=True)
            sw  = torch.from_numpy(sample_weights_np[idx]).to(DEVICE, non_blocking=True).unsqueeze(1).unsqueeze(2)
            tp  = torch.from_numpy(teacher_prob_np[idx]).to(DEVICE, non_blocking=True)
            out = model(e, lg, site_ids=st, hours=hr, prior_logit=pr, lambda_prior=LAMBDA_PRIOR)
            bce = F.binary_cross_entropy(out, lbl, reduction="none")
            loss_main = (bce * pw * sw).mean()
            loss_kd = F.binary_cross_entropy(out, tp, reduction="mean")
            loss = loss_main + LAMBDA_KD * loss_kd
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())
            del e, lg, lbl, st, hr, pr, sw, tp, out, loss
        scheduler.step()
        ep_loss = float(np.mean(ep_losses)) if ep_losses else float('inf')
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if USE_SWA and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_n += 1
    if USE_SWA and swa_n >= 1:
        model.load_state_dict(swa_model.module.state_dict())
    else:
        model.load_state_dict(best_state)
    model.eval()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, best_loss


@torch.no_grad()
def predict_files_cpu(models, emb_np, logits_np, site_np, hour_np, prior_np, batch=PRED_BATCH):
    N = emb_np.shape[0]
    out_sum = np.zeros((N, N_WINDOWS, N_CLASSES), dtype=np.float32)
    for m in models:
        m.eval()
        for i in range(0, N, batch):
            j = min(i + batch, N)
            e  = torch.from_numpy(emb_np[i:j]).to(DEVICE, non_blocking=True)
            lg = torch.from_numpy(logits_np[i:j]).to(DEVICE, non_blocking=True)
            st = torch.from_numpy(site_np[i:j]).long().to(DEVICE, non_blocking=True)
            hr = torch.from_numpy(hour_np[i:j]).long().to(DEVICE, non_blocking=True)
            pr = torch.from_numpy(prior_np[i:j]).to(DEVICE, non_blocking=True)
            o = m(e, lg, site_ids=st, hours=hr, prior_logit=pr, lambda_prior=LAMBDA_PRIOR)
            out_sum[i:j] += o.detach().cpu().numpy()
            del e, lg, st, hr, pr, o
    out_sum /= len(models)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out_sum


print(f"Helpers ready. BATCH_FILES={BATCH_FILES}, N_ROUNDS={N_ROUNDS}, "
      f"PSEUDO_WEIGHT={PSEUDO_WEIGHT}, SEEDS={SEEDS}")"""
cells.append(code_cell("helpers", HELPERS))

# ── Cell: Noisy-student loop ──
LOOP = r"""# ─── 5-round Noisy Student loop ───
lab_emb_np    = lab_emb_files
lab_logits_np = lab_scores_files
lab_labels_np = lab_labels_files
lab_site_np   = lab_site_ids
lab_hour_np   = lab_hours
lab_prior_np  = lab_prior_files

unlab_emb_np    = unlab_emb_files
unlab_logits_np = unlab_scores_files
unlab_site_np   = unlab_site_ids
unlab_hour_np   = unlab_hours
unlab_prior_np  = unlab_prior_files

_lab_lbl_t = torch.from_numpy(lab_labels_np)
_pc = _lab_lbl_t.sum(dim=(0, 1)).clamp(min=1)
_nc = _lab_lbl_t.shape[0] * _lab_lbl_t.shape[1] - _pc
pos_weight_t = (_nc / _pc).clamp(max=30.0)
del _lab_lbl_t

teacher_prob_lab_np = (1.0 / (1.0 + np.exp(-lab_logits_np))).clip(1e-7, 1 - 1e-7).astype(np.float32)
current_pseudo_np = None

for r in range(N_ROUNDS):
    print(f"\n========== ROUND {r} ==========")
    t_round = time.time()

    if r == 0 or current_pseudo_np is None:
        train_emb_np = lab_emb_np; train_logits_np = lab_logits_np
        train_labels_np = lab_labels_np; train_site_np = lab_site_np
        train_hour_np = lab_hour_np; train_prior_np = lab_prior_np
        sample_w_np = np.ones(lab_emb_np.shape[0], dtype=np.float32)
        teacher_prob_np = teacher_prob_lab_np
    else:
        train_emb_np    = np.concatenate([lab_emb_np,    unlab_emb_np],    axis=0)
        train_logits_np = np.concatenate([lab_logits_np, unlab_logits_np], axis=0)
        train_labels_np = np.concatenate([lab_labels_np, current_pseudo_np.astype(np.float32)], axis=0)
        train_site_np   = np.concatenate([lab_site_np,   unlab_site_np],   axis=0)
        train_hour_np   = np.concatenate([lab_hour_np,   unlab_hour_np],   axis=0)
        train_prior_np  = np.concatenate([lab_prior_np,  unlab_prior_np],  axis=0)
        sample_w_np = np.concatenate([
            np.ones(lab_emb_np.shape[0], dtype=np.float32),
            np.full(unlab_emb_np.shape[0], PSEUDO_WEIGHT, dtype=np.float32),
        ], axis=0)
        teacher_prob_np = np.concatenate([teacher_prob_lab_np, current_pseudo_np.astype(np.float32)], axis=0)

    print(f"  train data: {train_emb_np.shape[0]} files ({train_emb_np.nbytes/1e9:.2f} GB on CPU)")

    proto_models = []
    for seed in SEEDS:
        factory = lambda: ProtoSSM(d_input=1536, d_model=D_MODEL, d_state=D_STATE,
                                   n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
                                   n_windows=N_WINDOWS, dropout=DROPOUT,
                                   n_sites=N_SITES, meta_dim=META_DIM)
        t_seed = time.time()
        m, bl = train_one_seed(factory, seed, train_emb_np, train_logits_np, train_labels_np,
                               train_site_np, train_hour_np, train_prior_np,
                               sample_w_np, teacher_prob_np, pos_weight_t)
        proto_models.append(m)
        torch.save(m.state_dict(), WEIGHTS_DIR / f"round{r}_proto_seed{seed}.pt")
        print(f"  ProtoSSM[seed {seed}] best_loss={bl:.4f}, {time.time()-t_seed:.0f}s")

    mlp_models = []
    for seed in SEEDS:
        factory = lambda: MLPHead(d_input=1536, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                                  dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM)
        t_seed = time.time()
        m, bl = train_one_seed(factory, seed, train_emb_np, train_logits_np, train_labels_np,
                               train_site_np, train_hour_np, train_prior_np,
                               sample_w_np, teacher_prob_np, pos_weight_t)
        mlp_models.append(m)
        torch.save(m.state_dict(), WEIGHTS_DIR / f"round{r}_mlp_seed{seed}.pt")
        print(f"  MLPHead[seed {seed}] best_loss={bl:.4f}, {time.time()-t_seed:.0f}s")

    if r < N_ROUNDS - 1:
        proto_pred = predict_files_cpu(proto_models, unlab_emb_np, unlab_logits_np,
                                       unlab_site_np, unlab_hour_np, unlab_prior_np)
        mlp_pred = predict_files_cpu(mlp_models, unlab_emb_np, unlab_logits_np,
                                     unlab_site_np, unlab_hour_np, unlab_prior_np)
        current_pseudo_np = ((proto_pred + mlp_pred) / 2.0).clip(1e-3, 1 - 1e-3).astype(np.float32)
        print(f"  pseudo updated: shape={current_pseudo_np.shape}, "
              f"mean={current_pseudo_np.mean():.4f}, max={current_pseudo_np.max():.4f}")
        del proto_pred, mlp_pred

    del proto_models, mlp_models
    if r > 0:
        del train_emb_np, train_logits_np, train_labels_np
        del train_site_np, train_hour_np, train_prior_np
        del sample_w_np, teacher_prob_np
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"  Round {r} done in {(time.time()-t_round)/60:.1f} min")

print("\n=== ALL ROUNDS DONE ===")"""
cells.append(code_cell("ns-loop", LOOP))

# ── Cell: Save weights to Drive + zip for Kaggle Dataset ──
SAVE = r"""# ─── Persist weights ───
import shutil

saved = sorted(WEIGHTS_DIR.glob("*.pt"))
print(f"Total weight files: {len(saved)}")
total_mb = sum(p.stat().st_size for p in saved) / 1e6
print(f"Total size: {total_mb:.1f} MB")

# Copy to Drive
for p in saved:
    shutil.copy(p, DRIVE_OUT / p.name)
print(f"Copied to {DRIVE_OUT}")

# Build a zip for easy Kaggle Dataset upload
zip_path = Path("/content/ns_weights.zip")
shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", WEIGHTS_DIR)
print(f"Zip: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
shutil.copy(zip_path, DRIVE_OUT.parent / "ns_weights.zip")

print("\n=== NEXT STEPS ===")
print("1. Download ns_weights.zip from Drive (or use unzipped folder)")
print("2. Create Kaggle Dataset 'maekeso/birdclef2026-exp010-ns-weights'")
print("3. Upload all *.pt files to the dataset")
print("4. Update _push_nb6.py to add the dataset slug, then push NB6")"""
cells.append(code_cell("save", SAVE))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb5_colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
