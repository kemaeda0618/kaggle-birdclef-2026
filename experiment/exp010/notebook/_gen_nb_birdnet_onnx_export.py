"""Generate BirdNET head ONNX export NB.

Loads the ProtoSSM/MLPHead weights trained by NB4-BirdNET and exports a single
ONNX graph (per seed) that takes (BirdNET emb, site_id, hour, prior_logit) and
returns 234-class probabilities. Output goes to /kaggle/working/nb4_birdnet_onnx/
for download / Dataset upload.

Inputs:
- kernel_sources: maekeso/birdclef2026-exp010-nb4-birdnet (weights + config)
- dataset_sources: rishikeshjani/perch-onnx-for-birdclef-2026 (onnxruntime wheel)

Runs on Kaggle CPU, ~5-10 min.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr",
    "# BirdNET head ONNX export\n"
    "\n"
    "Load NB4-BirdNET trained ProtoSSM/MLPHead weights (5 seeds each) and export\n"
    "to ONNX. Inputs: (emb [B,12,6522], site_ids [B], hours [B], prior_logit [B,234]).\n"
    "Output: prob [B,12,234]. Used by downstream blend NBs to avoid retraining."))

cells.append(code_cell("install", r"""import subprocess, sys, os, time
START = time.time()

# Install onnxruntime from offline wheel if not present
try:
    import onnxruntime as _ort  # noqa: F401
    print("onnxruntime already installed")
except Exception:
    WHEEL_DIR = None
    for d in os.listdir("/kaggle/input"):
        sub = f"/kaggle/input/{d}"
        if os.path.isdir(sub) and any(f.endswith(".whl") for f in os.listdir(sub)):
            WHEEL_DIR = sub; break
    if WHEEL_DIR is None:
        for cand in [
            "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026",
            "/kaggle/input/perch-onnx-for-birdclef-2026",
        ]:
            if os.path.isdir(cand):
                WHEEL_DIR = cand; break
    if WHEEL_DIR is not None:
        whls = [f for f in os.listdir(WHEEL_DIR) if f.endswith(".whl")]
        if whls:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                                   os.path.join(WHEEL_DIR, whls[0])])
            print(f"Installed {whls[0]}")
print(f"Setup done in {time.time()-START:.0f}s")"""))

cells.append(code_cell("imports", r"""import os, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

print(f"torch {torch.__version__}, onnxruntime {ort.__version__}")"""))

cells.append(code_cell("locate", r"""# Locate NB4-BirdNET kernel output (weights + config + retrieval/prior)
NB4_OUT_DIR = None
for cand in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb4-birdnet"),
    Path("/kaggle/input/birdclef2026-exp010-nb4-birdnet"),
]:
    if cand.exists():
        NB4_OUT_DIR = cand
        break
if NB4_OUT_DIR is None:
    for p in Path("/kaggle/input").rglob("config.json"):
        if (p.parent / "proto_seed0.pt").exists():
            NB4_OUT_DIR = p.parent
            break
assert NB4_OUT_DIR is not None, "NB4-BirdNET output not attached"

# The weights live under nb4_birdnet_weights/ inside the kernel output
WEIGHTS_DIR = NB4_OUT_DIR / "nb4_birdnet_weights"
if not WEIGHTS_DIR.exists():
    # search for the actual location
    for p in NB4_OUT_DIR.rglob("config.json"):
        if (p.parent / "proto_seed0.pt").exists():
            WEIGHTS_DIR = p.parent
            break
print(f"WEIGHTS_DIR: {WEIGHTS_DIR}")

cfg = json.loads((WEIGHTS_DIR / "config.json").read_text())
print(json.dumps(cfg, indent=2)[:600])

EMB_DIM = cfg["EMB_DIM"]
D_MODEL = cfg["D_MODEL"]; D_STATE = cfg["D_STATE"]
N_SSM_LAYERS = cfg["N_SSM_LAYERS"]; MLP_HIDDEN = cfg["MLP_HIDDEN"]
DROPOUT = cfg["DROPOUT"]
N_WINDOWS = cfg["N_WINDOWS"]; N_CLASSES = cfg["N_CLASSES"]
N_SITES = cfg["N_SITES"]; META_DIM = cfg["META_DIM"]
SEEDS = cfg["SEEDS"]
PRIMARY_LABELS = cfg["PRIMARY_LABELS"]"""))

cells.append(code_cell("model", r"""# === Model defs (identical to NB4-BirdNET train, must match weights) ===
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_D = nn.Linear(d_model, d_model)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.log_A = nn.Parameter(torch.log(A))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B_sz, L, D = x.shape
        delta = F.softplus(self.proj_delta(x))
        B = self.proj_B(x); C = self.proj_C(x); D_param = self.proj_D(x)
        A = -torch.exp(self.log_A)
        h = torch.zeros(B_sz, self.d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            dt = delta[:, t].unsqueeze(-1)
            dA = torch.exp(A.unsqueeze(0) * dt)
            dB = dt * B[:, t].unsqueeze(1)
            h = h * dA + x[:, t].unsqueeze(-1) * dB
            y = (h * C[:, t].unsqueeze(1)).sum(-1) + D_param[:, t]
            outputs.append(y)
        return self.dropout(torch.stack(outputs, dim=1))


class BiSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.fwd_ssm = SelectiveSSM(d_model, d_state, dropout)
        self.bwd_ssm = SelectiveSSM(d_model, d_state, dropout)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        fwd = self.fwd_ssm(x)
        bwd = self.bwd_ssm(x.flip(1)).flip(1)
        out = self.proj(torch.cat([fwd, bwd], dim=-1))
        return self.norm(x + out)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(a))


class ProtoSSM(nn.Module):
    def __init__(self, d_input, d_model, d_state, n_ssm_layers, n_classes,
                 n_windows, dropout=0.1, n_sites=32, meta_dim=8, n_attn_heads=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_layers = nn.ModuleList([
            BiSSMBlock(d_model, d_state, dropout) for _ in range(n_ssm_layers)])
        self.attn_layers = nn.ModuleList([
            CrossAttnBlock(d_model, num_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_ssm_layers)])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None, prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb) + self.pos_emb[:, :emb.shape[1]]
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        for ssm_layer, attn_layer in zip(self.ssm_layers, self.attn_layers):
            x = ssm_layer(x)
            x = attn_layer(x)
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("btd,cd->btc", x_norm, p_norm) * self.temperature + self.bias
        alpha = torch.sigmoid(self.alpha)
        out = alpha * sim + (1 - alpha) * logits
        if prior_logit is not None and lambda_prior > 0:
            out = out + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(out)


class MLPHead(nn.Module):
    def __init__(self, d_input, d_hidden, n_classes, dropout=0.1, n_sites=32, meta_dim=8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, n_classes))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.ones(n_classes) * 0.5)

    def forward(self, emb, logits, site_ids=None, hours=None, prior_logit=None, lambda_prior=0.0):
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


print("Model classes defined.")"""))

cells.append(code_cell("wrapper", r"""# === Export wrapper ===
# Wraps a head so the ONNX graph signature is:
#   inputs: emb (B, N_WINDOWS, EMB_DIM), site_ids (B,), hours (B,), prior_logit (B, N_CLASSES)
#   output: prob (B, N_WINDOWS, N_CLASSES)
LAMBDA_PRIOR_EXPORT = 0.3   # Match NB4 v7 setting at export time

class ExportWrapper(nn.Module):
    # Wraps a head so the ONNX graph signature is:
    #   inputs: emb (B, N_WINDOWS, EMB_DIM), site_ids (B,), hours (B,),
    #           prior_logit (B, N_CLASSES)
    #   output: prob (B, N_WINDOWS, N_CLASSES)
    # logits=zeros is baked in (BirdNET NB has no Perch teacher).
    def __init__(self, inner, n_classes):
        super().__init__()
        self.inner = inner
        self.n_classes = n_classes

    def forward(self, emb, site_ids, hours, prior_logit):
        B = emb.shape[0]
        T = emb.shape[1]
        zeros_logits = torch.zeros(B, T, self.n_classes,
                                    device=emb.device, dtype=emb.dtype)
        return self.inner(emb, zeros_logits,
                          site_ids=site_ids, hours=hours,
                          prior_logit=prior_logit,
                          lambda_prior=LAMBDA_PRIOR_EXPORT)


print("Export wrapper defined.")"""))

cells.append(code_cell("export", r"""# === Export all heads ===
OUT_DIR = Path("/kaggle/working/nb4_birdnet_onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy config + prior + retrieval assets so downstream NB has everything it needs
import shutil
for fname in ["config.json", "prior_tables.npz", "retrieval_sc.npz"]:
    src = WEIGHTS_DIR / fname
    if src.exists():
        shutil.copy(src, OUT_DIR / fname)


def build_proto():
    return ProtoSSM(d_input=EMB_DIM, d_model=D_MODEL, d_state=D_STATE,
                    n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
                    n_windows=N_WINDOWS, dropout=DROPOUT,
                    n_sites=N_SITES, meta_dim=META_DIM)


def build_mlp():
    return MLPHead(d_input=EMB_DIM, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                    dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM)


def _export_one(model, out_path):
    model.eval()
    wrapper = ExportWrapper(model, N_CLASSES).eval()

    B = 1
    dummy_emb = torch.randn(B, N_WINDOWS, EMB_DIM, dtype=torch.float32)
    dummy_site = torch.zeros(B, dtype=torch.long)
    dummy_hour = torch.zeros(B, dtype=torch.long)
    dummy_prior = torch.zeros(B, N_CLASSES, dtype=torch.float32)

    with torch.no_grad():
        ref_out = wrapper(dummy_emb, dummy_site, dummy_hour, dummy_prior)

    torch.onnx.export(
        wrapper, (dummy_emb, dummy_site, dummy_hour, dummy_prior), str(out_path),
        input_names=["emb", "site_ids", "hours", "prior_logit"],
        output_names=["prob"],
        dynamic_axes={
            "emb": {0: "batch"},
            "site_ids": {0: "batch"},
            "hours": {0: "batch"},
            "prior_logit": {0: "batch"},
            "prob": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "emb": dummy_emb.numpy(),
        "site_ids": dummy_site.numpy(),
        "hours": dummy_hour.numpy(),
        "prior_logit": dummy_prior.numpy(),
    })[0]
    diff = float(np.abs(ref_out.numpy() - onnx_out).max())
    return diff


# ProtoSSM
for si, seed in enumerate(SEEDS):
    pt = WEIGHTS_DIR / f"proto_seed{si}.pt"
    if not pt.exists():
        print(f"  missing {pt}, skip"); continue
    m = build_proto()
    state = torch.load(str(pt), map_location="cpu")
    m.load_state_dict(state)
    out_path = OUT_DIR / f"proto_seed{si}.onnx"
    t0 = time.time()
    diff = _export_one(m, out_path)
    print(f"  proto seed{si} -> {out_path.name}  max|diff|={diff:.2e}  size={out_path.stat().st_size/1e6:.1f} MB  {time.time()-t0:.1f}s")

# MLPHead
for si, seed in enumerate(SEEDS):
    pt = WEIGHTS_DIR / f"mlp_seed{si}.pt"
    if not pt.exists():
        print(f"  missing {pt}, skip"); continue
    m = build_mlp()
    state = torch.load(str(pt), map_location="cpu")
    m.load_state_dict(state)
    out_path = OUT_DIR / f"mlp_seed{si}.onnx"
    t0 = time.time()
    diff = _export_one(m, out_path)
    print(f"  mlp   seed{si} -> {out_path.name}  max|diff|={diff:.2e}  size={out_path.stat().st_size/1e6:.1f} MB  {time.time()-t0:.1f}s")

print("Export complete.")"""))

cells.append(code_cell("summary", r"""# === Done. Upload OUT_DIR as Kaggle Dataset ===
print(f"Files in {OUT_DIR}:")
total = 0
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        sz = p.stat().st_size
        total += sz
        print(f"  {p.name:40s} {sz/1e6:7.2f} MB")
print(f"\nTotal: {total/1e6:.1f} MB")
print(f"\nSuggested Dataset slug: birdclef2026-exp010-nb4-birdnet-onnx")"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_birdnet_onnx_export.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
