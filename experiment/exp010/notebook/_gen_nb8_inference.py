"""Generate exp010 NB8: Inference with MLP head pretrained on train_audio (from NB7).

Same as NB4 but the MLP head training is replaced by loading NB7-trained weights.
ProtoSSM still trains in-NB on 66 labeled SS (unchanged).

CPU 90 min. Kernel sources: NB1 (Perch embeddings) + NB7 (pretrained MLP weights).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from _gen_nb4_blend import (
    code_cell, md_cell,
    INSTALL, IMPORTS, CONFIG, TAXONOMY, LOAD, MODEL, ONNX_TEST, INFERENCE, SUBMISSION,
)

cells = []

cells.append(md_cell("hdr",
    "# exp010 NB8: Inference with NB7 train_audio-pretrained MLP head\n"
    "\n"
    "NB4 と同じパイプライン。**MLP head の学習を NB7 の重み load に置換** (ProtoSSM はそのまま 66 SS で学習)。\n"
    "\n"
    "MLP head は train_audio 265k focal で pretrain → labeled SS 66 で fine-tune 済 (NB7)。"))

cells.append(code_cell("install", INSTALL))
cells.append(code_cell("imports", IMPORTS))
cells.append(code_cell("config", CONFIG))
cells.append(code_cell("taxonomy", TAXONOMY))
cells.append(code_cell("load", LOAD))
cells.append(code_cell("model", MODEL))

# ── Modified TRAIN: train ProtoSSM as before, LOAD MLP from NB7 ──
TRAIN_AND_LOAD = r"""# === Train ProtoSSM (same as NB4) + Load NB7 MLP weights ===
emb_flat = torch.tensor(lab_emb_flat, dtype=torch.float32)
lab_flat = torch.zeros(len(lab_meta), N_CLASSES, dtype=torch.float32)
for i, rid in enumerate(lab_meta["row_id"]):
    if rid in label_map:
        lab_flat[i] = torch.tensor(label_map[rid])

train_emb = torch.tensor(lab_emb_files, dtype=torch.float32)
train_logits = torch.tensor(lab_scores_files, dtype=torch.float32)
train_labels = torch.tensor(lab_labels_files, dtype=torch.float32)
train_site = torch.tensor(lab_site_ids, dtype=torch.long)
train_hour = torch.tensor(lab_hours, dtype=torch.long)
train_prior = torch.tensor(lab_prior_files, dtype=torch.float32)

pos_counts = train_labels.sum(dim=(0, 1)).clamp(min=1)
neg_counts = train_labels.shape[0] * train_labels.shape[1] - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(max=30.0)

teacher_prob = torch.sigmoid(train_logits).clamp(1e-7, 1 - 1e-7)


def _train_proto(seed):
    seed_everything(seed)
    model = ProtoSSM(
        d_input=1536, d_model=D_MODEL, d_state=D_STATE,
        n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
        n_windows=N_WINDOWS, dropout=DROPOUT,
        n_sites=N_SITES, meta_dim=META_DIM,
    ).to(DEVICE)
    if hasattr(model, "init_prototypes"):
        model.init_prototypes(emb_flat, lab_flat)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    swa_model = AveragedModel(model) if USE_SWA else None
    swa_start = int(N_EPOCHS * SWA_START_FRAC)
    swa_n = 0
    best_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(N_EPOCHS):
        model.train()
        out = model(train_emb, train_logits, site_ids=train_site, hours=train_hour,
                    prior_logit=train_prior, lambda_prior=LAMBDA_PRIOR)
        bce = F.binary_cross_entropy(out, train_labels, reduction="none")
        loss_main = (bce * pos_weight.unsqueeze(0).unsqueeze(0)).mean()
        loss_kd = F.binary_cross_entropy(out, teacher_prob, reduction="mean")
        loss = loss_main + LAMBDA_KD * loss_kd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
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


# Train ProtoSSM × SEEDS
proto_models = []
t0 = time.time()
for seed in SEEDS:
    t_s = time.time()
    m, bl = _train_proto(seed)
    proto_models.append(m)
    print(f"  ProtoSSM[seed {seed}] best_loss={bl:.4f}, {time.time()-t_s:.0f}s")
print(f"ProtoSSM ensemble: {len(proto_models)} models in {time.time()-t0:.0f}s")


# ── Load NB7 MLP weights ──
import re as _re
NB7_CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb7-pretrain-mlp"),
    Path("/kaggle/input/birdclef2026-exp010-nb7-pretrain-mlp"),
]
NB7_ROOT = None
for _c in NB7_CANDIDATES:
    if _c.exists():
        NB7_ROOT = _c
        break
if NB7_ROOT is None:
    for _c in Path("/kaggle/input").rglob("mlp_weights"):
        NB7_ROOT = _c.parent
        break
assert NB7_ROOT is not None, "NB7 not attached as kernel_source"
NB7_WEIGHTS = NB7_ROOT / "mlp_weights"
print(f"\nNB7 weights dir: {NB7_WEIGHTS}")

mlp_models = []
for seed in SEEDS:
    p = NB7_WEIGHTS / f"finetuned_seed{seed}.pt"
    assert p.exists(), f"missing {p.name}"
    m = MLPHead(d_input=1536, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM).to(DEVICE)
    sd = torch.load(p, map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    mlp_models.append(m)
print(f"MLP loaded from NB7: {len(mlp_models)} models")"""
cells.append(code_cell("train-proto-load-mlp", TRAIN_AND_LOAD))

cells.append(code_cell("onnx-test", ONNX_TEST))
cells.append(code_cell("inference", INFERENCE))
cells.append(code_cell("submission", SUBMISSION))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb8_inference.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
