"""Generate exp010 NB6: Inference using NB5 noisy-student trained weights.

Same as NB4 but TRAINING is replaced by loading per-round weights from NB5.
Uses round 4 (last round, fully self-trained) by default.

CPU 90min, kernel_sources includes NB1 (embeddings) + NB5 (NS weights).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Reuse cell strings from NB4 generator (re-runs NB4 gen as side-effect — harmless)
from _gen_nb4_blend import (
    code_cell, md_cell,
    INSTALL, IMPORTS, CONFIG, TAXONOMY, LOAD, MODEL, ONNX_TEST, INFERENCE, SUBMISSION,
)

cells = []

cells.append(md_cell("hdr",
    "# exp010 NB6: Inference with Noisy-Student weights from NB5\n"
    "\n"
    "推論パイプラインは NB4 v10 と同一。学習だけ NB5 (5 ラウンド noisy student) で事前に行った\n"
    "重みを kernel_sources 経由で読み込んで使う。\n"
    "\n"
    "デフォルトは **最終ラウンド (round 4)** のみ使用。複数ラウンドの blend は `INFERENCE_ROUNDS` を変更。"))

cells.append(code_cell("install", INSTALL))
cells.append(code_cell("imports", IMPORTS))
cells.append(code_cell("config", CONFIG))
cells.append(code_cell("taxonomy", TAXONOMY))
cells.append(code_cell("load", LOAD))
cells.append(code_cell("model", MODEL))

# NEW: load weights from NB5 instead of training
LOAD_WEIGHTS = r"""# ─── Load Noisy-Student weights from NB5 ───
import re as _re

# Detect NB5 weights directory
NS_KERNEL_CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb5-noisy-student"),
    Path("/kaggle/input/birdclef2026-exp010-nb5-noisy-student"),
]
NS_ROOT = None
for _c in NS_KERNEL_CANDIDATES:
    if _c.exists():
        NS_ROOT = _c
        break
if NS_ROOT is None:
    # Fallback search
    for _c in Path("/kaggle/input").rglob("ns_weights"):
        NS_ROOT = _c.parent
        break
assert NS_ROOT is not None, "NB5 weights kernel not attached"
NS_WEIGHTS_DIR = NS_ROOT / "ns_weights"
assert NS_WEIGHTS_DIR.exists(), f"ns_weights/ not found in {NS_ROOT}"

available = sorted(NS_WEIGHTS_DIR.glob("*.pt"))
print(f"NS_WEIGHTS_DIR: {NS_WEIGHTS_DIR}")
print(f"Available weight files: {len(available)}")

# Detect available rounds
rounds_avail = sorted({int(_re.match(r"round(\d+)_", p.name).group(1)) for p in available
                        if _re.match(r"round\d+_", p.name)})
print(f"Available rounds: {rounds_avail}")

# Default: use only the last round
INFERENCE_ROUNDS = [rounds_avail[-1]] if rounds_avail else [0]
print(f"Using rounds: {INFERENCE_ROUNDS}")


def _load_proto(round_idx, seed):
    m = ProtoSSM(d_input=1536, d_model=D_MODEL, d_state=D_STATE,
                 n_ssm_layers=N_SSM_LAYERS, n_classes=N_CLASSES,
                 n_windows=N_WINDOWS, dropout=DROPOUT,
                 n_sites=N_SITES, meta_dim=META_DIM)
    sd = torch.load(NS_WEIGHTS_DIR / f"round{round_idx}_proto_seed{seed}.pt",
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def _load_mlp(round_idx, seed):
    m = MLPHead(d_input=1536, d_hidden=MLP_HIDDEN, n_classes=N_CLASSES,
                dropout=DROPOUT, n_sites=N_SITES, meta_dim=META_DIM)
    sd = torch.load(NS_WEIGHTS_DIR / f"round{round_idx}_mlp_seed{seed}.pt",
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


proto_models = []
mlp_models = []
for r_idx in INFERENCE_ROUNDS:
    for s in SEEDS:
        proto_models.append(_load_proto(r_idx, s))
        mlp_models.append(_load_mlp(r_idx, s))

print(f"Loaded ProtoSSM ensemble: {len(proto_models)} models "
      f"({len(INFERENCE_ROUNDS)} rounds × {len(SEEDS)} seeds)")
print(f"Loaded MLPHead   ensemble: {len(mlp_models)} models")"""
cells.append(code_cell("load-weights", LOAD_WEIGHTS))

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
out_path = HERE / "nb6_inference_ns.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
