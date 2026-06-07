"""exp037 v313 — Perch + LightProtoSSM + sklearn MLP probes + Residual SSM
   (mtoshidesu v313904345 = REAL 0.932 LB version、Fast Safe 化前).

Source: https://www.kaggle.com/code/mtoshidesu/0-932-test-mod-version-4?scriptVersionId=313904345

Difference from previous port (which used latest version = Fast Safe 0.922 floor):
- ENSEMBLE_W 0.20 → 0.5 (proto + mlp 50:50)
- Prior lambda 0.10 → 0.4
- FCS_POWER 0.05 → 0.4
- TTA shifts [] → [0, 1, -1, 2, -2]
- ResidualSSM training: skip → full train on labeled SS first-pass
- Per-class threshold calibration: skip → full
- Final inference: Cell 28 Fast Safe → Cell 24 Full Pipeline

Architecture:
- Backbone:        Perch v2 (CPU ONNX、frozen)
- Sequence model:  LightProtoSSM (Selective SSM、Mamba 系、cross-attn、SWA)
- Classifier head: sklearn per-class MLP probes (PCA dim=64、MLPClassifier(256, 128))
- Second-pass:     ResidualSSM (zero-init output、~439K params、CPU 20-30 sec train)

Expected LB: 0.932 standalone (mtoshidesu actual claim).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC_NB_PATH = (
    HERE.parent.parent / "exp010" / "public_nb" / "_zushi" /
    "mtoshidesu_v313904345" / "0-932-v313904345.ipynb"
)
src_nb = json.loads(SRC_NB_PATH.read_text(encoding="utf-8"))
print(f"[exp037 v313] source NB: {SRC_NB_PATH.name}")
print(f"  source cells: {len(src_nb['cells'])}")


def code_cell(cell_id, source):
    if isinstance(source, list):
        src = source
    else:
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "outputs": [], "execution_count": None, "source": src,
    }


def md_cell(cell_id, source):
    if isinstance(source, list):
        src = source
    else:
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src,
    }


cells = []

# ── Our header ──
cells.append(md_cell("hdr", r"""# exp037 v313 — Perch + LightProtoSSM + MLP probes + ResidualSSM (REAL 0.932 path)

**In-NB trained** standalone pipeline targeting LB 0.932 (mtoshidesu v313904345 baseline).

> **重要**: mtoshidesu の最新版は Fast Safe 0.922 recovery 化されてるので、
> 0.932 を出すには **scriptVersionId=313904345** (v313) を使う必要あり。

## Architecture
| Component | Detail |
|---|---|
| Backbone | Perch v2 (CPU ONNX, frozen) |
| Sequence model | **LightProtoSSM** (Selective SSM / Mamba, cross-attn, SWA, 40 ep OneCycleLR) |
| Classifier head | **sklearn MLP probes** (PCA(emb, 64) → MLPClassifier(256, 128) per class) |
| Second-pass | **ResidualSSM** (residual = labels - sigmoid(first_pass), zero-init output) |

## Full Pipeline (v313 Cell 24)
1. Perch ONNX load → labeled SS embedding cache build
2. LightProtoSSM train (40 ep + OneCycleLR + SWA + Mixup α=0.4 + Focal γ=2.5)
3. sklearn per-class MLP probes train (alpha_blend=0.4, pca_dim=64)
4. Residual SSM train on training first-pass residuals (TTA included)
5. Per-class threshold calibration (Isotonic)
6. Test inference: Perch → ProtoSSM (TTA [0,±1,±2]) + MLP probes (50:50) → ResSSM correction → temperature scaling → FCS power=0.4 → rank-aware power=0.4 → adaptive delta α=0.20 → per-class thresholds → submission

## Expected LB
- Standalone: **0.932** (mtoshidesu v313 claim)
- exp031 blend 4th stream 追加: +0.001-0.003 (相関高、effort 比 marginal)

## Attribution
Source: mtoshidesu/0-932-test-mod-version-4 @ scriptVersionId=313904345.
Code logic は port、cell ID と header は我々の format に書換。
"""))

# ── Port v313 cells with our cell IDs ──
for i, c in enumerate(src_nb["cells"]):
    src = c.get("source", "")
    if isinstance(src, list):
        joined = "".join(src)
    else:
        joined = src
    if not joined.strip():
        continue

    cell_id = f"v313_{i:02d}"
    if c["cell_type"] == "code":
        cells.append(code_cell(cell_id, src))
    else:
        cells.append(md_cell(cell_id, src))

# ── Assemble NB ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python", "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python", "file_extension": ".py",
        },
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_perch_lightprotossm_resssm_v313.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path.name}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
