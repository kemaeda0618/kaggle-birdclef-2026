"""exp037: Perch + LightProtoSSM + sklearn MLP probes + Residual SSM
   — in-NB train on labeled SS (66 files、Colab 不使用).

Source ref: https://www.kaggle.com/code/mtoshidesu/0-932-test-mod-version-4

Architecture:
- Backbone:        Perch v2 (CPU ONNX、frozen)
- Sequence model:  LightProtoSSM (Selective SSM、Mamba 系、cross-attn=2 heads、SWA)
- Classifier head: sklearn per-class MLP probes (PCA dim=128、MLPClassifier(256, 128))
- Second-pass:     ResidualSSM (zero-init output、~439K params、CPU 20-30 sec train)

Training paradigm:
- All trained in-NB on Kaggle CPU using labeled train_soundscapes (66 files)
- No Colab assets used
- ProtoSSM 40 epochs OneCycleLR + SWA
- MLP probes per class with sklearn MLPClassifier
- Residual SSM trains residuals (Y - sigmoid(first_pass))

Expected LB: 0.932 standalone (mtoshidesu baseline).
exp031 blend 4th stream として組み込めば +0.001-0.003 期待 (high overlap reservations あり).

Differences from source NB:
- cell IDs と header markdown は我々の format
- Code logic は mtoshidesu のまま (clean port)
- Output: experiment/exp037/notebook/nb_perch_lightprotossm_resssm.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC_NB_PATH = (
    HERE.parent.parent / "exp010" / "public_nb" / "_zushi" /
    "mtoshidesu_0-932-test-mod-version-4" / "0-932-test-mod-version-4.ipynb"
)
src_nb = json.loads(SRC_NB_PATH.read_text(encoding="utf-8"))
print(f"[exp037 lightprotossm port] source NB: {SRC_NB_PATH.name}")
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

# ── Our header (replaces source flashy intro) ──
cells.append(md_cell("hdr", r"""# exp037 — Perch + LightProtoSSM + MLP probes + ResidualSSM

**In-NB trained** standalone pipeline targeting LB 0.932 (mtoshidesu baseline).

## Architecture
| Component | Detail |
|---|---|
| Backbone | Perch v2 (CPU ONNX, frozen) |
| Sequence model | **LightProtoSSM** (Selective SSM / Mamba, cross-attn=2 heads, SWA) |
| Classifier head | **sklearn MLP probes** (PCA(emb, 128) → MLPClassifier(256, 128) per class) |
| Second-pass | **ResidualSSM** (residual = labels - sigmoid(first_pass), zero-init output) |

## Training data
- labeled train_soundscapes (66 files × 12 windows = 792 windows)
- **No train_audio, no Colab assets** — everything trained in-NB on Kaggle CPU

## Pipeline
1. Perch ONNX load → labeled SS embedding cache build
2. LightProtoSSM train (40 ep OneCycleLR + SWA + Mixup α=0.4 + Focal γ=2.5)
3. sklearn per-class MLP probes train (alpha_blend=0.25)
4. Residual SSM train on first-pass residuals (~20 sec)
5. Test inference: Perch → first_pass (0.5 ProtoSSM + 0.5 MLP) → ResSSM correction → light PP

## Expected LB
- Standalone: **0.932** (mtoshidesu baseline)
- exp031 blend 4th stream 追加: +0.001-0.003 (相関高、effort 比 marginal)

## Attribution
Source: mtoshidesu/0-932-test-mod-version-4 (Kaggle public NB). Code logic は port、cell ID と header は我々の format に書換。
"""))

# ── Port mtoshidesu's cells with our cell IDs ──
for i, c in enumerate(src_nb["cells"]):
    src = c.get("source", "")
    if isinstance(src, list):
        joined = "".join(src)
    else:
        joined = src
    if not joined.strip():
        continue

    # Skip mtoshidesu's intro markdown (cell 0)
    if c["cell_type"] == "markdown" and i == 0:
        continue

    cell_id = f"port_{i:02d}"
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

out_path = HERE / "nb_perch_lightprotossm_resssm.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path.name}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
