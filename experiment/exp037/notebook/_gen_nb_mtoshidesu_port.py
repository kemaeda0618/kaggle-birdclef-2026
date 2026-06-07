"""exp037 mtoshidesu port — 0.932 standalone NB を我々の format に書き直し。

Source: https://www.kaggle.com/code/mtoshidesu/0-932-test-mod-version-4

Pipeline (mtoshidesu V18 流):
1. Perch ONNX load (TF SavedModel fallback)
2. Labeled SS (66 files) で Perch embedding 抽出 → cache
3. Per-fold OOF or in-NB train:
   - LightProtoSSM (40 ep + OneCycleLR + SWA)
   - sklearn per-class MLP probes (PCA dim=128)
   - Residual SSM (second-pass error correction)
4. Test inference: Perch → ProtoSSM + MLP blend → Residual correction → light PP → submission

Submission mode default (`MODE = "submit"`).

期待 LB: 0.932 standalone (mtoshidesu baseline).
我々の exp031 blend に 4th stream として組み込めば +0.001-0.003 期待 (high overlap reservations あり).

差分: cell ID, header markdown, exp037 references は我々のフォーマット。code logic は mtoshidesu のまま。
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC_NB_PATH = (
    HERE.parent.parent / "exp010" / "public_nb" / "_zushi" /
    "mtoshidesu_0-932-test-mod-version-4" / "0-932-test-mod-version-4.ipynb"
)
src_nb = json.loads(SRC_NB_PATH.read_text(encoding="utf-8"))
print(f"[exp037 port] reading mtoshidesu NB: {SRC_NB_PATH.name}")
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

# Our own header (replaces mtoshidesu's flashy title)
cells.append(md_cell("hdr", r"""# exp037 — mtoshidesu 0.932 path port

Standalone **Perch + LightProtoSSM + MLP probes + ResidualSSM** pipeline,
trained **in-NB on labeled SS only** (66 files、no Colab assets).

## Goal
Standalone LB **0.932** (mtoshidesu baseline) で exp027 standalone 0.918 を +0.014 上回る。
成功時は exp031 blend に 4th stream として組込検証。

## Pipeline
1. Perch ONNX (CPU) load + taxonomy mapping
2. Labeled SS (66 files) で Perch embedding cache 構築 (in-NB)
3. **LightProtoSSM** train (40 ep + OneCycleLR + SWA)
4. **sklearn per-class MLP probes** train (PCA dim=128、MLPClassifier(256,128))
5. **Residual SSM** train (second-pass error correction、~20 秒)
6. Test inference: Perch → first-pass (0.5 ProtoSSM + 0.5 MLP) → ResSSM correction → light PP

## Source attribution
Original NB: mtoshidesu/0-932-test-mod-version-4 (Kaggle public).
Code logic は port、cell ID と header は我々の format に書換。
"""))

# Port mtoshidesu's cells with our cell IDs
SKIP_INTRO_MD = True  # 最初の README md cell は我々の header で置き換え済
for i, c in enumerate(src_nb["cells"]):
    src = c.get("source", "")
    # Skip empty cells
    if isinstance(src, list):
        joined = "".join(src)
    else:
        joined = src
    if not joined.strip():
        continue

    # Skip mtoshidesu's intro markdown (we have our own header)
    if SKIP_INTRO_MD and c["cell_type"] == "markdown" and i == 0:
        continue

    cell_id = f"port_{i:02d}"
    if c["cell_type"] == "code":
        cells.append(code_cell(cell_id, src))
    else:
        cells.append(md_cell(cell_id, src))

# Assemble NB
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

out_path = HERE / "nb_mtoshidesu_port.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
