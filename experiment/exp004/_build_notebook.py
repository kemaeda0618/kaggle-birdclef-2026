"""Build submission.ipynb for exp004 from _reference_source.py.

Reads the reference source, splits on '# === Cell N ===' markers,
applies targeted modifications to Cell 1 and Cell 3, and writes
a valid Kaggle-compatible .ipynb notebook.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "_reference_source.py"
OUT = HERE / "notebook" / "submission.ipynb"

# ---------------------------------------------------------------------------
# 1. Read and split by cell markers
# ---------------------------------------------------------------------------
raw = REF.read_text(encoding="utf-8")

# Split on lines matching: # === Cell N ... ===
parts = re.split(r"^# === Cell \d+[^=]*===\s*$", raw, flags=re.MULTILINE)

# parts[0] is empty (before first marker); cells are parts[1:]
cells = [p.strip() for p in parts[1:]]
assert len(cells) >= 2, f"Expected at least 2 cells, got {len(cells)}"

print(f"Parsed {len(cells)} cells from reference source.")

# ---------------------------------------------------------------------------
# 2. Apply modifications
# ---------------------------------------------------------------------------

# --- Cell 1 (index 0): replace entirely ---
cells[0] = """\
import subprocess, sys
from pathlib import Path

# Try multiple known TF wheel locations
_candidates = [
    Path("/kaggle/input/tensorflow-2-20-wheels"),
    Path("/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel"),
]
_WHL = None
for d in _candidates:
    if d.exists():
        _WHL = d
        break
if _WHL is None:
    raise RuntimeError(
        "TF wheel not found. Add one of: tensorflow-2-20-wheels (Dataset) or kdmitrie/bc26-tensorflow-2-20-0 (Notebook)"
    )

# Find wheel files
import glob
tb_whl = glob.glob(str(_WHL / "tensorboard*.whl"))
tf_whl = glob.glob(str(_WHL / "tensorflow-2.20*.whl"))
if not tb_whl or not tf_whl:
    raise RuntimeError(f"Wheel files not found in {_WHL}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", tb_whl[0]], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", tf_whl[0]], check=True)
print("Installed TF 2.20.0 from", _WHL)"""

# --- Cell 3 (index 2): targeted replacements ---
c3 = cells[2]

# Replace BASE
c3 = c3.replace(
    'BASE = Path("/kaggle/input/competitions/birdclef-2026")',
    'BASE = Path("/kaggle/input/birdclef-2026")\n'
    'if not BASE.exists():\n'
    '    BASE = Path("/kaggle/input/competitions/birdclef-2026")',
)

# Replace MODEL_DIR
c3 = c3.replace(
    'MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")',
    '_model_candidates = [\n'
    '    Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"),\n'
    '    Path("/kaggle/input/bird-vocalization-classifier/google/bird-vocalization-classifier/perch_v2_cpu/1"),\n'
    ']\n'
    'MODEL_DIR = None\n'
    'for p in _model_candidates:\n'
    '    if p.exists():\n'
    '        MODEL_DIR = p\n'
    '        break\n'
    'if MODEL_DIR is None:\n'
    '    found = list(Path("/kaggle/input").rglob("saved_model.pb"))\n'
    '    if found:\n'
    '        MODEL_DIR = found[0].parent\n'
    '    else:\n'
    '        MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")',
)

# Replace training hyperparameters
c3 = c3.replace('"n_epochs": 120,', '"n_epochs": 80,')
c3 = c3.replace('"patience": 20,', '"patience": 15,')
c3 = c3.replace('"max_iter": 300,', '"max_iter": 200,')
c3 = c3.replace(
    '"dryrun_n_files": 50 if MODE == "train" else 20,',
    '"dryrun_n_files": 30 if MODE == "train" else 10,',
)

cells[2] = c3

# ---------------------------------------------------------------------------
# 3. Build notebook JSON
# ---------------------------------------------------------------------------
nb_cells = []

# Markdown title cell (very first cell)
title_md = """\
# BirdCLEF+ 2026 — exp004: ProtoSSM Ensemble
Based on kamongi/pantanal-distill-birdclef2026

**Kaggle Input に追加するもの:**
- `birdclef-2026` — コンペデータ
- `jaejohn/perch-meta` — Perch推論キャッシュ
- `google/bird-vocalization-classifier (perch_v2_cpu)` — Perch v2 SavedModel
- `tensorflow-2-20-wheels` — TF 2.20.0 ホイール（または kdmitrie/bc26-tensorflow-2-20-0）"""

nb_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": title_md.splitlines(True),
})

# Code cells
for cell_src in cells:
    # Ensure source lines end with newlines for .ipynb convention
    lines = cell_src.splitlines(True)
    # Last line should NOT have trailing newline per nbformat convention
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    nb_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    })

notebook = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
        "kaggle": {
            "accelerator": "none",
            "dataSources": [
                {
                    "sourceType": "competition",
                    "sourceId": 129329,
                    "databundleVersionId": 15996945,
                },
                {
                    "sourceType": "modelInstanceVersion",
                    "sourceId": 542750,
                    "databundleVersionId": 13512426,
                    "modelInstanceId": 418187,
                    "modelId": 319,
                },
            ],
            "isInternetRequired": False,
            "isGpuEnabled": False,
            "kernelType": "notebook",
            "language": "python",
            "title": "BirdCLEF 2026 - exp004 ProtoSSM",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# ---------------------------------------------------------------------------
# 4. Write notebook
# ---------------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

size_kb = OUT.stat().st_size / 1024
print(f"Wrote {OUT}")
print(f"Size: {size_kb:.1f} KB")
print(f"Cells: {len(nb_cells)} (1 markdown + {len(cells)} code)")
