"""exp039: Single-NB integration of exp037 (LightProtoSSM) + Tucker SED + exp029.

NB4 を完全廃止、exp037 v313 路線 (LB 0.930 standalone) を slot 1 に。
全推論 1 NB で完結 (no kernel_source for exp037 output).

Structure:
- Cells 0-N (from exp037 v313): Perch ONNX + LightProtoSSM + MLP probes + ResSSM + Full Pipeline
- Bridge cell: build audio_cache for Tucker/exp029 + map probs_exp037 → probs_exp010
- Tucker SED inference cell (from exp031)
- exp029 inference cell (from exp031)
- 3-way blend cell (from exp031)

期待 LB: 0.951-0.954 (exp031 0.949 baseline + exp037 standalone +0.012 → blend +0.002-0.005)
期待 Run time: 55-65 min (exp037 30 min + Tucker 15-20 min + exp029 8-12 min + audio reload 5-8 min + blend 2-3 min)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
EXP037_NB_PATH = HERE.parent.parent / "exp037" / "notebook" / "nb_perch_lightprotossm_resssm_v313.ipynb"
EXP031_NB_PATH = HERE.parent.parent / "exp031" / "notebook" / "nb_blend.ipynb"

exp037_nb = json.loads(EXP037_NB_PATH.read_text(encoding="utf-8"))
exp031_nb = json.loads(EXP031_NB_PATH.read_text(encoding="utf-8"))
print(f"[exp039] exp037 v313: {len(exp037_nb['cells'])} cells")
print(f"[exp039] exp031:      {len(exp031_nb['cells'])} cells")


def code_cell(cell_id, source):
    if isinstance(source, list):
        src = source
    else:
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

# ── Header (replaces exp037 v313 hdr) ──
cells.append(md_cell("hdr", r"""# exp039 — exp037 + Tucker SED + exp029 single-NB blend (NB4 replaced)

NB4 を **exp037 (LightProtoSSM + MLP probes + ResSSM in-NB train)** に置換した 3-way rank blend.
全推論 1 NB で完結 (no kernel_source for exp037 output).

## Architecture
| Slot | Model | Weight |
|---|---|---|
| 1 | **exp037 (LightProtoSSM + MLP probes + ResSSM)** | **0.35** |
| 2 | Tucker SED (5-fold ONNX) | 0.40 |
| 3 | exp029 R3 (NFNet l1 single fold) | 0.25 |

## Expected
- LB **0.951-0.954** (exp031 0.949 baseline + standalone +0.012 → blend +0.002-0.005)
- Run time **55-65 min** on Kaggle CPU (exp037 30 min + Tucker 15-20 + exp029 8-12 + audio reload 5-8)

## NB4 → exp037 swap rationale
- exp037 v313 standalone LB **0.930** vs NB4 standalone ~0.918-0.924 = +0.006-0.012
- Removed deps: anura/inat pool、exp010-nb1-embedding kernel_source
- Kept deps: Perch ONNX、Tucker SED、exp029 ckpt
"""))

# ── Modify Cell 24 (exp037 Full Pipeline) to save probs_exp037 instead of writing submission ──
EXP037_FULL_PIPELINE_PATCH = {
    'sub.to_csv("submission.csv", index=False)':
        '# exp039: save probs for blend (do NOT write submission yet)\n'
        'probs_exp037 = probs.copy().astype(np.float32)\n'
        'print(f"probs_exp037 ready for blend: shape={probs_exp037.shape}")',
    'assert len(sub) == len(test_paths) * N_WINDOWS':
        '# exp039: skip (will be checked at final submission write)',
    'sub.to_csv("submission.csv", index=False)\n\nprint(f"\\nsubmission.csv saved — shape {sub.shape}")':
        'probs_exp037 = probs.copy().astype(np.float32)\n'
        'print(f"\\nexp037 Full Pipeline done — probs_exp037 shape={probs_exp037.shape}")',
}

# ── Port exp037 v313 cells (excluding original hdr) ──
for c in exp037_nb["cells"]:
    cid = c.get("id", "")
    if cid == "hdr":
        continue  # skip exp037's header (we have our own)
    src = "".join(c.get("source", []))
    if cid == "v313_24":
        # Modify Full Pipeline cell to save probs_exp037 (no submission write)
        for old, new in EXP037_FULL_PIPELINE_PATCH.items():
            src = src.replace(old, new)
        cells.append(code_cell("exp037_full_pipeline", src))
    else:
        cells.append(code_cell(cid, src))

# ── Bridge cell: build audio_cache for Tucker/exp029 + map probs_exp037 → probs_exp010 ──
BRIDGE_SRC = r"""# === exp039: Bridge — build audio_cache for Tucker SED + exp029, map probs to exp031 blend format ===
import soundfile as sf
import librosa as _librosa_bridge
import torch.nn as nn
import torch.nn.functional as F

# === exp039: Constants used by exp031 blend code (need to be defined here) ===
# FCS (file_confidence_scale): top_K=2, power=0.4 standard from Karnakbayev V18 pipeline
FCS_TOP_K = 2
FCS_POWER = 0.4

# Build audio_cache: list of 60s raw float32 waveforms per test file
FILE_SAMPLES_60s = SR * 60
audio_cache = []
t0_audio = time.time()
print(f"\n[exp039] Building audio_cache from {len(test_paths)} test files (for Tucker + exp029)...")
for fi, fp in enumerate(test_paths):
    try:
        wav, sr_read = sf.read(str(fp), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr_read != SR:
            wav = _librosa_bridge.resample(wav, orig_sr=sr_read, target_sr=SR)
    except Exception:
        wav, _ = _librosa_bridge.load(str(fp), sr=SR, mono=True)
        wav = wav.astype(np.float32)

    target_len = FILE_SAMPLES_60s
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    audio_cache.append(wav.astype(np.float32))
    if (fi + 1) % 50 == 0 or fi == 0 or fi == len(test_paths) - 1:
        print(f"  audio_cache [{fi+1}/{len(test_paths)}] {time.time()-t0_audio:.0f}s")
print(f"audio_cache built: {len(audio_cache)} files in {time.time()-t0_audio:.0f}s")

# Bridge probs_exp037 → probs_exp010 (var name kept for exp031 blend code reuse)
all_row_ids = meta_te["row_id"].astype(str).tolist()
n_test_rows = len(all_row_ids)
WINDOW_SAMPLES = SR * 5  # 32000 * 5 = 160000
WINDOW_SEC = 5
assert n_test_rows % N_WINDOWS == 0, f"n_test_rows {n_test_rows} not divisible by N_WINDOWS {N_WINDOWS}"
probs_exp010 = probs_exp037.reshape(-1, N_WINDOWS, N_CLASSES).astype(np.float32)
print(f"probs_exp010 (=exp037 reshaped): {probs_exp010.shape}")

# BASE etc. (exp037 already defines BASE, reuse)
TEST_DIR = BASE / "test_soundscapes"
TRAIN_SC_DIR = BASE / "train_soundscapes"
test_files = [str(p) for p in test_paths]  # convert to str list for exp031 code

# START reference (used by blend cell)
START = time.time() - (time.time() - _WALL_START)  # cumulative wall time from exp037

# onnxruntime as ort (already imported in exp037, but reaffirm)
import onnxruntime as ort

# PRIMARY_LABELS — exp037 already defines it
print(f"PRIMARY_LABELS: {len(PRIMARY_LABELS)}")
"""
cells.append(code_cell("bridge", BRIDGE_SRC))

# ── Append Tucker SED, exp029, blend cells from exp031 ──
def get_exp031_cell(cid):
    return next(c for c in exp031_nb["cells"] if c.get("id") == cid)

cells.append(code_cell("sed-infer", "".join(get_exp031_cell("sed-infer")["source"])))
cells.append(code_cell("e29-infer", "".join(get_exp031_cell("e29-infer")["source"])))
cells.append(code_cell("blend", "".join(get_exp031_cell("blend")["source"])))

# ── Assemble ──
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
out_path = HERE / "nb_blend.ipynb"
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nWritten: {out_path.name}")
print(f"  cells: {len(cells)}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
