"""Build exp104 standalone: exp102 3-fold ensemble OV inference only.
NO NB4, NO Tucker — just exp102 OV 3-fold predictions written to submission.csv.

Use case: measure exp102 3-fold ensemble's standalone LB (compare to exp029 R3 0.923).
Expected: 0.92-0.93 range, very light (~5-10 min sub time).
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path("experiment/exp104/notebook/nb_exp104_standalone_3fold.ipynb")


def cell(src, cid):
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1 = """# === exp102 3-fold OV standalone inference ===
# Light NB: skip NB4 / Tucker, just exp102 3-fold ensemble -> submission.csv
import os, sys, glob, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
"""

CELL2 = """# === Install openvino offline (ttahara wheels) ===
WHEEL_HITS = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
assert WHEEL_HITS, "openvino wheel not found — attach ttahara/birdclef-2026-download-wheels"
WHEEL_DIR = str(Path(WHEEL_HITS[0]).parent)
print(f"[ov-install] wheels: {WHEEL_DIR}")

try:
    import openvino as ov
    print(f"openvino preinstalled: {ov.__version__}")
except ImportError:
    !pip install -q --no-deps {WHEEL_DIR}/openvino-*.whl {WHEEL_DIR}/openvino_telemetry-*.whl
    import openvino as ov
    print(f"openvino installed: {ov.__version__}")
"""

CELL3 = """# === Constants (matches exp102 / exp029 R3 training config) ===
BASE = Path("/kaggle/input/birdclef-2026")
TEST_DIR = BASE / "test_soundscapes"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC          # 160000
N_WINDOWS = 12                             # 12 × 5s = 60s
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
print(f"N_CLASSES={N_CLASSES}, sample_sub rows={len(sample_sub)}")
print(f"TEST_DIR exists: {TEST_DIR.exists()}")
"""

CELL4 = """# === Locate exp102 OV IR (3-fold) ===
EXP102_DIR = None
for p in [
    Path("/kaggle/input/birdclef2026-exp102-3fold-ov"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp102-3fold-ov"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp102-3fold-ov"),
]:
    if p.exists():
        EXP102_DIR = p; break
if EXP102_DIR is None:
    hits = glob.glob("/kaggle/input/**/exp102_fold*.xml", recursive=True)
    if hits:
        EXP102_DIR = Path(hits[0]).parent
assert EXP102_DIR is not None, "exp102 OV IR not found"
print(f"EXP102_DIR: {EXP102_DIR}")

FOLDS = [0, 1, 2]
IR_PATHS = {}
for f in FOLDS:
    hits = list(EXP102_DIR.rglob(f"exp102_fold{f}.xml"))
    assert hits, f"exp102_fold{f}.xml not found"
    IR_PATHS[f] = hits[0]
    print(f"  fold {f}: {hits[0]}")
"""

CELL5 = """# === Compile 3 OV models ===
core = ov.Core()
compiled = {f: core.compile_model(str(IR_PATHS[f]), "CPU") for f in FOLDS}
print(f"compiled {len(compiled)} OV models")
"""

CELL6 = """# === Locate test audio files (fallback to train_soundscapes for dry-run) ===
if TEST_DIR.exists():
    test_files = sorted(TEST_DIR.glob("*.ogg"))
else:
    test_files = []
if not test_files:
    train_sc_dir = BASE / "train_soundscapes"
    test_files = sorted(train_sc_dir.glob("*.ogg"))[:20]  # dry-run
    print(f"[DRY-RUN] using {len(test_files)} train_soundscape files")
else:
    print(f"hidden test_soundscapes: {len(test_files)} files")
"""

CELL7 = """# === Mel transform ===
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
)
db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)


def load_60s(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    target = N_WINDOWS * WINDOW_SAMPLES
    if len(wav) < target:
        wav = np.concatenate([wav, np.zeros(target - len(wav), dtype=np.float32)])
    else:
        wav = wav[:target]
    return wav.astype(np.float32)
"""

CELL8 = """# === Inference loop (3-fold ensemble) ===
t0 = time.time()
all_probs = []  # per file (12, N_CLASSES)
file_ids = []

with torch.no_grad():
    for fi, fpath in enumerate(test_files):
        raw_60s = load_60s(fpath)
        chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
        wav_t = torch.from_numpy(chunks).unsqueeze(1)  # (12, 1, 160000)
        mel = db_tf(mel_tf(wav_t))
        # per-instance standardize (matches exp029 R3 train/infer)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        mel_np = mel.numpy()

        clip_sum = None; frame_sum = None
        for f, c in compiled.items():
            ov_out = c(mel_np)
            clip_ov = ov_out[c.outputs[0]]
            frame_ov = ov_out[c.outputs[1]]
            if clip_sum is None:
                clip_sum = clip_ov.copy()
                frame_sum = frame_ov.copy()
            else:
                clip_sum += clip_ov
                frame_sum += frame_ov
        clip = clip_sum / len(compiled)
        frame = frame_sum / len(compiled)
        frame_max = frame.max(axis=1)

        p_clip = 1.0 / (1.0 + np.exp(-np.clip(clip, -50, 50)))
        p_frame = 1.0 / (1.0 + np.exp(-np.clip(frame_max, -50, 50)))
        p_mean = (0.5 * p_clip + 0.5 * p_frame).astype(np.float32)
        p_smooth = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        all_probs.append(p_smooth)

        fname_no_ext = fpath.stem
        for w in range(N_WINDOWS):
            end_sec = (w + 1) * WINDOW_SEC
            file_ids.append(f"{fname_no_ext}_{end_sec}")

        if (fi + 1) % 50 == 0 or fi == len(test_files) - 1:
            print(f"  [{fi+1}/{len(test_files)}] {time.time()-t0:.0f}s")

all_probs = np.concatenate(all_probs, axis=0)  # (n_files * 12, N_CLASSES)
print(f"all_probs shape: {all_probs.shape}, time: {time.time()-t0:.0f}s")
"""

CELL9 = """# === Build submission.csv (align to sample_submission row_ids) ===
preds_df = pd.DataFrame(all_probs, columns=PRIMARY_LABELS)
preds_df.insert(0, "row_id", file_ids)

# align to sample_sub row order (fill missing with 0)
sub = sample_sub[["row_id"]].merge(preds_df, on="row_id", how="left")
sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(0.0)
n_missing = (sub[PRIMARY_LABELS].sum(axis=1) == 0).sum()
print(f"missing row_ids (filled 0): {n_missing}")

sub.to_csv("submission.csv", index=False)
print(f"submission.csv: shape {sub.shape}")
print(f"  mean pred: {sub[PRIMARY_LABELS].values.mean():.6f}")
print(f"  max pred: {sub[PRIMARY_LABELS].values.max():.6f}")
print(sub.head(3))
"""

CELLS = [
    (CELL1, "imports"),
    (CELL2, "ov-install"),
    (CELL3, "const"),
    (CELL4, "locate"),
    (CELL5, "compile"),
    (CELL6, "test-files"),
    (CELL7, "mel-transform"),
    (CELL8, "infer-loop"),
    (CELL9, "submission"),
]

nb = {
    "cells": [cell(s, cid) for s, cid in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes, {len(CELLS)} cells)")

# compile check
for src, cid in CELLS:
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<{cid}>", "exec")
        print(f"  [OK] {cid}")
    except SyntaxError as e:
        print(f"  [FAIL] {cid}: {e}")
