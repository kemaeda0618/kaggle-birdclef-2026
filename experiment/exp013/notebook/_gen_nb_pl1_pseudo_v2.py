"""Generate exp013 NB1 v2: Pseudo-label generation with improved filter & gamma.

NB1 v1 の問題:
- Dynamic class threshold (70%ile per class) が OR 条件で実質 100% pass-through
  → 沈黙 window 大量に残った
- Power Transform γ=2.0 が BC2026 sparse data で強すぎ
  → 中-低 conf を near-zero に押し下げ、student が trivial 解に収束

NB1 v2 の修正:
- **row_max filter**: row.max() < 0.05 の沈黙 window を除外
- **γ 緩和**: 2.0 → 1.2 (信号維持しつつ若干 sharpen)

Hardware: Kaggle GPU T4x2 (12h limit)
Estimated runtime: ~75 min
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code", "id": cell_id, "metadata": {},
        "outputs": [], "execution_count": None,
        "source": src,
    }


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": src,
    }


cells = []

cells.append(md_cell("hdr",
    "# exp013 NB1 v2: 擬似ラベル生成 (改善版)\n"
    "\n"
    "v1 → v2 の修正:\n"
    "- **row_max filter** を row.max() < 0.05 で適用 (v1 の class 別 70%ile OR は実質 100% pass-through で機能してなかった)\n"
    "- **Power Transform γ=2.0 → 1.2** (BC2026 sparse data 用に緩和)\n"
    "- それ以外は v1 と同じ (Tucker SED 5-fold ensemble、gaussian + uniform smoothing)\n"
    "\n"
    "**期待される改善**:\n"
    "- pseudo CSV 行数: 127,896 → ~30,000-60,000 (沈黙 window 削減)\n"
    "- student の trivial 解への collapse を回避\n"
    "- R1 v2 の AUC 上昇期待 (vs R1 v1 の 0.67)"))

INSTALL = '''import subprocess, sys, os, time
START = time.time()

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "onnxruntime", "onnxruntime-gpu"], check=False)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "onnxruntime-gpu"])

for _mod_name in list(sys.modules):
    if _mod_name.startswith("onnxruntime"):
        del sys.modules[_mod_name]

import onnxruntime as ort
print(f"onnxruntime {ort.__version__} (providers: {ort.get_available_providers()})")
assert "CUDAExecutionProvider" in ort.get_available_providers(), \\
    "CUDAExecutionProvider missing — GPU not enabled?"
'''
cells.append(code_cell("install", INSTALL))

IMPORTS = '''import glob
import re
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

BASE = None
for _c in [
    Path("/kaggle/input/birdclef-2026"),
    Path("/kaggle/input/competitions/birdclef-2026"),
]:
    if (_c / "taxonomy.csv").exists():
        BASE = _c
        break
if BASE is None:
    for _hit in Path("/kaggle/input").rglob("taxonomy.csv"):
        BASE = _hit.parent
        break
assert BASE is not None
print(f"BASE: {BASE}")
TRAIN_SC_DIR = BASE / "train_soundscapes"
assert TRAIN_SC_DIR.is_dir()

SED_DIR = None
for hit in Path("/kaggle/input").rglob("sed_fold0.onnx"):
    SED_DIR = hit.parent
    break
assert SED_DIR is not None, "Tucker SED ONNX not found"
print(f"SED dir: {SED_DIR}")

taxonomy = pd.read_csv(BASE / "taxonomy.csv")
PRIMARY_LABELS = sorted(taxonomy["primary_label"].astype(str).tolist())
N_CLASSES = len(PRIMARY_LABELS)
assert N_CLASSES == 234
print(f"N_CLASSES: {N_CLASSES}")
'''
cells.append(code_cell("imports", IMPORTS))

CONFIG = '''SR          = 32000
WINDOW_SEC  = 5.0
WINDOW_SAMP = int(WINDOW_SEC * SR)
N_WINDOWS   = 12
N_MELS      = 256
N_FFT       = 2048
HOP         = 512
FMIN        = 20
FMAX        = 16000
TOP_DB      = 80

SMOOTH_SIZE        = 3
GAUSS_SIGMA        = 0.65
POWER_GAMMA        = 1.2          # ★ v2: 2.0 → 1.2 (緩和)
ROW_MAX_THRESHOLD  = 0.05         # ★ v2: row.max() < 0.05 → 沈黙 window と判定して除外

print(f"v2 config: γ={POWER_GAMMA}, row_max threshold={ROW_MAX_THRESHOLD}")
'''
cells.append(code_cell("config", CONFIG))

cells.append(code_cell("files", '''sc_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))
print(f"train_soundscapes: {len(sc_files)} files")
assert len(sc_files) > 0
'''))

AUDIO = '''def read_audio(path, target_samples=SR * 60):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y


def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)
'''
cells.append(code_cell("audio", AUDIO))

SED_SESS = '''def make_sed_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 2
    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


sed_paths = sorted(SED_DIR.glob("sed_fold*.onnx"),
                   key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1)))
sed_sessions = [make_sed_session(p) for p in sed_paths]
print(f"Loaded {len(sed_sessions)} SED sessions on GPU 0")


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def sed_infer_one_file(mel):
    p_sum = np.zeros((mel.shape[0], N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max   = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)
    p_mean = p_sum / len(sed_sessions)
    p_mean = gaussian_filter1d(p_mean, sigma=GAUSS_SIGMA, axis=0, mode="nearest").astype(np.float32)
    return p_mean


# Sanity check
print("\\n=== Sanity check ===")
_test_path = sc_files[0]
_y = read_audio(_test_path, target_samples=N_WINDOWS * WINDOW_SAMP)
_chunks = _y.reshape(N_WINDOWS, WINDOW_SAMP)
_mel = audio_to_mel(_chunks)
_probs = sed_infer_one_file(_mel)
print(f"  probs shape: {_probs.shape}, mean={_probs.mean():.4f}, max={_probs.max():.4f}")
assert _probs.shape == (N_WINDOWS, N_CLASSES)
print("  PASSED")
'''
cells.append(code_cell("sed-sess", SED_SESS))

INFER = '''all_rows = []
t0 = time.time()
N_FILES = len(sc_files)

for fi, fpath in enumerate(sc_files):
    stem = Path(fpath).stem
    try:
        y = read_audio(fpath, target_samples=N_WINDOWS * WINDOW_SAMP)
    except Exception as e:
        print(f"WARN: failed to read {stem}: {e}")
        continue
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMP)
    mel = audio_to_mel(chunks)
    probs = sed_infer_one_file(mel)

    for wi in range(N_WINDOWS):
        start_sec = wi * WINDOW_SEC
        end_sec   = (wi + 1) * WINDOW_SEC
        all_rows.append((stem, start_sec, end_sec, probs[wi].astype(np.float32)))

    if (fi + 1) % 200 == 0 or fi == N_FILES - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / elapsed
        eta = (N_FILES - fi - 1) / rate
        print(f"  [{fi+1}/{N_FILES}] {elapsed:.0f}s, {rate:.1f} files/s, ETA {eta/60:.1f} min")

print(f"\\nInference done: {len(all_rows)} rows in {time.time()-t0:.0f}s")
'''
cells.append(code_cell("infer", INFER))

POST = '''filenames  = [r[0] for r in all_rows]
start_secs = np.array([r[1] for r in all_rows], dtype=np.float32)
end_secs   = np.array([r[2] for r in all_rows], dtype=np.float32)
prob_mat   = np.stack([r[3] for r in all_rows], axis=0)
print(f"Probs matrix: {prob_mat.shape}")

# 中間保存 (post-processing 失敗時の救済)
np.savez_compressed(
    "/kaggle/working/raw_probs_v2.npz",
    prob_mat=prob_mat,
    filenames=np.array(filenames),
    start_secs=start_secs,
    end_secs=end_secs,
)

# === per-file uniform_filter1d (size=3) ===
print("Applying per-file uniform_filter1d...")
fname_array = np.array(filenames)
unique_files = np.unique(fname_array)
for fname in unique_files:
    idx = np.where(fname_array == fname)[0]
    if len(idx) >= 3:
        prob_mat[idx] = uniform_filter1d(prob_mat[idx], size=SMOOTH_SIZE, axis=0, mode="nearest")

# === Power Transform (γ=1.2、v2 で緩和) ===
print(f"Applying Power Transform (gamma={POWER_GAMMA})...")
prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)

print(f"Post-PT: mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}, "
      f"95%ile={np.percentile(prob_mat, 95):.4f}, "
      f"50%ile={np.percentile(prob_mat, 50):.4f}")
'''
cells.append(code_cell("post", POST))

SAVE = '''# === ★ v2: row_max filter (実際に効くフィルタ) ===
row_max = prob_mat.max(axis=1)
keep_mask = row_max >= ROW_MAX_THRESHOLD
n_kept = keep_mask.sum()
n_total = len(keep_mask)
print(f"\\nrow_max filter (>= {ROW_MAX_THRESHOLD}):")
print(f"  Kept: {n_kept}/{n_total} ({n_kept/n_total*100:.1f}%) rows")
print(f"  Dropped: {n_total - n_kept} silent windows")
print(f"  row_max distribution:")
print(f"    25%ile: {np.percentile(row_max, 25):.4f}")
print(f"    50%ile: {np.percentile(row_max, 50):.4f}")
print(f"    75%ile: {np.percentile(row_max, 75):.4f}")
print(f"    99%ile: {np.percentile(row_max, 99):.4f}")

# Build output DataFrame
df = pd.DataFrame(prob_mat[keep_mask], columns=PRIMARY_LABELS)
df.insert(0, "filename",  np.array(filenames)[keep_mask])
df.insert(1, "start_sec", start_secs[keep_mask])
df.insert(2, "end_sec",   end_secs[keep_mask])

out_path = Path("/kaggle/working/pseudo_labels.csv")
df.to_csv(out_path, index=False)
print(f"\\nSaved: {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Total time: {(time.time()-START)/60:.1f} min")

print("\\n=== Preview ===")
print(df.head(3))

print("\\n=== Per-row max prob (kept rows) ===")
kept_max = row_max[keep_mask]
print(f"  mean={kept_max.mean():.4f}, median={np.median(kept_max):.4f}, "
      f"min={kept_max.min():.4f}, max={kept_max.max():.4f}")
'''
cells.append(code_cell("save", SAVE))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_pl1_pseudo_v2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
