"""Generate exp013 NB1: Pseudo-label generation using Tucker public 5-fold SED ONNX.

Babych H1 recipe applied:
- 5-fold SED ensemble (Tucker public ONNX)
- Temporal smoothing (uniform_filter1d, size=3)
- Power Transform (gamma=2.0) — sharpens distribution
- Dynamic class-specific threshold (70th percentile per class)

Input: train_soundscapes (10,658 files, 60s each)
Output: pseudo_labels.csv (Kaggle Dataset 化 → NB2 で参照)

Hardware: Kaggle GPU T4x2 (12h limit)
Estimated runtime: ~1.5-2.5h
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

# ── Header ──
cells.append(md_cell("hdr",
    "# exp013 NB1: 擬似ラベル生成 (Babych H1 recipe)\n"
    "\n"
    "Tucker 公開 5-fold SED ONNX で train_soundscapes 全 10,658 ファイルを推論し、"
    "Babych Multi-Iterative Noisy Student の H1 recipe で擬似ラベル CSV を生成。\n"
    "\n"
    "**処理パイプライン**\n"
    "1. mel spectrogram (256 mels, 32 kHz, 5sec window) を 12 windows ぶん算出\n"
    "2. Tucker SED 5-fold ONNX で `0.5*sigmoid(clip) + 0.5*sigmoid(frame_max)` 平均\n"
    "3. **Temporal smoothing** (uniform_filter1d, size=3、time 軸)\n"
    "4. **Power Transform** $P^{\\gamma=2.0}$ で sharpen (低 conf を 0 に、高 conf を保持)\n"
    "5. **Dynamic threshold** 各 class の 70th percentile を算出 → どの class も超えてない window は捨てる\n"
    "\n"
    "**出力**\n"
    "- `/kaggle/working/pseudo_labels.csv` (filename, start_sec, end_sec, label_0...label_233)\n"
    "- これを Kaggle Dataset 化して NB2 (Colab) から読み込む\n"
    "\n"
    "**ハードウェア**: Kaggle GPU T4x2 (12h 上限、想定 1.5-2.5h)"))

# ── Install ──
INSTALL = '''import subprocess, sys, os, time
START = time.time()

# onnxruntime-gpu (Tucker ONNX を CUDA で実行する)
# NumPy 2.x 互換のため >=1.19 が必要 (Kaggle GPU env は Python 3.12 + NumPy 2.x)
# 既に壊れた 1.18 がインストールされている可能性があるので uninstall してから入れる
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "onnxruntime", "onnxruntime-gpu"], check=False)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "onnxruntime-gpu"])

# import を強制リロード (前のセルで壊れた module が sys.modules に残っている可能性)
for _mod_name in list(sys.modules):
    if _mod_name.startswith("onnxruntime"):
        del sys.modules[_mod_name]

import onnxruntime as ort
print(f"onnxruntime {ort.__version__} (providers: {ort.get_available_providers()})")
assert "CUDAExecutionProvider" in ort.get_available_providers(), \\
    "CUDAExecutionProvider missing — GPU not enabled?"
'''
cells.append(code_cell("install", INSTALL))

# ── Imports & paths ──
IMPORTS = '''import glob
import re
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# Find competition data dir robustly (Kaggle path varies by mount style)
BASE = None
for _c in [
    Path("/kaggle/input/birdclef-2026"),
    Path("/kaggle/input/competitions/birdclef-2026"),
]:
    if (_c / "taxonomy.csv").exists():
        BASE = _c
        break
if BASE is None:
    print("Searching for taxonomy.csv under /kaggle/input ...")
    for _hit in Path("/kaggle/input").rglob("taxonomy.csv"):
        BASE = _hit.parent
        break
assert BASE is not None, "birdclef-2026 dataset not attached or taxonomy.csv missing"
print(f"BASE: {BASE}")
TRAIN_SC_DIR = BASE / "train_soundscapes"
assert TRAIN_SC_DIR.is_dir(), f"train_soundscapes dir missing: {TRAIN_SC_DIR}"

# Find Tucker SED 5-fold ONNX
SED_DIR = None
for hit in Path("/kaggle/input").rglob("sed_fold0.onnx"):
    SED_DIR = hit.parent
    break
assert SED_DIR is not None, "Tucker SED ONNX not found — attach tuckerarrants/bc2026-distilled-sed-public"
print(f"SED dir: {SED_DIR}")

# Taxonomy (234 species, ordered by primary_label)
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
PRIMARY_LABELS = sorted(taxonomy["primary_label"].astype(str).tolist())
N_CLASSES = len(PRIMARY_LABELS)
print(f"N_CLASSES: {N_CLASSES}")
assert N_CLASSES == 234, f"Expected 234 species, got {N_CLASSES}"
'''
cells.append(code_cell("imports", IMPORTS))

# ── Config ──
CONFIG = '''# Audio / mel spec (Tucker public ONNX 仕様、mattiaangeli 0.943 NB から抽出)
SR          = 32000
WINDOW_SEC  = 5.0
WINDOW_SAMP = int(WINDOW_SEC * SR)
N_WINDOWS   = 12               # 60sec / 5sec
N_MELS      = 256
N_FFT       = 2048
HOP         = 512
FMIN        = 20
FMAX        = 16000
TOP_DB      = 80

# Babych post-processing
SMOOTH_SIZE        = 3            # uniform_filter1d, time 軸
GAUSS_SIGMA        = 0.65         # gaussian_filter1d, mattiaangeli 0.943 NB と同設定
POWER_GAMMA        = 2.0          # ★ Babych H1 — Power Transform
THRESHOLD_PCTILE   = 0.70         # 各 class の 70%ile を超える window を採用

# SED inference batch
SED_BATCH = 12   # 1 file = 12 windows、1 file ずつ処理
'''
cells.append(code_cell("config", CONFIG))

# ── Audio file list ──
FILES = '''sc_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))
print(f"train_soundscapes: {len(sc_files)} files")
assert len(sc_files) > 0, "No soundscape files found"
'''
cells.append(code_cell("files", FILES))

# ── Audio loading ──
AUDIO = '''def read_audio(path, target_samples=SR * 60):
    """Read .ogg, return mono float32 array of exactly target_samples length."""
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
    """chunks: (N_WINDOWS, WINDOW_SAMP) -> mel (N_WINDOWS, 1, N_MELS, T)."""
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

# ── SED ONNX session ──
# 重要: 全 fold を GPU 0 単一で実行する。
# T4x2 で同一プロセスから GPU 0/1 両方使うと
# "CUDA failure 400: invalid resource handle" が発生する (multi-context 問題)。
# emanuellcs の参考 NB は multiprocessing で GPU 分割していたが、
# 我々は速度より安定性を取る。T4 16GB に 5 セッション load して問題なし。
SED_SESS = '''def make_sed_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 2
    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,                          # ★ GPU 0 単独 (multi-GPU の context 問題回避)
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


sed_paths = sorted(SED_DIR.glob("sed_fold*.onnx"),
                   key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1)))
print(f"SED folds: {[p.name for p in sed_paths]}")
sed_sessions = [make_sed_session(p) for p in sed_paths]
print(f"Loaded {len(sed_sessions)} SED sessions on GPU 0")
print(f"Session 0 providers: {sed_sessions[0].get_providers()}")
print(f"Session 0 input: {sed_sessions[0].get_inputs()[0].name} {sed_sessions[0].get_inputs()[0].shape}")


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def sed_infer_one_file(mel):
    """mel: (12, 1, 256, T) -> probs (12, 234) averaged over folds."""
    p_sum = np.zeros((mel.shape[0], N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]                # (12, 234)
        frame_max   = outs[1].max(axis=1)    # (12, 234)
        p_sum += 0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)
    p_mean = p_sum / len(sed_sessions)
    # gaussian smoothing across windows (mattiaangeli 0.943 NB と同様)
    p_mean = gaussian_filter1d(p_mean, sigma=GAUSS_SIGMA, axis=0, mode="nearest").astype(np.float32)
    return p_mean   # (12, 234)


# === Sanity check: 1 file 推論して shape/値を確認 ===
print("\\n=== Sanity check ===")
_test_path = sc_files[0]
print(f"Test file: {Path(_test_path).name}")
_y = read_audio(_test_path, target_samples=N_WINDOWS * WINDOW_SAMP)
print(f"  audio shape: {_y.shape}")
_chunks = _y.reshape(N_WINDOWS, WINDOW_SAMP)
_mel = audio_to_mel(_chunks)
print(f"  mel shape: {_mel.shape}, dtype={_mel.dtype}")
_probs = sed_infer_one_file(_mel)
print(f"  probs shape: {_probs.shape}, mean={_probs.mean():.4f}, max={_probs.max():.4f}")
assert _probs.shape == (N_WINDOWS, N_CLASSES), f"Shape mismatch: {_probs.shape}"
print("  Sanity check PASSED")
'''
cells.append(code_cell("sed-sess", SED_SESS))

# ── Main inference loop ──
INFER = '''all_rows = []   # list of (filename, start_sec, end_sec, prob[234])
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
    probs = sed_infer_one_file(mel)   # (12, 234) — already gaussian-smoothed

    for wi in range(N_WINDOWS):
        start_sec = wi * WINDOW_SEC
        end_sec   = (wi + 1) * WINDOW_SEC
        all_rows.append((stem, start_sec, end_sec, probs[wi].astype(np.float32)))

    if (fi + 1) % 100 == 0 or fi == N_FILES - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / elapsed
        eta = (N_FILES - fi - 1) / rate
        print(f"  [{fi+1}/{N_FILES}] {elapsed:.0f}s, {rate:.1f} files/s, ETA {eta/60:.1f} min")

print(f"\\nInference done: {len(all_rows)} rows in {time.time()-t0:.0f}s")
'''
cells.append(code_cell("infer", INFER))

# ── Babych post-processing ──
POST = '''# Stack into matrix shape: (N_rows, 234)
filenames  = [r[0] for r in all_rows]
start_secs = np.array([r[1] for r in all_rows], dtype=np.float32)
end_secs   = np.array([r[2] for r in all_rows], dtype=np.float32)
prob_mat   = np.stack([r[3] for r in all_rows], axis=0)  # (N_rows, 234)
print(f"Probs matrix: {prob_mat.shape}")

# 中間保存 (post-processing が失敗しても raw 結果を救済)
np.savez_compressed(
    "/kaggle/working/raw_probs.npz",
    prob_mat=prob_mat,
    filenames=np.array(filenames),
    start_secs=start_secs,
    end_secs=end_secs,
)
print(f"  Raw saved: /kaggle/working/raw_probs.npz")

# === Babych H1 — Temporal smoothing (uniform_filter1d, size=3) ===
# Note: Already applied gaussian smoothing per file. emanuellcs's recipe also
# does uniform_filter1d size=3 on top across all rows (within file).
# We do per-file smoothing here to avoid cross-file leakage.
print("Applying per-file uniform_filter1d (size=3)...")
fname_array = np.array(filenames)
unique_files = np.unique(fname_array)
for fname in unique_files:
    idx = np.where(fname_array == fname)[0]
    if len(idx) >= 3:
        prob_mat[idx] = uniform_filter1d(prob_mat[idx], size=SMOOTH_SIZE, axis=0, mode="nearest")

# === Babych H1 — Power Transform (gamma=2.0) ===
print(f"Applying Power Transform (gamma={POWER_GAMMA})...")
prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)

print(f"Post-PT: mean={prob_mat.mean():.6f}, max={prob_mat.max():.4f}, "
      f"99%ile={np.percentile(prob_mat, 99):.4f}")
'''
cells.append(code_cell("post", POST))

# ── Dynamic thresholding & save ──
SAVE = '''# === Dynamic class-specific threshold (70th percentile per class) ===
# 各 class について prob_mat[:, k] の 70%ile を計算
# どの class でもその閾値を超えなかった row は信号がない → 捨てる
class_thresholds = np.quantile(prob_mat, THRESHOLD_PCTILE, axis=0)  # (234,)
print(f"Class thresholds: mean={class_thresholds.mean():.4f}, "
      f"min={class_thresholds.min():.4f}, max={class_thresholds.max():.4f}")

keep_mask = (prob_mat >= class_thresholds[None, :]).any(axis=1)   # (N_rows,)
n_kept = keep_mask.sum()
n_total = len(keep_mask)
print(f"Keep mask: {n_kept}/{n_total} ({n_kept/n_total*100:.1f}%) rows kept")

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

# Save thresholds for inspection
np.save("/kaggle/working/class_thresholds.npy", class_thresholds)
print(f"Saved class_thresholds.npy")

# Preview
print("\\n=== Preview ===")
print(df.head(3))
print("\\n=== Per-row max prob ===")
row_max = prob_mat[keep_mask].max(axis=1)
print(f"  mean={row_max.mean():.4f}, median={np.median(row_max):.4f}, "
      f"min={row_max.min():.4f}, max={row_max.max():.4f}")
'''
cells.append(code_cell("save", SAVE))

# ── Notebook assembly ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_pl1_pseudo.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
