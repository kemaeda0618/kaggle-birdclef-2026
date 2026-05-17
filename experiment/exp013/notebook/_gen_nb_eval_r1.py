"""Generate exp013 R1 eval NB: macro-AUC on labeled train_soundscapes.

Quick validation NB (Kaggle CPU, ~10-20 min) to determine if R1 student is
genuinely good/bad before burning more LB submissions.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr",
    "# exp013 R1 Eval: macro-AUC on labeled train_soundscapes\n"
    "\n"
    "Sub 消費せずに R1 student の品質を直接測る。Tucker fold 0 と同じ評価。\n"
    "\n"
    "**結果の解釈**\n"
    "- R1 macro-AUC ≥ 0.85: 高品質、blend 構成が悪い (重み調整で改善余地)\n"
    "- 0.75-0.85: 中品質、低重みで併用\n"
    "- < 0.75: 低品質、R1 路線撤退\n"
    "- anti-correlated クラス (AUC<0.5) が多い (>30): 致命的、撤退\n"
))

cells.append(code_cell("install", '''import time, os, sys, json, glob, re
START = time.time()

# onnxruntime 確認
try:
    import onnxruntime as ort
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
    import onnxruntime as ort
print(f"onnxruntime {ort.__version__}")
'''))

cells.append(code_cell("imports", '''import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path

# Find BC2026 data
BASE = None
for c in [Path("/kaggle/input/birdclef-2026"),
          Path("/kaggle/input/competitions/birdclef-2026")]:
    if (c / "taxonomy.csv").exists():
        BASE = c
        break
assert BASE is not None
print(f"BASE: {BASE}")

# Find R1 student ONNX
R1_PATH = None
for h in Path("/kaggle/input").rglob("student_sed_fold0.onnx"):
    R1_PATH = h
    break
assert R1_PATH, "Attach maekeso/birdclef2026-exp013-r1-student-sed"
print(f"R1: {R1_PATH}")

# Find Tucker fold 0
TUCKER_PATH = None
for h in Path("/kaggle/input").rglob("sed_fold0.onnx"):
    TUCKER_PATH = h
    break
assert TUCKER_PATH, "Attach tuckerarrants/bc2026-distilled-sed-public"
print(f"Tucker: {TUCKER_PATH}")
'''))

cells.append(code_cell("labels", '''# Build label matrix
labels_df = pd.read_csv(BASE / "train_soundscapes_labels.csv").drop_duplicates()
labels_df["start_sec"] = pd.to_timedelta(labels_df["start"]).dt.total_seconds().astype(int)

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
PRIMARY = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY)}
N_CLASSES = len(PRIMARY)

unique_windows = labels_df[["filename", "start_sec"]].drop_duplicates().reset_index(drop=True)
Y = np.zeros((len(unique_windows), N_CLASSES), dtype=np.float32)
for i, row in unique_windows.iterrows():
    matches = labels_df[(labels_df["filename"] == row["filename"]) &
                        (labels_df["start_sec"] == row["start_sec"])]
    for _, m in matches.iterrows():
        for lbl in str(m["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in LABEL2IDX:
                Y[i, LABEL2IDX[lbl]] = 1.0

print(f"Eval windows: {len(unique_windows)}")
print(f"Total positives: {int(Y.sum())}")
print(f"Classes with positives: {int((Y.sum(axis=0) > 0).sum())}/{N_CLASSES}")

# Add S22 mask (Tucker excludes S22 in training eval — known noise)
unique_windows["site"] = unique_windows["filename"].str.extract(r"_S(\\d+)_")[0]
unique_windows["is_s22"] = unique_windows["site"] == "22"
print(f"S22 windows: {unique_windows['is_s22'].sum()}, non-S22: {(~unique_windows['is_s22']).sum()}")
'''))

cells.append(code_cell("inference", '''# Setup mel + ONNX
SR = 32000
WINDOW = 5 * SR
N_FFT = 2048
HOP = 512
N_MELS = 256
FMIN = 20
FMAX = 16000


def read_window(filepath, start_sec):
    y, _sr = sf.read(str(filepath), start=int(start_sec*SR), frames=WINDOW, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if len(y) < WINDOW: y = np.pad(y, (0, WINDOW - len(y)))
    return y


def to_mel_lib_slaney(wav):
    s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=80)
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s[None, None].astype(np.float32)


# torchaudio mel for R1 (matches NB2 training)
import torch, torchaudio
mel_t = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP,
                                              n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0)
db_t = torchaudio.transforms.AmplitudeToDB(top_db=80)


def to_mel_torchaudio(wav):
    w = torch.from_numpy(wav[None].astype(np.float32))
    s = db_t(mel_t(w))
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s.unsqueeze(1).numpy().astype(np.float32)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
r1_sess = ort.InferenceSession(str(R1_PATH), sess_options=so, providers=["CPUExecutionProvider"])
tucker_sess = ort.InferenceSession(str(TUCKER_PATH), sess_options=so, providers=["CPUExecutionProvider"])
print(f"R1 input:     {r1_sess.get_inputs()[0].name} {r1_sess.get_inputs()[0].shape}")
print(f"Tucker input: {tucker_sess.get_inputs()[0].name} {tucker_sess.get_inputs()[0].shape}")

TRAIN_SC = BASE / "train_soundscapes"
preds_r1 = np.zeros((len(unique_windows), N_CLASSES), dtype=np.float32)
preds_tucker = np.zeros((len(unique_windows), N_CLASSES), dtype=np.float32)

t0 = time.time()
for i, row in unique_windows.iterrows():
    fp = TRAIN_SC / row["filename"]
    if not fp.exists():
        print(f"  Missing: {fp}")
        continue
    wav = read_window(fp, row["start_sec"])

    # R1 (torchaudio mel)
    mel_r = to_mel_torchaudio(wav)
    out = r1_sess.run(None, {r1_sess.get_inputs()[0].name: mel_r})
    clip = sigmoid(out[0][0])
    fmax = sigmoid(out[1][0].max(axis=0))
    preds_r1[i] = 0.5*clip + 0.5*fmax

    # Tucker (librosa Slaney mel)
    mel_t_lib = to_mel_lib_slaney(wav)
    out = tucker_sess.run(None, {tucker_sess.get_inputs()[0].name: mel_t_lib})
    clip = sigmoid(out[0][0])
    fmax = sigmoid(out[1][0].max(axis=0))
    preds_tucker[i] = 0.5*clip + 0.5*fmax

    if (i+1) % 100 == 0:
        rate = (i+1) / (time.time() - t0)
        eta = (len(unique_windows) - i - 1) / rate
        print(f"  [{i+1}/{len(unique_windows)}] {time.time()-t0:.1f}s, ETA {eta:.0f}s")

print(f"\\nInference done in {time.time()-t0:.1f}s")
print(f"R1     stats: mean={preds_r1.mean():.4f}, std={preds_r1.std():.4f}, has_nan={np.isnan(preds_r1).any()}")
print(f"Tucker stats: mean={preds_tucker.mean():.4f}, std={preds_tucker.std():.4f}, has_nan={np.isnan(preds_tucker).any()}")
'''))

cells.append(code_cell("auc", '''# Macro-AUC comparison
from sklearn.metrics import roc_auc_score


def macro_auc(y_true, y_pred, mask=None):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    aucs = []
    for c in range(N_CLASSES):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except:
            pass
    return np.array(aucs)


print("=== All windows ===")
r1_aucs = macro_auc(Y, preds_r1)
tucker_aucs = macro_auc(Y, preds_tucker)
print(f"R1     macro-AUC: {r1_aucs.mean():.4f} (n={len(r1_aucs)})")
print(f"Tucker macro-AUC: {tucker_aucs.mean():.4f} (n={len(tucker_aucs)})")
print(f"  R1     median={np.median(r1_aucs):.4f}, min={r1_aucs.min():.4f}, max={r1_aucs.max():.4f}")
print(f"  Tucker median={np.median(tucker_aucs):.4f}, min={tucker_aucs.min():.4f}, max={tucker_aucs.max():.4f}")
print(f"\\nR1     classes >0.7: {(r1_aucs > 0.7).sum()}/{len(r1_aucs)}")
print(f"Tucker classes >0.7: {(tucker_aucs > 0.7).sum()}/{len(tucker_aucs)}")
print(f"R1     classes <0.5 (anti-correlated): {(r1_aucs < 0.5).sum()}/{len(r1_aucs)}")
print(f"Tucker classes <0.5 (anti-correlated): {(tucker_aucs < 0.5).sum()}/{len(tucker_aucs)}")

# Per-class diff
print("\\n=== R1 vs Tucker per-class diff ===")
# Compute aligned AUCs (only classes evaluable for both)
diffs = []
for c in range(N_CLASSES):
    col = Y[:, c]
    if col.sum() == 0 or col.sum() == len(col):
        continue
    try:
        a_r1 = roc_auc_score(col, preds_r1[:, c])
        a_tu = roc_auc_score(col, preds_tucker[:, c])
        diffs.append(a_r1 - a_tu)
    except:
        pass
diffs = np.array(diffs)
print(f"R1-Tucker diff: mean={diffs.mean():+.4f}, median={np.median(diffs):+.4f}")
print(f"  R1 better: {(diffs > 0).sum()}/{len(diffs)}")
print(f"  R1 worse:  {(diffs < 0).sum()}/{len(diffs)}")

# Non-S22 (Tucker training exclude site)
print("\\n=== Non-S22 only ===")
ns22 = ~unique_windows["is_s22"].values
r1_aucs_n = macro_auc(Y, preds_r1, mask=ns22)
tucker_aucs_n = macro_auc(Y, preds_tucker, mask=ns22)
print(f"R1     non-S22: {r1_aucs_n.mean():.4f} (n={len(r1_aucs_n)})")
print(f"Tucker non-S22: {tucker_aucs_n.mean():.4f} (n={len(tucker_aucs_n)})")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_eval_r1.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
