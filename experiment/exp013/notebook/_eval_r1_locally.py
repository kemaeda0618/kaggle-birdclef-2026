"""Evaluate R1 student SED on labeled train_soundscapes (no sub burn).

Downloads:
- R1 ONNX from maekeso/birdclef2026-exp013-r1-student-sed
- Tucker fold 0 ONNX from tuckerarrants/bc2026-distilled-sed-public (for comparison)
- train_soundscapes_labels.csv from BC2026 competition
- Only the .ogg files referenced in labels (not full 25GB)

Computes:
- macro-AUC for R1 on 792 labeled windows
- macro-AUC for Tucker fold 0 (baseline)
- Direct comparison
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

WORK = Path("./eval_r1")
WORK.mkdir(exist_ok=True)

# ---------- 1. Download R1 ONNX ----------
print("=== Downloading R1 ONNX ===")
r1_dir = WORK / "r1"
r1_dir.mkdir(exist_ok=True)
if not (r1_dir / "student_sed_fold0.onnx").exists():
    api.dataset_download_files(
        "maekeso/birdclef2026-exp013-r1-student-sed",
        path=str(r1_dir), unzip=True, quiet=False,
    )
print(f"R1 dir: {list(r1_dir.iterdir())}")

# ---------- 2. Download Tucker fold 0 ONNX ----------
print("\n=== Downloading Tucker fold 0 ONNX ===")
tucker_dir = WORK / "tucker"
tucker_dir.mkdir(exist_ok=True)
if not (tucker_dir / "sed_fold0.onnx").exists():
    api.dataset_download_file(
        "tuckerarrants/bc2026-distilled-sed-public",
        file_name="sed_fold0.onnx",
        path=str(tucker_dir),
    )
    # unzip if zipped
    z = tucker_dir / "sed_fold0.onnx.zip"
    if z.exists():
        import zipfile
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tucker_dir)
        z.unlink()
print(f"Tucker dir: {list(tucker_dir.iterdir())}")

# ---------- 3. Download labels + sample_submission ----------
print("\n=== Downloading labels + sample_submission ===")
comp_dir = WORK / "comp"
comp_dir.mkdir(exist_ok=True)
for fn in ["train_soundscapes_labels.csv", "sample_submission.csv", "taxonomy.csv"]:
    if not (comp_dir / fn).exists():
        api.competition_download_file("birdclef-2026", file_name=fn, path=str(comp_dir))
        # unzip if needed
        z = comp_dir / (fn + ".zip")
        if z.exists():
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(comp_dir)
            z.unlink()
print(f"Comp files: {list(comp_dir.iterdir())}")

# ---------- 4. Build labels matrix ----------
import pandas as pd
import numpy as np
labels_df = pd.read_csv(comp_dir / "train_soundscapes_labels.csv").drop_duplicates()
labels_df["start_sec"] = pd.to_timedelta(labels_df["start"]).dt.total_seconds().astype(int)
sample_sub = pd.read_csv(comp_dir / "sample_submission.csv")
PRIMARY = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY)}
print(f"\nPrimary labels: {len(PRIMARY)}")

# Need 12 windows per file (0, 5, 10, ..., 55) — eval windows are subset
unique_files = labels_df["filename"].unique()
print(f"Labeled files: {len(unique_files)}")

# Build all labeled windows
unique_windows = labels_df[["filename", "start_sec"]].drop_duplicates().reset_index(drop=True)
Y = np.zeros((len(unique_windows), 234), dtype=np.float32)
for i, row in unique_windows.iterrows():
    matches = labels_df[(labels_df["filename"] == row["filename"]) &
                        (labels_df["start_sec"] == row["start_sec"])]
    for _, m in matches.iterrows():
        for lbl in str(m["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in LABEL2IDX:
                Y[i, LABEL2IDX[lbl]] = 1.0
print(f"Eval windows: {len(unique_windows)}, total positives: {int(Y.sum())}")
print(f"Classes with positives: {int((Y.sum(axis=0) > 0).sum())}/234")

# ---------- 5. Download only the .ogg files we need ----------
print(f"\n=== Downloading {len(unique_files)} .ogg files ===")
audio_dir = WORK / "audio"
audio_dir.mkdir(exist_ok=True)
for fn in unique_files:
    target = audio_dir / fn
    if target.exists():
        continue
    try:
        api.competition_download_file("birdclef-2026",
            file_name=f"train_soundscapes/{fn}", path=str(audio_dir))
        # The api downloads to audio/train_soundscapes/{fn} as zip — handle
        z = audio_dir / (fn + ".zip")
        if z.exists():
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(audio_dir)
            z.unlink()
    except Exception as e:
        print(f"  Failed {fn}: {e}")

# Find actual audio file locations (might be nested)
ogg_paths = {}
for fn in unique_files:
    candidates = list(audio_dir.rglob(fn))
    if candidates:
        ogg_paths[fn] = candidates[0]
print(f"Found {len(ogg_paths)}/{len(unique_files)} audio files")

# ---------- 6. Build mel + ONNX ----------
print("\n=== Setting up inference ===")
import soundfile as sf
import onnxruntime as ort

SR = 32000
WINDOW = 5 * SR


def read_window(fp, start_sec):
    y, _sr = sf.read(str(fp), start=int(start_sec*SR), frames=WINDOW, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if len(y) < WINDOW: y = np.pad(y, (0, WINDOW - len(y)))
    return y


# torchaudio mel for R1
import torch
import torchaudio
mel_t = torchaudio.transforms.MelSpectrogram(SR, n_fft=2048, hop_length=512,
                                              n_mels=256, f_min=20, f_max=16000, power=2.0)
db_t = torchaudio.transforms.AmplitudeToDB(top_db=80)


def to_mel_ta(wav_np):
    w = torch.from_numpy(wav_np[None].astype(np.float32))
    s = db_t(mel_t(w))
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s.unsqueeze(1).numpy().astype(np.float32)


# librosa mel for Tucker (Slaney)
import librosa


def to_mel_lib(wav_np):
    s = librosa.feature.melspectrogram(y=wav_np, sr=SR, n_fft=2048, hop_length=512,
                                        n_mels=256, fmin=20, fmax=16000, power=2.0)
    s = librosa.power_to_db(s, top_db=80)
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s[None, None].astype(np.float32)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ---------- 7. Run R1 ----------
print("\n=== Running R1 ===")
r1_path = next(r1_dir.glob("student_sed_fold0.onnx"))
r1_sess = ort.InferenceSession(str(r1_path), providers=["CPUExecutionProvider"])
preds_r1 = np.zeros((len(unique_windows), 234), dtype=np.float32)
t0 = time.time()
for i, row in unique_windows.iterrows():
    if row["filename"] not in ogg_paths:
        continue
    try:
        wav = read_window(ogg_paths[row["filename"]], row["start_sec"])
        mel = to_mel_ta(wav)
        out = r1_sess.run(None, {r1_sess.get_inputs()[0].name: mel})
        clip = sigmoid(out[0][0])
        fmax = sigmoid(out[1][0].max(axis=0))
        preds_r1[i] = 0.5*clip + 0.5*fmax
    except Exception as e:
        print(f"  R1 failed {row['filename']} @ {row['start_sec']}: {e}")
    if (i+1) % 100 == 0:
        print(f"  [{i+1}/{len(unique_windows)}] {time.time()-t0:.1f}s")
print(f"R1 done: {time.time()-t0:.1f}s")

# ---------- 8. Run Tucker fold 0 ----------
print("\n=== Running Tucker fold 0 ===")
tucker_path = tucker_dir / "sed_fold0.onnx"
tucker_sess = ort.InferenceSession(str(tucker_path), providers=["CPUExecutionProvider"])
preds_tucker = np.zeros((len(unique_windows), 234), dtype=np.float32)
t0 = time.time()
for i, row in unique_windows.iterrows():
    if row["filename"] not in ogg_paths:
        continue
    try:
        wav = read_window(ogg_paths[row["filename"]], row["start_sec"])
        mel = to_mel_lib(wav)
        out = tucker_sess.run(None, {tucker_sess.get_inputs()[0].name: mel})
        clip = sigmoid(out[0][0])
        fmax = sigmoid(out[1][0].max(axis=0))
        preds_tucker[i] = 0.5*clip + 0.5*fmax
    except Exception as e:
        print(f"  Tucker failed {row['filename']} @ {row['start_sec']}: {e}")
    if (i+1) % 100 == 0:
        print(f"  [{i+1}/{len(unique_windows)}] {time.time()-t0:.1f}s")
print(f"Tucker done: {time.time()-t0:.1f}s")

# ---------- 9. Compute AUCs ----------
print("\n=== AUC comparison ===")
from sklearn.metrics import roc_auc_score


def macro_auc(y_true, y_pred):
    aucs = []
    for c in range(234):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except:
            pass
    return np.array(aucs)


r1_aucs = macro_auc(Y, preds_r1)
tucker_aucs = macro_auc(Y, preds_tucker)
print(f"\nR1     macro-AUC: {r1_aucs.mean():.4f} (median {np.median(r1_aucs):.4f}, min {r1_aucs.min():.4f}, max {r1_aucs.max():.4f})")
print(f"Tucker macro-AUC: {tucker_aucs.mean():.4f} (median {np.median(tucker_aucs):.4f}, min {tucker_aucs.min():.4f}, max {tucker_aucs.max():.4f})")
print(f"\nR1     classes >0.7: {(r1_aucs > 0.7).sum()}/{len(r1_aucs)}")
print(f"Tucker classes >0.7: {(tucker_aucs > 0.7).sum()}/{len(tucker_aucs)}")
print(f"\nR1     classes <0.5: {(r1_aucs < 0.5).sum()}/{len(r1_aucs)}    ← anti-correlated")
print(f"Tucker classes <0.5: {(tucker_aucs < 0.5).sum()}/{len(tucker_aucs)}")

# Per-class diff (R1 vs Tucker)
diff = r1_aucs - tucker_aucs
print(f"\nR1-Tucker per-class diff: mean={diff.mean():+.4f}, median={np.median(diff):+.4f}")
print(f"  R1 better than Tucker: {(diff > 0).sum()}/{len(diff)}")
print(f"  R1 worse than Tucker:  {(diff < 0).sum()}/{len(diff)}")
