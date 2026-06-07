"""Build OV5b: INT8 quantize + measure REAL macro ROC-AUC (FP32 vs INT8).

Fixes OV5's flawed rank-corr metric. Uses train_soundscapes_labels.csv ground truth
to compute the actual BirdCLEF metric (macro AUC, skipping classes with no positives).

This directly answers: "does INT8 hurt LB?"
  - AUC delta < 0.002 → INT8 safe → enables 3-fold exp112
  - AUC delta > 0.005 → INT8 risky → abandon

Eval set: labeled train_soundscapes segments (have ground truth).
Note: models trained on labeled SS (contaminated) → absolute AUC inflated, BUT
      the FP32-vs-INT8 DELTA is what matters (contamination cancels).
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path("experiment/exp112/notebook/nb_ov5b_auc.ipynb")


def cell(src, cid):
    return {
        "cell_type": "code", "id": cid, "metadata": {},
        "execution_count": None, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1_INSTALL = """!pip install -q openvino nncf
import openvino as ov
import nncf
print(f"openvino={ov.__version__}, nncf={nncf.__version__}")
"""

CELL2_IMPORTS = """import os, sys, time, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
import librosa
from sklearn.metrics import roc_auc_score

SR = 32_000
N_MELS = 256; N_FFT = 2048; HOP = 512; FMIN = 20; FMAX = 16000; TOP_DB = 80
N_WINDOWS = 12; WINDOW_SAMPLES = SR * 5; N_TF_DIM = WINDOW_SAMPLES // HOP + 1

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
"""

CELL3_PATHS = """# === Locate competition data + IR dirs (robust) ===
def find_first(cands, marker=None):
    for p in cands:
        p = Path(p)
        if p.exists() and (marker is None or list(p.rglob(marker)) or (p / marker).exists()):
            return p
    return None

print("=== /kaggle/input ===")
for p in sorted(Path("/kaggle/input").iterdir()):
    print(f"  {p.name}/")

COMP = find_first([
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
], marker="taxonomy.csv")
assert COMP is not None, "competition data not found"
print(f"COMP: {COMP}")

SC_DIR = COMP / "train_soundscapes"
LABELS_CSV = COMP / "train_soundscapes_labels.csv"
SAMPLE_SUB = COMP / "sample_submission.csv"
assert SC_DIR.exists() and LABELS_CSV.exists(), f"sc/labels missing"

def find_ir_dir(slugs, marker):
    cands = []
    for s in slugs:
        cands += [f"/kaggle/input/notebooks/maekeso/{s}",
                  f"/kaggle/input/{s}",
                  f"/kaggle/input/datasets/maekeso/{s}"]
    d = find_first(cands, marker=marker)
    if d is None:
        hits = sorted(Path("/kaggle/input").rglob(marker))
        d = hits[0].parent if hits else None
    assert d is not None, f"{marker} not found"
    return d

E106_DIR = find_ir_dir(["birdclef2026-e106-3fold-ov"], "exp106_fold0.xml")
TUCKER_DIR = find_ir_dir(["birdclef2026-tucker-sed-ov"], "sed_fold0.xml")
print(f"E106_DIR: {E106_DIR}")
print(f"TUCKER_DIR: {TUCKER_DIR}")

MODEL_SPECS = []
for f in [0, 1, 2]:
    MODEL_SPECS.append((f"e106_fold{f}", E106_DIR / f"exp106_fold{f}.xml", "torch"))
for f in range(5):
    MODEL_SPECS.append((f"tucker_fold{f}", TUCKER_DIR / f"sed_fold{f}.xml", "librosa"))
"""

CELL4_LABELS = """# === Parse train_soundscapes_labels.csv -> ground truth Y per 5s window ===
sample_sub = pd.read_csv(SAMPLE_SUB)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}
N_CLASSES = len(PRIMARY_LABELS)

labels_df = pd.read_csv(LABELS_CSV)
print(f"labels_df columns: {list(labels_df.columns)}")
print(f"labels_df rows: {len(labels_df)}")
print(labels_df.head(3))

# Determine segment end-time column (start/end). BirdCLEF: filename, start, end (or end_time), primary_label
end_col = None
for c in ["end", "end_time", "end_sec", "seconds"]:
    if c in labels_df.columns:
        end_col = c; break
# Some versions use start only (5s segments)
start_col = None
for c in ["start", "start_time", "start_sec"]:
    if c in labels_df.columns:
        start_col = c; break
print(f"end_col={end_col}, start_col={start_col}")

label_col = "primary_label" if "primary_label" in labels_df.columns else labels_df.columns[-1]
fname_col = "filename" if "filename" in labels_df.columns else labels_df.columns[0]

def parse_time_to_sec(v):
    # handles numeric seconds OR 'HH:MM:SS' / 'MM:SS' time strings
    s = str(v).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    return float(s)

# Build dict: (filename_stem, end_sec) -> set of species
seg_labels = {}
for _, row in labels_df.iterrows():
    fn = str(row[fname_col])
    stem = Path(fn).stem
    if end_col is not None:
        end_sec = int(round(parse_time_to_sec(row[end_col])))
    elif start_col is not None:
        end_sec = int(round(parse_time_to_sec(row[start_col]))) + 5
    else:
        continue
    sp_raw = str(row[label_col])
    species = [s.strip() for s in sp_raw.replace(",", ";").split(";") if s.strip() and s.strip() != "nan"]
    seg_labels.setdefault((stem, end_sec), set()).update(species)

print(f"labeled segments: {len(seg_labels)}")
labeled_files = sorted(set(k[0] for k in seg_labels.keys()))
print(f"labeled files: {len(labeled_files)}")
"""

CELL5_EVAL_DATA = """# === Build eval set: labeled-SS windows with ground truth ===
def load_chunks(path, n=N_WINDOWS):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    target = n * WINDOW_SAMPLES
    if len(wav) < target:
        wav = np.concatenate([wav, np.zeros(target - len(wav), dtype=np.float32)])
    else:
        wav = wav[:target]
    return wav.reshape(n, WINDOW_SAMPLES)

# Match labeled files to actual ogg files
sc_ogg = {p.stem: p for p in SC_DIR.glob("*.ogg")}
eval_files = [f for f in labeled_files if f in sc_ogg][:60]   # cap 60 files
print(f"eval files (labeled & present): {len(eval_files)}")

eval_chunks = []
eval_Y = []
for stem in eval_files:
    chunks = load_chunks(sc_ogg[stem])
    for w in range(N_WINDOWS):
        end_sec = (w + 1) * 5
        species = seg_labels.get((stem, end_sec), set())
        y = np.zeros(N_CLASSES, dtype=np.float32)
        for sp in species:
            if sp in LABEL2IDX:
                y[LABEL2IDX[sp]] = 1.0
        eval_chunks.append(chunks[w])
        eval_Y.append(y)

eval_Y = np.stack(eval_Y)  # (N_win, 234)
print(f"eval windows: {len(eval_chunks)}, positives total: {int(eval_Y.sum())}")
print(f"classes with >=1 positive: {int((eval_Y.sum(0) > 0).sum())} / {N_CLASSES}")

# Calibration set: separate (unlabeled or other SS files) — use first 24 ogg files
all_ogg = sorted(SC_DIR.glob("*.ogg"))
calib_stems = set(eval_files)
calib_files = [p for p in all_ogg if p.stem not in calib_stems][:24]
calib_chunks = []
for p in calib_files:
    calib_chunks.extend(load_chunks(p))
calib_chunks = calib_chunks[:200]
print(f"calib windows: {len(calib_chunks)}")
"""

CELL6_MELS = """# === Pre-compute mels (torch for e106, librosa for Tucker) ===
_mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
    f_min=FMIN, f_max=FMAX, power=2.0)
_db_tf = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

def mel_torch_batch(chunks):
    w = torch.from_numpy(np.stack(chunks).astype(np.float32))  # (N,160000)
    m = _db_tf(_mel_tf(w))  # (N,256,313)
    mean = m.mean(dim=(1,2), keepdim=True); std = m.std(dim=(1,2), keepdim=True) + 1e-6
    m = (m - mean) / std
    return m.unsqueeze(1).numpy().astype(np.float32)  # (N,1,256,313)

def mel_librosa_batch(chunks):
    out = []
    for x in chunks:
        s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        out.append(s)
    return np.stack(out)[:, None].astype(np.float32)

t0 = time.time()
calib_mel_t = mel_torch_batch(calib_chunks)
calib_mel_l = mel_librosa_batch(calib_chunks)
eval_mel_t = mel_torch_batch(eval_chunks)
eval_mel_l = mel_librosa_batch(eval_chunks)
print(f"mels ready in {time.time()-t0:.0f}s: eval_t={eval_mel_t.shape}")
"""

CELL7_QUANT_PREDICT = """# === Quantize + collect FP32/INT8 predictions per model ===
core = ov.Core()
OUT_DIR = Path("/kaggle/working")

def run(compiled, mels):
    probs = []
    for bs in range(0, len(mels), 24):
        be = min(bs+24, len(mels))
        out = compiled(mels[bs:be])
        clip = out[compiled.outputs[0]]
        frame = out[compiled.outputs[1]]
        fmax = frame.max(axis=1)
        probs.append((0.5*sigmoid(clip) + 0.5*sigmoid(fmax)).astype(np.float32))
    return np.concatenate(probs)

fp32_preds = {}; int8_preds = {}
for name, xml, mt in MODEL_SPECS:
    print(f"\\n=== {name} ({mt}) ===")
    t0 = time.time()
    eval_mels = eval_mel_t if mt == "torch" else eval_mel_l
    calib_mels = calib_mel_t if mt == "torch" else calib_mel_l

    fp32_model = core.read_model(str(xml))
    comp_fp32 = core.compile_model(fp32_model, "CPU")
    fp32_preds[name] = run(comp_fp32, eval_mels)

    calib_ds = nncf.Dataset(list(calib_mels), lambda s: s[None])
    int8_model = nncf.quantize(fp32_model, calib_ds,
                               subset_size=min(200, len(calib_mels)),
                               preset=nncf.QuantizationPreset.PERFORMANCE)
    ov.save_model(int8_model, str(OUT_DIR / f"{xml.stem}_int8.xml"))
    comp_int8 = core.compile_model(int8_model, "CPU")
    int8_preds[name] = run(comp_int8, eval_mels)

    print(f"  done {time.time()-t0:.0f}s, fp32={fp32_preds[name].shape}")
    del fp32_model, int8_model, comp_fp32, comp_int8
    import gc; gc.collect()
"""

CELL8_AUC = """# === Compute REAL macro ROC-AUC (FP32 vs INT8) ===
def macro_auc(y_true, y_score):
    # BirdCLEF metric: AUC per class with >=1 positive, averaged
    aucs = []
    for c in range(y_true.shape[1]):
        pos = y_true[:, c].sum()
        if pos > 0 and pos < len(y_true):   # need both pos and neg
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)

def ens(names, d):
    return np.mean([d[n] for n in names], axis=0)

print(f"\\n{'='*64}\\n  REAL macro ROC-AUC (eval = labeled train_soundscapes)\\n{'='*64}")
print(f"{'model/ensemble':>26} {'FP32_AUC':>10} {'INT8_AUC':>10} {'delta':>9} {'n_cls':>6}")

def report(label, fp, i8):
    a_fp, n = macro_auc(eval_Y, fp)
    a_i8, _ = macro_auc(eval_Y, i8)
    print(f"{label:>26} {a_fp:>10.5f} {a_i8:>10.5f} {a_i8-a_fp:>+9.5f} {n:>6}")
    return a_fp, a_i8

# per-model
for name, _, _ in MODEL_SPECS:
    report(name, fp32_preds[name], int8_preds[name])

# ensembles
print("  " + "-"*60)
e106_names = [f"e106_fold{f}" for f in [0,1,2]]
tuck_names = [f"tucker_fold{f}" for f in range(5)]
a_e106_fp, a_e106_i8 = report("e106 3-fold ENS", ens(e106_names, fp32_preds), ens(e106_names, int8_preds))
a_tuck_fp, a_tuck_i8 = report("Tucker 5-fold ENS", ens(tuck_names, fp32_preds), ens(tuck_names, int8_preds))

# Combined stack proxy (e106 3-fold + Tucker 5-fold, simple avg)
comb_fp = 0.5*ens(e106_names, fp32_preds) + 0.5*ens(tuck_names, fp32_preds)
comb_i8 = 0.5*ens(e106_names, int8_preds) + 0.5*ens(tuck_names, int8_preds)
a_comb_fp, a_comb_i8 = report("e106+Tucker combined", comb_fp, comb_i8)

print(f"\\n{'='*64}\\n  VERDICT (AUC delta = INT8 - FP32, on this eval set)\\n{'='*64}")
print(f"  e106 3-fold:     {a_e106_i8 - a_e106_fp:+.5f}")
print(f"  Tucker 5-fold:   {a_tuck_i8 - a_tuck_fp:+.5f}")
print(f"  combined:        {a_comb_i8 - a_comb_fp:+.5f}")
d = a_comb_i8 - a_comb_fp
verdict = ("SAFE (delta > -0.002)" if d > -0.002 else
           "ACCEPTABLE (-0.005..-0.002)" if d > -0.005 else
           "RISKY (delta < -0.005)")
print(f"  >>> INT8 verdict: {verdict}")
print(f"      NOTE: eval is contaminated (models saw labeled SS), so DELTA matters, not absolute AUC")
"""

CELL9_LIST = """out_dir = Path("/kaggle/working")
print(f"\\nINT8 IR files:")
total = 0
for f in sorted(out_dir.glob("*_int8.*")):
    sz = f.stat().st_size; total += sz
    print(f"  {f.name:35s} {sz/1e6:8.2f}MB")
print(f"  TOTAL {total/1e6:.1f}MB")
"""

CELLS = [
    (CELL1_INSTALL, "cell-install"),
    (CELL2_IMPORTS, "cell-imports"),
    (CELL3_PATHS, "cell-paths"),
    (CELL4_LABELS, "cell-labels"),
    (CELL5_EVAL_DATA, "cell-evaldata"),
    (CELL6_MELS, "cell-mels"),
    (CELL7_QUANT_PREDICT, "cell-quant"),
    (CELL8_AUC, "cell-auc"),
    (CELL9_LIST, "cell-list"),
]

nb = {
    "cells": [cell(s, cid) for s, cid in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes)")

for src, cid in CELLS:
    if cid == "cell-install":
        continue
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<{cid}>", "exec")
        print(f"  [OK] {cid}")
    except SyntaxError as e:
        print(f"  [FAIL] {cid}: {e}")
