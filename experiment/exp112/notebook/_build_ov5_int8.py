"""Build OV5 NB: INT8 quantize e106 (3-fold) + Tucker (5-fold) with DETAILED error logging.

Goal: measure FP32 vs INT8 error BEFORE committing to INT8 exp112.

Outputs:
  - INT8 IR: exp106_fold{0,1,2}_int8.xml/.bin, sed_fold{0..4}_int8.xml/.bin
  - Error report log:
      per-model: sigmoid-space max/mean/p99 diff
      ensemble:  Spearman rank correlation (FP32 vs INT8) — directly proxies ROC-AUC preservation
      verdict line

Why rank correlation: competition metric is macro ROC-AUC (rank-based). If per-class
rank order is preserved (corr ~1.0), AUC is unchanged regardless of absolute prob shift.

Input datasets:
  - maekeso/birdclef2026-e106-3fold-ov  (FP32 e106 IR)
  - maekeso/birdclef2026-tucker-sed-ov  (FP32 Tucker IR)
  - ttahara/birdclef-2026-download-wheels (openvino wheels — for consistency)
  - competition birdclef-2026 (train_soundscapes for calibration/eval audio)
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path("experiment/exp112/notebook/nb_ov5_int8.ipynb")


def cell(src, cid):
    return {
        "cell_type": "code", "id": cid, "metadata": {},
        "execution_count": None, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1_INSTALL = """# Install openvino + nncf (internet=True). nncf for post-training INT8 quantization.
!pip install -q openvino nncf
import openvino as ov
import nncf
print(f"openvino={ov.__version__}, nncf={nncf.__version__}")
"""

CELL2_IMPORTS = """import os, sys, time, glob, re
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from scipy.stats import spearmanr

SR = 32_000
N_MELS = 256; N_FFT = 2048; HOP = 512; FMIN = 20; FMAX = 16000; TOP_DB = 80
N_WINDOWS = 12
WINDOW_SAMPLES = SR * 5
N_TF_DIM = WINDOW_SAMPLES // HOP + 1  # 313

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
"""

CELL3_LOCATE = """# === Locate FP32 IR (e106 + Tucker) ===
def find_dir(slugs, marker):
    cands = []
    for s in slugs:
        cands += [Path(f"/kaggle/input/notebooks/maekeso/{s}"),
                  Path(f"/kaggle/input/{s}"),
                  Path(f"/kaggle/input/datasets/maekeso/{s}")]
    for p in cands:
        if p.exists() and list(p.rglob(marker)):
            return p
    hits = sorted(Path("/kaggle/input").rglob(marker))
    assert hits, f"{marker} not found"
    return hits[0].parent

E106_DIR = find_dir(["birdclef2026-e106-3fold-ov"], "exp106_fold0.xml")
TUCKER_DIR = find_dir(["birdclef2026-tucker-sed-ov"], "sed_fold0.xml")
print(f"E106_DIR:   {E106_DIR}")
print(f"TUCKER_DIR: {TUCKER_DIR}")

# Model spec: (name, fp32_xml, mel_type)
MODEL_SPECS = []
for f in [0, 1, 2]:
    MODEL_SPECS.append((f"e106_fold{f}", E106_DIR / f"exp106_fold{f}.xml", "torch"))
for f in [0, 1, 2, 3, 4]:
    MODEL_SPECS.append((f"tucker_fold{f}", TUCKER_DIR / f"sed_fold{f}.xml", "librosa"))
for name, xml, mt in MODEL_SPECS:
    assert xml.exists(), f"missing {xml}"
    print(f"  {name}: {xml.name} ({mt} mel)")
"""

CELL4_MEL = """# === Mel generators (torch for e106, librosa for Tucker) ===
_mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
    f_min=FMIN, f_max=FMAX, power=2.0)
_db_tf = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

def mel_torch(chunk):
    # chunk: (160000,) -> (1,1,256,313)
    w = torch.from_numpy(chunk[None].astype(np.float32))
    m = _db_tf(_mel_tf(w))  # (1,256,313)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m.unsqueeze(0).numpy().astype(np.float32)

def mel_librosa(chunk):
    s = librosa.feature.melspectrogram(y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=TOP_DB)
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s[None, None].astype(np.float32)

def load_chunks(path, max_windows=N_WINDOWS):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    target = max_windows * WINDOW_SAMPLES
    if len(wav) < target:
        wav = np.concatenate([wav, np.zeros(target - len(wav), dtype=np.float32)])
    else:
        wav = wav[:target]
    return wav.reshape(max_windows, WINDOW_SAMPLES)
"""

CELL5_DATA = """# === Build calibration + eval chunk sets from train_soundscapes ===
# Robust path detection (competition mount varies: /kaggle/input/birdclef-2026 OR /competitions/)
print("=== /kaggle/input/ contents ===")
for p in sorted(Path("/kaggle/input").iterdir()):
    print(f"  {p.name}/")

SC_DIR = None
for cand in [
    Path("/kaggle/input/birdclef-2026/train_soundscapes"),
    Path("/kaggle/input/competitions/birdclef-2026/train_soundscapes"),
]:
    if cand.exists() and list(cand.glob("*.ogg")):
        SC_DIR = cand; break
if SC_DIR is None:
    for h in sorted(Path("/kaggle/input").rglob("train_soundscapes")):
        if list(h.glob("*.ogg")):
            SC_DIR = h; break
# Fallback: use train_audio (.ogg in subdirs) if train_soundscapes unavailable
if SC_DIR is None:
    for ta_name in ["train_audio"]:
        for base in [Path("/kaggle/input/birdclef-2026"),
                     Path("/kaggle/input/competitions/birdclef-2026")]:
            ta = base / ta_name
            if ta.exists():
                SC_DIR = ta; break
        if SC_DIR is not None: break
assert SC_DIR is not None, "no audio source found (train_soundscapes / train_audio)"
print(f"audio source: {SC_DIR}")

sc_files = sorted(SC_DIR.rglob("*.ogg"))[:60]   # rglob covers train_audio subdirs too
print(f"audio files (capped 60): {len(sc_files)}")
assert len(sc_files) >= 40, f"need >= 40 audio files, got {len(sc_files)}"

# calibration: 24 files -> 288 chunks (use up to 200)
# eval:        16 files -> 192 chunks (separate set)
calib_files = sc_files[:24]
eval_files  = sc_files[24:40]

t0 = time.time()
calib_chunks = []
for fp in calib_files:
    calib_chunks.extend(load_chunks(fp))
eval_chunks = []
for fp in eval_files:
    eval_chunks.extend(load_chunks(fp))
calib_chunks = calib_chunks[:200]
eval_chunks = eval_chunks[:192]
print(f"calib chunks: {len(calib_chunks)}, eval chunks: {len(eval_chunks)} ({time.time()-t0:.0f}s)")

# Pre-compute mels for both mel types
calib_mel_torch = np.stack([mel_torch(c) for c in calib_chunks])[:, 0]   # (N,1,256,313)
calib_mel_libr  = np.stack([mel_librosa(c) for c in calib_chunks])[:, 0]
eval_mel_torch  = np.stack([mel_torch(c) for c in eval_chunks])[:, 0]
eval_mel_libr   = np.stack([mel_librosa(c) for c in eval_chunks])[:, 0]
print(f"mels ready: calib_torch={calib_mel_torch.shape}, eval_torch={eval_mel_torch.shape}")
"""

CELL6_QUANTIZE = """# === Quantize each model + measure FP32 vs INT8 error ===
core = ov.Core()
OUT_DIR = Path("/kaggle/working")

def run_model(compiled, mels):
    # mels: (N,1,256,313). Return final prob (N,234) = 0.5*sig(clip)+0.5*sig(frame_max)
    out = compiled(mels)
    clip = out[compiled.outputs[0]]              # (N,234)
    frame = out[compiled.outputs[1]]             # (N,T,234)
    frame_max = frame.max(axis=1)                # (N,234)
    return (0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)).astype(np.float32)

results = {}    # name -> dict of metrics
fp32_probs = {} # name -> eval probs (FP32)
int8_probs = {} # name -> eval probs (INT8)

for name, xml, mel_type in MODEL_SPECS:
    print(f"\\n{'='*60}\\n  {name}  ({mel_type} mel)\\n{'='*60}")
    t0 = time.time()
    calib_mels = calib_mel_torch if mel_type == "torch" else calib_mel_libr
    eval_mels  = eval_mel_torch  if mel_type == "torch" else eval_mel_libr

    fp32_model = core.read_model(str(xml))

    # --- NNCF post-training INT8 quantization ---
    def transform_fn(sample):
        return sample[None]  # (1,1,256,313)
    calib_ds = nncf.Dataset(list(calib_mels), transform_fn)
    int8_model = nncf.quantize(
        fp32_model, calib_ds,
        subset_size=min(200, len(calib_mels)),
        preset=nncf.QuantizationPreset.PERFORMANCE,
    )
    int8_xml = OUT_DIR / f"{xml.stem}_int8.xml"
    ov.save_model(int8_model, str(int8_xml))
    print(f"  INT8 IR saved: {int8_xml.name} (.bin {Path(str(int8_xml).replace('.xml','.bin')).stat().st_size/1e6:.1f}MB)")

    # --- compile + run both on eval ---
    comp_fp32 = core.compile_model(fp32_model, "CPU")
    comp_int8 = core.compile_model(int8_model, "CPU")

    # batched eval (avoid OOM: process in mini-batches of 24)
    p_fp32 = []; p_int8 = []
    for bs in range(0, len(eval_mels), 24):
        be = min(bs + 24, len(eval_mels))
        p_fp32.append(run_model(comp_fp32, eval_mels[bs:be]))
        p_int8.append(run_model(comp_int8, eval_mels[bs:be]))
    p_fp32 = np.concatenate(p_fp32); p_int8 = np.concatenate(p_int8)
    fp32_probs[name] = p_fp32; int8_probs[name] = p_int8

    # --- numerical diff (sigmoid space) ---
    d = np.abs(p_fp32 - p_int8)
    max_d = float(d.max()); mean_d = float(d.mean()); p99_d = float(np.percentile(d, 99))
    # --- per-class rank correlation (ROC-AUC proxy) ---
    corrs = []
    for c in range(p_fp32.shape[1]):
        if np.ptp(p_fp32[:, c]) > 1e-9 and np.ptp(p_int8[:, c]) > 1e-9:
            rc, _ = spearmanr(p_fp32[:, c], p_int8[:, c])
            if not np.isnan(rc):
                corrs.append(rc)
    mean_corr = float(np.mean(corrs)) if corrs else float("nan")
    p5_corr = float(np.percentile(corrs, 5)) if corrs else float("nan")

    results[name] = dict(max_diff=max_d, mean_diff=mean_d, p99_diff=p99_d,
                         rank_corr=mean_corr, rank_corr_p5=p5_corr,
                         n_classes_eval=len(corrs), elapsed=time.time()-t0)
    print(f"  sigmoid diff: max={max_d:.5f} mean={mean_d:.6f} p99={p99_d:.5f}")
    print(f"  rank corr: mean={mean_corr:.5f} p5={p5_corr:.5f} (n_cls={len(corrs)})")
    print(f"  elapsed {time.time()-t0:.0f}s")

    del fp32_model, int8_model, comp_fp32, comp_int8
    import gc; gc.collect()
"""

CELL7_ENSEMBLE = """# === Ensemble-level error (the metric that matters for LB) ===
print(f"\\n{'='*60}\\n  ENSEMBLE error report\\n{'='*60}")

def ensemble_probs(names, probs_dict):
    return np.mean([probs_dict[n] for n in names], axis=0)

# e106 3-fold ensemble
e106_names = [f"e106_fold{f}" for f in [0, 1, 2]]
e106_fp32 = ensemble_probs(e106_names, fp32_probs)
e106_int8 = ensemble_probs(e106_names, int8_probs)
# Tucker 5-fold ensemble
tuck_names = [f"tucker_fold{f}" for f in range(5)]
tuck_fp32 = ensemble_probs(tuck_names, fp32_probs)
tuck_int8 = ensemble_probs(tuck_names, int8_probs)

def report(label, fp, i8):
    d = np.abs(fp - i8)
    corrs = []
    for c in range(fp.shape[1]):
        if np.ptp(fp[:, c]) > 1e-9 and np.ptp(i8[:, c]) > 1e-9:
            rc, _ = spearmanr(fp[:, c], i8[:, c])
            if not np.isnan(rc): corrs.append(rc)
    print(f"\\n[{label}]")
    print(f"  sigmoid diff: max={d.max():.5f} mean={d.mean():.6f} p99={np.percentile(d,99):.5f}")
    print(f"  rank corr:    mean={np.mean(corrs):.5f} p5={np.percentile(corrs,5):.5f} min={np.min(corrs):.4f}")
    return float(np.mean(corrs))

c_e106 = report("e106 3-fold ensemble (FP32 vs INT8)", e106_fp32, e106_int8)
c_tuck = report("Tucker 5-fold ensemble (FP32 vs INT8)", tuck_fp32, tuck_int8)

print(f"\\n{'='*60}\\n  VERDICT\\n{'='*60}")
print(f"  e106  ensemble rank corr: {c_e106:.5f}")
print(f"  Tucker ensemble rank corr: {c_tuck:.5f}")
thresh = 0.999
verdict = "SAFE (AUC preserved)" if (c_e106 >= thresh and c_tuck >= thresh) else \\
          ("ACCEPTABLE (~-0.001 AUC)" if (c_e106 >= 0.995 and c_tuck >= 0.995) else "RISKY (>-0.002 AUC possible)")
print(f"  >>> INT8 verdict: {verdict}")
print(f"      (rank corr >= 0.999 = AUC unchanged; 0.995-0.999 = ~-0.001; <0.995 = risky)")
"""

CELL8_SUMMARY = """# === Per-model summary table + output files ===
print(f"\\n{'model':>14} {'max_diff':>10} {'mean_diff':>10} {'rank_corr':>10} {'corr_p5':>9} {'s':>5}")
for name, r in results.items():
    print(f"{name:>14} {r['max_diff']:>10.5f} {r['mean_diff']:>10.6f} "
          f"{r['rank_corr']:>10.5f} {r['rank_corr_p5']:>9.5f} {r['elapsed']:>5.0f}")

out_dir = Path("/kaggle/working")
print(f"\\nINT8 IR files in {out_dir}:")
total = 0
for f in sorted(out_dir.glob("*_int8.*")):
    sz = f.stat().st_size; total += sz
    print(f"  {f.name:35s} {sz/1e6:8.2f}MB")
print(f"  {'TOTAL':35s} {total/1e6:8.2f}MB")
"""

CELLS = [
    (CELL1_INSTALL, "cell-install"),
    (CELL2_IMPORTS, "cell-imports"),
    (CELL3_LOCATE, "cell-locate"),
    (CELL4_MEL, "cell-mel"),
    (CELL5_DATA, "cell-data"),
    (CELL6_QUANTIZE, "cell-quantize"),
    (CELL7_ENSEMBLE, "cell-ensemble"),
    (CELL8_SUMMARY, "cell-summary"),
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
