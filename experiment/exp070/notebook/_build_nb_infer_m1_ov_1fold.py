"""Build M1 best 1-fold OpenVINO inference NB (Kaggle CPU sub).

Strategy:
  - Best fold (val_ns22 max) を auto-detect、1 fold only inference
  - OpenVINO runtime (existing .onnx を直接 load、IR 変換不要)
  - ONNX runtime fallback (OpenVINO install fail 時)
  - Expected runtime: 15-25min、90min limit 余裕

Required Kaggle dataset_sources:
  - maekeso/birdclef2026-exp070-m1-onnx (M1 ONNX 5-fold)
  - romantamrazov/onnxruntime-1-24-4 (ONNX runtime wheel)
  - OpenVINO wheel dataset (要 attach、user 側で choose)
"""
import json
from pathlib import Path

NB_OUT = Path(__file__).with_name("nb_infer_m1_ov_1fold.ipynb")

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True) if src else []})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
                  "outputs": [], "execution_count": None})

# =============================================================
md("""# exp070 M1 best 1-fold OpenVINO inference (Kaggle CPU sub)

**Strategy**: 安全運用 = 1-fold only + OpenVINO 最速
- Best fold (val_ns22 max) を auto-detect
- OpenVINO runtime (existing .onnx を直接 load)
- ONNX runtime fallback あり (OpenVINO 不在時)

**Expected runtime**: 15-25min (90min limit に 65-75min margin) ★ 確実通過

**Required dataset_sources**:
- `maekeso/birdclef2026-exp070-m1-onnx` (M1 ONNX 5-fold)
- `romantamrazov/onnxruntime-1-24-4` (ONNX wheel、fallback 用)
- OpenVINO wheel dataset (要 attach、user choose)
""")

# =============================================================
code("""# Cell 1: Install OpenVINO + ONNX runtime (offline)
import subprocess, sys, os, importlib
from pathlib import Path
import importlib.util

print(f"Python: {sys.version[:50]}")
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"Python tag: {py_tag}")

# 1. Try install OpenVINO (priority)
USE_OPENVINO = False
if importlib.util.find_spec("openvino") is None:
    # Search for openvino wheel dataset
    OV_WHEEL_DIRS = sorted(Path("/kaggle/input").glob("*openvino*")) + sorted(Path("/kaggle/input").glob("*intel-openvino*"))
    ov_wheel_dir = OV_WHEEL_DIRS[0] if OV_WHEEL_DIRS else None
    if ov_wheel_dir and ov_wheel_dir.exists():
        print(f"OpenVINO wheel dir: {ov_wheel_dir}")
        whls = list(ov_wheel_dir.glob("**/openvino*.whl"))
        print(f"  Wheels: {[w.name for w in whls]}")
        matching = [w for w in whls if py_tag in w.name]
        target = matching[0] if matching else (whls[0] if whls else None)
        if target:
            print(f"  Installing {target.name}...")
            r = subprocess.run([sys.executable, "-m", "pip", "install",
                                "--no-index", "--no-deps",
                                "--find-links", str(ov_wheel_dir), "openvino"],
                               capture_output=True, text=True)
            print(f"    returncode={r.returncode}")
            if r.stderr: print(f"    stderr: {r.stderr[-300:]}")
    else:
        print("[INFO] No openvino wheel dataset attached, will fallback to ONNX runtime")

try:
    import openvino as ov
    USE_OPENVINO = True
    print(f"\\nOK OpenVINO version: {ov.__version__}")
except ImportError:
    print("\\n[Fallback] OpenVINO not available, will use ONNX runtime")

# 2. ONNX runtime (fallback)
if importlib.util.find_spec("onnxruntime") is None:
    WHEEL_CANDIDATES = [
        Path("/kaggle/input/datasets/romantamrazov/onnxruntime-1-24-4"),
        Path("/kaggle/input/onnxruntime-1-24-4"),
    ]
    wheel_dir = next((p for p in WHEEL_CANDIDATES if p.exists()), None)
    if wheel_dir is not None:
        print(f"\\nInstalling ONNX runtime from {wheel_dir}...")
        r = subprocess.run([sys.executable, "-m", "pip", "install",
                            "--no-index", "--no-deps",
                            "--find-links", str(wheel_dir), "onnxruntime"],
                           capture_output=True, text=True)
        print(f"  returncode={r.returncode}")

import onnxruntime as ort
print(f"OK ONNX runtime version: {ort.__version__}")

print(f"\\n=== Runtime priority: {'OpenVINO' if USE_OPENVINO else 'ONNX runtime'} ===")
""")

# =============================================================
code("""# Cell 2: Imports + Setup
import time, gc, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import soundfile as sf

torch.set_num_threads(4)
print(f"torch: {torch.__version__}, torchaudio: {torchaudio.__version__}")
START = time.time()
""")

# =============================================================
code("""# Cell 3: Config (M1 spec)
NUM_CLASSES = 234
SR = 32000
CHUNK_SEC = 5
CHUNK_SAMPLES = SR * CHUNK_SEC
N_WINDOWS = 12
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000

BATCH_SIZE = 32

# Paths
DATA_PATHS = ["/kaggle/input/competitions/birdclef-2026",
              "/kaggle/input/birdclef-2026"]
DATA_PATH = next((Path(p) for p in DATA_PATHS if Path(p).exists()), None)
assert DATA_PATH is not None

TEST_DIR = DATA_PATH / "test_soundscapes"
SAMPLE_SUB_PATH = DATA_PATH / "sample_submission.csv"

# M1 ONNX dataset
M1_ONNX_DIR = None
for _p in ["/kaggle/input/birdclef2026-exp070-m1-onnx",
           "/kaggle/input/datasets/maekeso/birdclef2026-exp070-m1-onnx"]:
    if Path(_p).exists():
        M1_ONNX_DIR = Path(_p); break
assert M1_ONNX_DIR is not None, "M1 ONNX not attached"
print(f"M1 ONNX dir: {M1_ONNX_DIR}")

onnx_paths = sorted(M1_ONNX_DIR.glob("*.onnx"))
assert len(onnx_paths) > 0, "No ONNX files found"
print(f"Found {len(onnx_paths)} ONNX files:")
for f in onnx_paths:
    print(f"  {f.name} ({f.stat().st_size/1e6:.1f}MB)")
""")

# =============================================================
code("""# Cell 4: Load BC26 labels
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES
print(f"BC26 labels: {len(PRIMARY_LABELS)}")
""")

# =============================================================
code("""# Cell 5: Pick BEST fold (val_ns22 max)
# Try to load .pth ckpts to get val_ns22 (if attached as same dataset)
# Otherwise default to fold 0 (or specified by BEST_FOLD)
BEST_FOLD = 0   # ★ Override here if needed

# Look for .pth ckpts in same dir (val_ns22 metadata 取得用)
pth_paths = sorted(M1_ONNX_DIR.glob("*.pth"))
if pth_paths:
    print(f"Found {len(pth_paths)} .pth ckpts, scanning val_ns22...")
    best_score = -1
    for p in pth_paths:
        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            score = ckpt.get("best_ns22", ckpt.get("val_ns22", -1))
            # Extract fold idx from filename (e.g., m1_fold3_ckpt_best.pth → 3)
            import re
            m = re.search(r"fold(\\d+)", p.name)
            if m:
                fi = int(m.group(1))
                print(f"  fold {fi}: val_ns22={score:.4f}")
                if score > best_score:
                    best_score = score
                    BEST_FOLD = fi
        except Exception as e:
            print(f"  {p.name}: failed to load ({e})")
    print(f"\\nBest fold: {BEST_FOLD} (val_ns22={best_score:.4f})")
else:
    print(f"No .pth ckpts found, using BEST_FOLD={BEST_FOLD} (default)")

# Pick the corresponding .onnx
import re
best_onnx = None
for op in onnx_paths:
    m = re.search(r"fold(\\d+)", op.name)
    if m and int(m.group(1)) == BEST_FOLD:
        best_onnx = op; break
if best_onnx is None:
    best_onnx = onnx_paths[BEST_FOLD]   # fallback by index
print(f"\\nSelected ONNX: {best_onnx.name}")
""")

# =============================================================
code("""# Cell 6: Test file enumeration + empty handling
test_files = sorted(TEST_DIR.glob("*.ogg"))
print(f"Test soundscape files: {len(test_files)}")

HAS_TEST_FILES = len(test_files) > 0
if not HAS_TEST_FILES:
    print("[INFO] No test files (local commit run) — placeholder submission")
    placeholder = sample_sub.copy()
    for col in PRIMARY_LABELS:
        placeholder[col] = 0.5
    placeholder.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"  Placeholder saved")
""")

# =============================================================
code("""# Cell 7: Mel transform + helpers
class MelSpecTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


def load_audio_60s(path):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return np.zeros(SR * 60, dtype=np.float32)


def split_into_chunks(wav, chunk_samples=CHUNK_SAMPLES, n_chunks=N_WINDOWS):
    target = chunk_samples * n_chunks
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target:
        wav = wav[:target]
    return np.stack([wav[i*chunk_samples:(i+1)*chunk_samples] for i in range(n_chunks)])

print("OK helpers")
""")

# =============================================================
code("""# Cell 8: Pre-compute mel for all chunks
if HAS_TEST_FILES:
    t0 = time.time()
    print(f"Loading {len(test_files)} test files + computing mel...")

    mel_transform = MelSpecTransform()
    mel_transform.eval()

    all_mels = []
    file_ids = []

    with torch.no_grad():
        for i, f in enumerate(test_files):
            wav = load_audio_60s(f)
            chunks = split_into_chunks(wav)
            chunks_t = torch.from_numpy(chunks).unsqueeze(1)
            mel = mel_transform(chunks_t)
            for c in range(mel.shape[0]):
                mel[c] = (mel[c] - mel[c].mean()) / (mel[c].std() + 1e-6)
            all_mels.append(mel.numpy().astype(np.float32))
            fname = f.stem
            for c in range(N_WINDOWS):
                end_sec = (c + 1) * 5
                file_ids.append(f"{fname}_{end_sec}")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_files)} ({(time.time()-t0)/60:.1f}min)")

    all_mels = np.concatenate(all_mels, axis=0)
    print(f"All mels shape: {all_mels.shape}")
    print(f"Mel pre-compute time: {(time.time()-t0)/60:.1f}min")
else:
    all_mels = None
    file_ids = []
""")

# =============================================================
code("""# Cell 9: Single best fold inference (OpenVINO or ONNX runtime)
if HAS_TEST_FILES:
    t0_infer = time.time()
    n_chunks = len(all_mels)
    print(f"\\nInferring on {n_chunks} chunks with best fold ({best_onnx.name})...")

    if USE_OPENVINO:
        # OpenVINO runtime
        import openvino as ov
        core = ov.Core()
        compiled = core.compile_model(model=str(best_onnx), device_name="CPU")
        print(f"  Runtime: OpenVINO (compiled)")
        # Find output keys
        out_keys = {o.any_name for o in compiled.outputs}
        print(f"  Output keys: {out_keys}")
        clip_key = next((k for k in out_keys if "clip" in k.lower()), None)
        frame_key = next((k for k in out_keys if "frame" in k.lower()), None)

        preds = np.zeros((n_chunks, NUM_CLASSES), dtype=np.float32)
        for s in range(0, n_chunks, BATCH_SIZE):
            batch = all_mels[s:s+BATCH_SIZE]
            output = compiled({"mel": batch})
            clip_logit = output[clip_key] if clip_key else list(output.values())[0]
            framewise = output[frame_key] if frame_key else list(output.values())[1]
            frame_max = framewise.max(axis=1)
            p_clip = 1.0 / (1.0 + np.exp(-clip_logit))
            p_fmax = 1.0 / (1.0 + np.exp(-frame_max))
            preds[s:s+len(p_clip)] = 0.5 * p_clip + 0.5 * p_fmax
    else:
        # ONNX runtime fallback
        sess = ort.InferenceSession(str(best_onnx), providers=["CPUExecutionProvider"])
        print(f"  Runtime: ONNX runtime")

        preds = np.zeros((n_chunks, NUM_CLASSES), dtype=np.float32)
        for s in range(0, n_chunks, BATCH_SIZE):
            batch = all_mels[s:s+BATCH_SIZE]
            outputs = sess.run(["clip_logit", "framewise"], {"mel": batch})
            clip_logit = outputs[0]
            framewise = outputs[1]
            frame_max = framewise.max(axis=1)
            p_clip = 1.0 / (1.0 + np.exp(-clip_logit))
            p_fmax = 1.0 / (1.0 + np.exp(-frame_max))
            preds[s:s+len(p_clip)] = 0.5 * p_clip + 0.5 * p_fmax

    print(f"\\nInference done in {(time.time()-t0_infer)/60:.1f}min")
    print(f"Preds shape: {preds.shape}, stats: mean={preds.mean():.4f}, max={preds.max():.4f}")
else:
    preds = None
""")

# =============================================================
code("""# Cell 10: Build submission + save
if HAS_TEST_FILES:
    sub_df = pd.DataFrame(preds, columns=PRIMARY_LABELS)
    sub_df.insert(0, "row_id", file_ids)

    expected_rows = len(sample_sub)
    if len(sub_df) != expected_rows:
        print(f"[WARN] Row mismatch {len(sub_df)} vs {expected_rows}, aligning...")
        sub_map = sub_df.set_index("row_id")
        aligned = sample_sub[["row_id"]].copy()
        for lbl in PRIMARY_LABELS:
            if lbl in sub_map.columns:
                aligned[lbl] = aligned["row_id"].map(sub_map[lbl]).fillna(0.0)
            else:
                aligned[lbl] = 0.0
        sub_df = aligned

    out_path = Path("/kaggle/working/submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"\\nOK Submission saved: {out_path} ({out_path.stat().st_size/1e6:.1f}MB)")
else:
    print("[INFO] Placeholder already written")

print(f"\\nTotal time: {(time.time()-START)/60:.1f}min")
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_OUT}")
for i, c in enumerate(CELLS):
    src = "".join(c.get("source", []))
    head = src.split('\n')[0][:70] if src else "(empty)"
    print(f"  Cell {i:2d} ({c['cell_type']:8s}) {len(src):6d} chars | {head}")
