"""Generate exp095 R3 (l0 + R2 spec) single fold inference NB.

Cloned from experiment/exp029/notebook/_gen_nb_infer_r3_single.py with:
  - BACKBONE: eca_nfnet_l1 → eca_nfnet_l0
  - Dataset: birdclef2026-exp029-l1-single → birdclef2026-exp095-l0-r2spec
  - Header / comments updated for exp095 R3 (val 0.9187 @ ep8)

Run: python _gen_nb_infer_r3.py  ->  writes nb_infer_r3.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}


cells = []

cells.append(md_cell(r"""# exp095 R3 single fold inference (l0 + R2 exact spec, Kaggle CPU 90min, internet=off)

**Single fold** inference using R3 ckpt (eca_nfnet_l0、exp017 R2 teacher distill、R2 exact spec).
1 model forward per chunk, Gaussian smoothed + sigmoid.

## exp095 R3 fold 0 val 実績
- ep 8 best:  val_ns22 = 0.9187 ★ (best ckpt)
- ep 7:       val_ns22 = 0.9173
- ep 11:      val_ns22 = 0.9183
- ep 17:      val_ns22 = 0.9178
- ep 20 final: 0.9168

## Comparison
- exp017 R2 (l0 single fold): val 0.9249, LB 0.921 (gap -0.004)
- exp029 R3 (l1 single fold): val 0.9358, LB 0.923 (gap -0.014)
- **exp095 R3 (l0 + R2 spec)**: val 0.9187 → 予測 LB **0.914-0.918** (l0 gap -0.004 適用)

## Inputs
- `birdclef-2026` (competition data)
- `maekeso/birdclef2026-exp095-l0-r2spec` (R3 ckpt: `r3_fold0_ckpt_best_ns22.pth`)
- `romantamrazov/onnxruntime-1-24-4` (onnxruntime wheel for offline install)

## Time budget
- ONNX CPU 1 fold: ~10-20 min (5-fold より遥かに余裕)
- 90 min 制約には大幅余裕
""", "hdr"))

cells.append(code_cell(r"""# ============================================================
# Cell 1: Setup
# ============================================================
import os, sys, time, json, math, glob, re, gc
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

# ONNX runtime (offline install)
import subprocess
WHEEL_CANDIDATES = [
    Path("/kaggle/input/datasets/romantamrazov/onnxruntime-1-24-4"),
    Path("/kaggle/input/onnxruntime-1-24-4"),
]
wheel_dir = next((p for p in WHEEL_CANDIDATES if p.exists()), None)
assert wheel_dir is not None, (
    "onnxruntime wheel dir not found. "
    "Attach romantamrazov/onnxruntime-1-24-4 as dataset_sources"
)
print(f"Wheel dir: {wheel_dir}")
print(f"  contents: {[f.name for f in wheel_dir.iterdir()]}")

try:
    import onnxruntime as ort
    print(f"onnxruntime (pre-installed): {ort.__version__}")
except ImportError:
    print("Installing onnxruntime offline (--no-deps)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-index", "--no-deps",
         "--find-links", str(wheel_dir), "onnxruntime"],
        capture_output=True, text=True,
    )
    print(f"  returncode={result.returncode}")
    if result.stdout: print(f"  stdout:\n{result.stdout}")
    if result.stderr: print(f"  stderr:\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError("offline onnxruntime install failed")
    import onnxruntime as ort
    print(f"onnxruntime (installed offline): {ort.__version__}")

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
print(f"Device: {device}")
torch.set_num_threads(4)
""", "setup"))

cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths - locate competition data + R3 ckpt
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"

TEST_DIR = BASE / "test_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
print(f"BASE: {BASE}")
print(f"  test_soundscapes exists: {TEST_DIR.exists()}")

# Locate R3 ckpt (r3_fold0_ckpt_best_ns22.pth) for exp095 (l0 + R2 spec)
STATE_DIR = None
CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp095-l0-r2spec"),
    Path("/kaggle/input/birdclef2026-exp095-l0-r2spec"),
]
for p in CANDIDATES:
    if p.exists():
        if any(p.rglob("r3_fold*_ckpt_best_ns22.pth")):
            STATE_DIR = p; break

if STATE_DIR is None:
    for hit in Path("/kaggle/input").rglob("r3_fold0_ckpt_best_ns22.pth"):
        STATE_DIR = hit.parent; break

assert STATE_DIR is not None, (
    "ckpt not found. Attach maekeso/birdclef2026-exp095-l0-r2spec as dataset_sources"
)
print(f"State dir: {STATE_DIR}")
ckpt_files = sorted(STATE_DIR.rglob("r3_fold*_ckpt_best_ns22.pth"))
print(f"Found {len(ckpt_files)} R3 fold ckpt(s):")
for f in ckpt_files:
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
assert len(ckpt_files) >= 1, "Need at least 1 fold ckpt"
""", "paths"))

cells.append(code_cell(r"""# ============================================================
# Cell 3: Config - must match exp095 R3 training (= exp017 R2 spec)
# ============================================================
NUM_CLASSES = 234
SR = 32000
TRAIN_DURATION = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 256
FMIN       = 20
FMAX       = 16000

BACKBONE = "eca_nfnet_l0"   # ★ exp095: l0 (val gap -0.004) + R2 exact spec
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES

print(f"Backbone: {BACKBONE}")
print(f"Single fold inference: {len(ckpt_files)} ckpt(s)")
""", "config"))

cells.append(code_cell(r"""# ============================================================
# Cell 4: Model - rebuild eca_nfnet_l0 SED architecture
# ============================================================
class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits

print("OK model def ready")
""", "model"))

cells.append(code_cell(r"""# ============================================================
# Cell 5: Load fold0 ckpt -> Export to ONNX -> Create ort.InferenceSession
# ============================================================
ONNX_DIR = Path("/tmp/onnx_r3_exp095")
ONNX_DIR.mkdir(exist_ok=True)

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, mel):
        return self.model(mel, return_framewise=True)

mel_tf_export = MelSpecTransform().to(device)
with torch.no_grad():
    dummy_wav = torch.randn(1, 1, SR * TRAIN_DURATION, device=device)
    dummy_mel = mel_tf_export(dummy_wav)
    dummy_mel = (dummy_mel - dummy_mel.mean()) / (dummy_mel.std() + 1e-6)
print(f"Dummy mel shape for ONNX export: {tuple(dummy_mel.shape)}")
del mel_tf_export

ort_sessions = []
fold_ids = []

t_export_start = time.time()
for ckpt_path in ckpt_files:
    fold_id = int(re.search(r"fold(\d+)", ckpt_path.name).group(1))
    print(f"\n[Fold {fold_id}] Loading {ckpt_path.name}")
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    print(f"  epoch={state.get('epoch')}, "
          f"best_ns22={state.get('best_ns22', float('nan')):.4f}, "
          f"best_macro={state.get('best_macro', float('nan')):.4f}")

    model = BirdSEDModel().to(device)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()

    wrapper = ONNXWrapper(model).eval()
    onnx_path = ONNX_DIR / f"r3_fold{fold_id}.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy_mel, str(onnx_path),
            opset_version=17,
            input_names=["mel"],
            output_names=["clip_logits", "framewise"],
            dynamic_axes={
                "mel": {0: "batch"},
                "clip_logits": {0: "batch"},
                "framewise": {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  exported to {onnx_path.name} ({onnx_path.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")

    # 出力 一致確認 (PyTorch vs ONNX、最初の fold のみ)
    if len(ort_sessions) == 0:
        with torch.no_grad():
            pt_clip, pt_frame = wrapper(dummy_mel)
        sess_chk = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess_chk.run(["clip_logits", "framewise"], {"mel": dummy_mel.numpy()})
        diff_clip = np.abs(pt_clip.numpy() - ort_out[0]).max()
        diff_frame = np.abs(pt_frame.numpy() - ort_out[1]).max()
        print(f"  validation (PyTorch vs ONNX): clip_diff={diff_clip:.2e}, framewise_diff={diff_frame:.2e}")
        assert diff_clip < 1e-3 and diff_frame < 1e-3, f"ONNX output mismatch! fold {fold_id}"
        del sess_chk

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opt, providers=["CPUExecutionProvider"])
    ort_sessions.append(sess)
    fold_ids.append(fold_id)

    del model, wrapper, state
    gc.collect()

print(f"\nOK exported + loaded {len(ort_sessions)} ONNX session(s) in {time.time()-t_export_start:.1f}s")
print(f"  ONNX dir: {ONNX_DIR}, total size: {sum(f.stat().st_size for f in ONNX_DIR.glob('*.onnx'))/1e6:.1f} MB")
""", "load_ckpts"))

cells.append(code_cell(r"""# ============================================================
# Cell 6: Inference - single fold
# ============================================================
try:
    import soundfile as sf
    DECODER = "soundfile"
except ImportError:
    DECODER = "librosa"
print(f"Audio decoder: {DECODER}")

import librosa
from scipy.ndimage import convolve1d

GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
N_WINDOWS = 12
CHUNK_N = SR * TRAIN_DURATION

def load_audio_32k_mono(path):
    if DECODER == "soundfile":
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    else:
        wav, _ = librosa.load(str(path), sr=SR, mono=True)
        return wav.astype(np.float32)

def file_to_chunks(path):
    wav = load_audio_32k_mono(path)
    target_len = 60 * SR
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    chunks = wav.reshape(N_WINDOWS, CHUNK_N)
    end_times = np.arange(1, N_WINDOWS + 1) * TRAIN_DURATION
    return chunks.astype(np.float32), end_times

def sigmoid_np(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                    np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50)))
                    ).astype(np.float32)

def gauss_smooth(scores):
    smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], GAUSSIAN_KERNEL, axis=0, mode="nearest")
    return smoothed.reshape(-1, scores.shape[1])

mel_tf = MelSpecTransform().to(device)

test_files = sorted(glob.glob(f"{TEST_DIR}/*.ogg")) if TEST_DIR.is_dir() else []
if len(test_files) == 0:
    fallback = BASE / "train_soundscapes"
    if fallback.is_dir():
        test_files = sorted(glob.glob(f"{fallback}/*.ogg"))[:5]
        print(f"No test_soundscapes - using {len(test_files)} train files for debug")
print(f"Test files: {len(test_files)}")

all_rows, all_logits = [], []
t0 = time.time()

with torch.no_grad():
    for fi, fp in enumerate(test_files):
        basename = os.path.basename(fp).replace(".ogg", "")
        chunks, end_times = file_to_chunks(fp)

        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)
        mel = mel_tf(wav_t)
        for i in range(mel.size(0)):
            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
        mel_np = mel.cpu().numpy().astype(np.float32)

        # Single fold (or n-fold if >1): average framewise + clip logits
        clip_logits_sum = np.zeros((mel_np.shape[0], NUM_CLASSES), dtype=np.float32)
        frame_max_sum   = np.zeros((mel_np.shape[0], NUM_CLASSES), dtype=np.float32)
        for sess in ort_sessions:
            outs = sess.run(["clip_logits", "framewise"], {"mel": mel_np})
            clip_np      = outs[0]
            framewise_np = outs[1]
            frame_max_np = framewise_np.max(axis=1)
            clip_logits_sum += clip_np
            frame_max_sum   += frame_max_np
        n_models = len(ort_sessions)
        clip_logits_avg = clip_logits_sum / n_models
        frame_max_avg   = frame_max_sum / n_models

        blend_logits = 0.5 * clip_logits_avg + 0.5 * frame_max_avg
        all_rows.extend([f"{basename}_{int(t)}" for t in end_times])
        all_logits.append(blend_logits.astype(np.float32))

        if (fi + 1) % 50 == 0 or fi == 0 or fi == len(test_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 1e-6)
            eta = (len(test_files) - fi - 1) / max(rate, 1e-6)
            print(f"  [{fi+1:4d}/{len(test_files)}]  {elapsed:.1f}s  "
                  f"{rate:.2f} files/s  ETA {eta/60:.1f}min")

if all_logits:
    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
    logits_smoothed = gauss_smooth(logits_arr)
    probs = sigmoid_np(logits_smoothed)
else:
    probs = np.zeros((0, NUM_CLASSES), dtype=np.float32)

print(f"\n{len(ort_sessions)}-fold (ONNX) inference: {len(all_rows)} rows in {(time.time()-t0)/60:.1f} min")
""", "infer"))

cells.append(code_cell(r"""# ============================================================
# Cell 7: Write submission.csv
# ============================================================
sub = pd.DataFrame(probs, columns=PRIMARY_LABELS)
sub.insert(0, "row_id", all_rows)
out_path = Path("/kaggle/working/submission.csv")
sub.to_csv(out_path, index=False)
print(f"submission.csv: {len(sub)} rows, {sub.shape[1]-1} species, "
      f"{out_path.stat().st_size/1e6:.1f}MB")
print(sub.head(3))
""", "submit"))

nb_out = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python",
                          "file_extension": ".py"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out_path = HERE / "nb_infer_r3.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
