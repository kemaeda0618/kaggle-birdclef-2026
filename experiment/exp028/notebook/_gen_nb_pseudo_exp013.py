"""Generate exp028 nb_pseudo_exp013: Tucker public SED 5-fold ONNX predictions on train_soundscapes (Kaggle GPU).

目的: Multi-teacher R3 pseudo の teacher 2 として、Tucker public 5-fold SED の
train_soundscapes 全件 (~10,658 files) 予測を生成・保存する。

Tucker は ONNX (HGNetV2 系)、CUDAExecutionProvider 優先で GPU 推論 (fallback CPU)。
output: /kaggle/working/pseudo_exp013.csv

Run: python _gen_nb_pseudo_exp013.py  ->  writes nb_pseudo_exp013.ipynb
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

cells.append(md_cell(r"""# exp028 nb_pseudo_exp013: l0 R2 5-fold predictions on train_soundscapes (Kaggle GPU)

Multi-teacher R3 pseudo の teacher 1 として、exp020 R2 5-fold ensemble の
train_soundscapes 全件 (~10,658 files) 予測を生成・保存する。

PyTorch 直 (ONNX 不要)、Kaggle T4 GPU で ~1-2 h。

## Inputs
- `birdclef-2026` train_soundscapes (~10,658 files)
- `maekeso/birdclef2026-exp020-weights-5fold` (`r2_fold{0..4}_ckpt_best_ns22.pth`)

## Output
- `/kaggle/working/pseudo_exp013.csv` (filename, start_sec, end_sec, [234 species probs])
- ~10,658 files × 12 windows = ~128k rows × 234 species ≈ 100-200 MB CSV

## 後段
R3 学習 NB が pseudo_exp013 + pseudo_exp013 + pseudo_e17 の 3 つを平均 →
Multi-teacher pseudo として student に投入 (Babych BC25 1位 recipe)。

## R2 5-fold val 実績 (teacher 品質の参考)
- fold 0: 0.9177、fold 1: 0.9469、fold 2: 0.9254、fold 3: 0.9457、fold 4: 0.9431
- avg: 0.9358 (R1 avg 0.9212 から +0.0146)
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

import warnings
warnings.filterwarnings("ignore")

# Tucker は ONNX → onnxruntime offline install from wheel dataset
import subprocess
try:
    import onnxruntime as ort
    print(f"onnxruntime (pre-installed): {ort.__version__}")
except ImportError:
    WHEEL_CANDIDATES = [
        Path("/kaggle/input/datasets/romantamrazov/onnxruntime-1-24-4"),
        Path("/kaggle/input/onnxruntime-1-24-4"),
    ]
    wheel_dir = next((p for p in WHEEL_CANDIDATES if p.exists()), None)
    assert wheel_dir is not None, "wheel dir not found - attach romantamrazov/onnxruntime-1-24-4"
    print(f"Installing onnxruntime offline from {wheel_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-index", "--no-deps",
         "--find-links", str(wheel_dir), "onnxruntime"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError("offline install failed")
    import onnxruntime as ort
    print(f"onnxruntime (installed offline): {ort.__version__}")
print(f"  available providers: {ort.get_available_providers()}")

# GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device} (mel transform に使用)")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

torch.set_num_threads(4)
""", "setup"))

cells.append(code_cell(r"""# ============================================================
# Cell 2: Paths — locate competition data + R1 ckpts
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

# Locate exp013 student_sed ONNX (single fold R1)
STATE_DIR = None
CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp013-r1-student-sed"),
    Path("/kaggle/input/birdclef2026-exp013-r1-student-sed"),
]
for p in CANDIDATES:
    if p.exists() and (p / "student_sed_fold0.onnx").exists():
        STATE_DIR = p; break

if STATE_DIR is None:
    for hit in Path("/kaggle/input").rglob("student_sed_fold0.onnx"):
        STATE_DIR = hit.parent; break

assert STATE_DIR is not None, (
    "exp013 ONNX not found. Attach maekeso/birdclef2026-exp013-r1-student-sed"
)
print(f"exp013 ONNX dir: {STATE_DIR}")
ckpt_files = [STATE_DIR / "student_sed_fold0.onnx"]   # single ONNX
print(f"Found {len(ckpt_files)} exp013 ONNX (single fold R1):")
for f in ckpt_files:
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
assert len(ckpt_files) >= 1, "Need at least 1 fold ckpt"
""", "paths"))

cells.append(code_cell(r"""# ============================================================
# Cell 3: Config — must match training (R1 NB)
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

BACKBONE = "eca_nfnet_l0"
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM = 1536

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == NUM_CLASSES

print(f"Backbone: {BACKBONE}")
print(f"Ensemble: {len(ckpt_files)} folds")
""", "config"))

cells.append(code_cell(r"""# ============================================================
# Cell 4: Model — rebuild eca_nfnet_l0 SED architecture
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
# Cell 5: Tucker 5-fold ONNX sessions (CUDA provider 優先、CPU fallback)
# ============================================================
# Tucker は ONNX 形式 (PyTorch model なし)、onnxruntime で直接 inference

def make_tucker_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # CUDA 優先、利用不可なら CPU fallback (Kaggle GPU image なら CUDA 期待)
    providers_preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=providers_preferred)

tucker_sessions = []
for ckpt_path in ckpt_files:
    fold_id = 0   # exp013 は single fold (R1 only)
    sess = make_tucker_session(ckpt_path)
    tucker_sessions.append(sess)
    actual_provider = sess.get_providers()[0]
    print(f"[Fold {fold_id}] {ckpt_path.name} loaded, provider={actual_provider}")
    # Input/output 確認
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    if fold_id == 0:
        print(f"  inputs:  {[(i.name, i.shape) for i in inputs]}")
        print(f"  outputs: {[(o.name, o.shape) for o in outputs]}")

print(f"\nOK loaded {len(tucker_sessions)} exp013 ONNX session (single fold R1)")
""", "load_ckpts"))

cells.append(code_cell(r"""# ============================================================
# Cell 6: Inference on train_soundscapes (Tucker 5-fold ONNX ensemble)
# ============================================================
try:
    import soundfile as sf
    DECODER = "soundfile"
except ImportError:
    DECODER = "librosa"
print(f"Audio decoder: {DECODER}")

import librosa

# Tucker SED mel config (exp019 blend NB / exp010 _gen_nb_blend_tucker.py より)
N_MELS_TUCKER = 256
N_FFT_TUCKER  = 2048
HOP_TUCKER    = 512
FMIN_TUCKER   = 20
FMAX_TUCKER   = 16000
TOP_DB_TUCKER = 80

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
    start_times = np.arange(0, N_WINDOWS) * TRAIN_DURATION
    end_times   = np.arange(1, N_WINDOWS + 1) * TRAIN_DURATION
    return chunks.astype(np.float32), start_times, end_times

def chunks_to_mel_tucker(chunks):
    # Tucker SED 用 mel spec (librosa、CPU)。出力 shape: (N_WINDOWS, 1, N_MELS, T)
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_TUCKER, hop_length=HOP_TUCKER,
            n_mels=N_MELS_TUCKER, fmin=FMIN_TUCKER, fmax=FMAX_TUCKER, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_TUCKER)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)

def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

# train_soundscapes 全件発見
TRAIN_SS_DIR = BASE / "train_soundscapes"
assert TRAIN_SS_DIR.is_dir(), f"train_soundscapes not found: {TRAIN_SS_DIR}"
train_ss_files = sorted(glob.glob(f"{TRAIN_SS_DIR}/*.ogg"))
print(f"train_soundscapes files: {len(train_ss_files)}")
assert len(train_ss_files) > 0

DEBUG_LIMIT = None   # 動作確認時は 20 など
if DEBUG_LIMIT is not None:
    train_ss_files = train_ss_files[:DEBUG_LIMIT]
    print(f"DEBUG MODE: limiting to {DEBUG_LIMIT} files")

# session input name 確認
input_name = tucker_sessions[0].get_inputs()[0].name
print(f"Tucker ONNX input name: {input_name}")

all_filenames = []
all_start_sec = []
all_end_sec   = []
all_probs     = []
t0 = time.time()

for fi, fp in enumerate(train_ss_files):
    basename = os.path.basename(fp).replace(".ogg", "")
    try:
        chunks, start_times, end_times = file_to_chunks(fp)
        mel = chunks_to_mel_tucker(chunks)   # (12, 1, 256, T)
    except Exception as e:
        print(f"  [skip] {basename}: {e}")
        continue

    # 5-fold ensemble: 各 fold で clip + frame の sigmoid 後 平均
    p_sum = np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)
    for sess in tucker_sessions:
        outs = sess.run(None, {input_name: mel})
        clip_logits = outs[0]                  # (B, num_classes)
        frame_max   = outs[1].max(axis=1)      # (B, num_classes)
        # 公開 NB と同じ: clip と frame の sigmoid 後 50:50 blend
        p_sum += 0.5 * sigmoid_np(clip_logits) + 0.5 * sigmoid_np(frame_max)
    p_mean = p_sum / len(tucker_sessions)

    all_filenames.extend([basename] * N_WINDOWS)
    all_start_sec.extend(start_times.tolist())
    all_end_sec.extend(end_times.tolist())
    all_probs.append(p_mean.astype(np.float32))

    if (fi + 1) % 200 == 0 or fi == 0 or fi == len(train_ss_files) - 1:
        elapsed = time.time() - t0
        rate = (fi + 1) / max(elapsed, 1e-6)
        eta = (len(train_ss_files) - fi - 1) / max(rate, 1e-6)
        print(f"  [{fi+1:5d}/{len(train_ss_files)}]  {elapsed/60:.1f}min  "
              f"{rate:.2f} files/s  ETA {eta/60:.1f}min")

if all_probs:
    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
else:
    probs = np.zeros((0, NUM_CLASSES), dtype=np.float32)

print(f"\n5-fold Tucker ONNX ensemble: {len(all_filenames)} rows in {(time.time()-t0)/60:.1f} min")
print(f"probs shape: {probs.shape}, range: [{probs.min():.4f}, {probs.max():.4f}], mean: {probs.mean():.4f}")
""", "infer"))

cells.append(code_cell(r"""# ============================================================
# Cell 7: Write pseudo_exp013.csv (R3 学習の input になる teacher pseudo)
# ============================================================
df_pseudo = pd.DataFrame(probs, columns=PRIMARY_LABELS)
df_pseudo.insert(0, "filename",  all_filenames)
df_pseudo.insert(1, "start_sec", all_start_sec)
df_pseudo.insert(2, "end_sec",   all_end_sec)

out_path = Path("/kaggle/working/pseudo_exp013.csv")
df_pseudo.to_csv(out_path, index=False)
print(f"pseudo_exp013.csv: {len(df_pseudo)} rows, {df_pseudo.shape[1]-3} species cols, "
      f"{out_path.stat().st_size/1e6:.1f}MB")
print(f"\nSummary:")
print(f"  files covered: {df_pseudo['filename'].nunique()}")
print(f"  rows per file: {len(df_pseudo) / df_pseudo['filename'].nunique():.1f}")
print(f"\nHead:")
print(df_pseudo.head(3).iloc[:, :8])
print(f"\nMean prob per species (top 10):")
print(probs.mean(axis=0)[np.argsort(probs.mean(axis=0))[::-1][:10]])

# 必要なら numpy 形式でも保存 (load 速度高速、size 小)
np.save("/kaggle/working/pseudo_exp013.npy", probs.astype(np.float16))
np.save("/kaggle/working/pseudo_exp013_meta.npy", np.array(list(zip(all_filenames, all_start_sec, all_end_sec)), dtype=object))
print(f"\nAlso saved: pseudo_exp013.npy (float16, {Path('/kaggle/working/pseudo_exp013.npy').stat().st_size/1e6:.1f}MB)")
print(f"            pseudo_exp013_meta.npy")
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
out_path = HERE / "nb_pseudo_exp013.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path} ({len(cells)} cells)")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
