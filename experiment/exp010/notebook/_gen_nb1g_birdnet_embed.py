"""Generate exp010 NB1g: BirdNET ONNX classifier-output extraction.

Extracts 6522-d BirdNET v2.4 classifier output (sigmoid scores) for:
- train_soundscapes
- train_audio

Note: dhruvpaidukle/birdnet-onnx exposes classifier output (6522d), not embeddings.
We use full 6522d output as feature vector.

Output: birdnet_sc_*, birdnet_trainaudio_*

Runs on T4 GPU, ~30-60 min.
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
    "# exp010 NB1g: BirdNET v2.4 ONNX classifier-output extraction\n"
    "\n"
    "Extract 6522-d BirdNET classifier output for BC2026 SS + train_audio.\n"
    "EDA result: Spearman vs Perch = 0.660, gap_normalized = 0.836 (high signal)."))

cells.append(code_cell("install",
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'onnxruntime-gpu', 'librosa', 'soundfile'], check=False)\n"
    "print('Install attempted')"))

cells.append(code_cell("imports", r"""import os, gc, time, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
import torch
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"onnxruntime={ort.__version__}, providers={ort.get_available_providers()}")"""))

cells.append(code_cell("config", r"""BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")
TRAIN_SC_DIR = BASE / "train_soundscapes"
AUDIO_DIR    = BASE / "train_audio"
TRAIN_CSV    = BASE / "train.csv"
OUT_DIR = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)

SR_BC = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR_BC * WINDOW_SEC
N_WINDOWS_SC = 12

# BirdNET v2.4 expects 48kHz, 3-second windows
SR_BN = 48_000
WIN_BN = 3 * SR_BN   # 144000
BATCH = 32"""))

cells.append(code_cell("model", r"""# Locate BirdNET ONNX
BN_MODEL = None
for f in Path("/kaggle/input").rglob("*.onnx"):
    if "birdnet" in str(f).lower():
        BN_MODEL = f; break
assert BN_MODEL is not None, "BirdNET ONNX not found — attach dhruvpaidukle/birdnet-onnx"
print(f"BirdNET model: {BN_MODEL}")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(BN_MODEL), sess_options=so,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"Providers: {sess.get_providers()}")
inp = sess.get_inputs()[0]
print(f"  Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
for o in sess.get_outputs():
    print(f"  Output: {o.name}, shape={o.shape}")

# smoke test
out = sess.run(None, {inp.name: np.random.randn(2, WIN_BN).astype(np.float32)})
for arr in out:
    print(f"  out shape={arr.shape}, dtype={arr.dtype}")
EMB_DIM = out[0].shape[-1] if out[0].ndim == 2 else 6522
print(f"emb_dim = {EMB_DIM}")"""))

cells.append(code_cell("inference-fn", r"""def read_audio_32k(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR_BC:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR_BC)
    return y.astype(np.float32)


def window_to_birdnet(w_32k):
    # 5sec @ 32kHz → 5sec @ 48kHz → center 3sec
    w48 = librosa.resample(w_32k, orig_sr=SR_BC, target_sr=SR_BN)
    start = max(0, (len(w48) - WIN_BN) // 2)
    x = w48[start:start + WIN_BN]
    if len(x) < WIN_BN:
        x = np.pad(x, (0, WIN_BN - len(x)))
    return x.astype(np.float32)


def extract_birdnet(waves_3s_48k):
    out_emb = []
    for i in range(0, len(waves_3s_48k), BATCH):
        batch = np.stack(waves_3s_48k[i:i + BATCH]).astype(np.float32)
        outs = sess.run(None, {inp.name: batch})
        # Pick the first 2D output as our feature
        for arr in outs:
            if arr.ndim == 2:
                out_emb.append(arr.astype(np.float32))
                break
    return np.concatenate(out_emb, axis=0)


print("Helpers ready.")"""))

cells.append(code_cell("soundscapes", r"""# === train_soundscapes ===
sc_files = sorted(TRAIN_SC_DIR.glob("*.ogg"))
n_sc = len(sc_files)
print(f"Soundscape files: {n_sc}")
CHUNK = 50

sc_emb = np.zeros((n_sc * N_WINDOWS_SC, EMB_DIM), dtype=np.float16)
sc_meta_rows = []
t0 = time.time()

for ci in range(0, n_sc, CHUNK):
    chunk_paths = sc_files[ci:ci + CHUNK]
    waves_bn = []
    for fp in chunk_paths:
        y = read_audio_32k(fp)
        target = SR_BC * 60
        if len(y) < target: y = np.pad(y, (0, target - len(y)))
        else: y = y[:target]
        windows_32k = y.reshape(N_WINDOWS_SC, WINDOW_SAMPLES)
        for wi in range(N_WINDOWS_SC):
            waves_bn.append(window_to_birdnet(windows_32k[wi]))
    emb = extract_birdnet(waves_bn)
    row_start = ci * N_WINDOWS_SC
    sc_emb[row_start:row_start + len(emb)] = emb.astype(np.float16)
    for fp in chunk_paths:
        for wi in range(N_WINDOWS_SC):
            end_sec = (wi + 1) * WINDOW_SEC
            sc_meta_rows.append({
                "row_id": f"{fp.stem}_{end_sec}",
                "filename": fp.name,
                "window_idx": wi,
            })
    if (ci // CHUNK) % 5 == 0 or ci + CHUNK >= n_sc:
        print(f"  SC [{min(ci+CHUNK, n_sc)}/{n_sc}] {time.time()-t0:.0f}s")
    gc.collect()

sc_meta_df = pd.DataFrame(sc_meta_rows)
np.savez_compressed(OUT_DIR / "birdnet_sc_embeddings.npz", embeddings=sc_emb)
sc_meta_df.to_parquet(OUT_DIR / "birdnet_sc_meta.parquet", index=False)
print(f"Saved birdnet_sc_*  rows={len(sc_meta_df)}")
del sc_emb; gc.collect()"""))

cells.append(code_cell("trainaudio", r"""# === train_audio ===
train_df = pd.read_csv(TRAIN_CSV)
audio_files = []
for _, row in train_df.iterrows():
    fp = AUDIO_DIR / row["filename"]
    if fp.exists():
        audio_files.append({"path": fp, "filename": row["filename"],
                             "primary_label": str(row["primary_label"])})
n_audio = len(audio_files)
print(f"Train audio: {n_audio}")
CHUNK_A = 200

ta_emb_chunks = []
ta_meta_rows = []
t0 = time.time()
for ci in range(0, n_audio, CHUNK_A):
    chunk = audio_files[ci:ci + CHUNK_A]
    waves_bn = []
    chunk_meta = []
    for finfo in chunk:
        try:
            y = read_audio_32k(finfo["path"])
        except Exception as e:
            continue
        n_full = len(y) // WINDOW_SAMPLES
        rem = len(y) % WINDOW_SAMPLES
        ws_32k = []
        if n_full > 0: ws_32k.append(y[:n_full*WINDOW_SAMPLES].reshape(n_full, WINDOW_SAMPLES))
        if rem > 0:
            last = np.zeros(WINDOW_SAMPLES, dtype=np.float32); last[:rem] = y[n_full*WINDOW_SAMPLES:]
            ws_32k.append(last.reshape(1, WINDOW_SAMPLES))
        if not ws_32k: ws_32k.append(np.zeros((1, WINDOW_SAMPLES), dtype=np.float32))
        ws_32k = np.concatenate(ws_32k, axis=0)
        n_win = ws_32k.shape[0]
        for wi in range(n_win):
            waves_bn.append(window_to_birdnet(ws_32k[wi]))
            chunk_meta.append({"filename": finfo["filename"],
                                "primary_label": finfo["primary_label"],
                                "window_idx": wi, "n_windows": n_win})
    if not waves_bn: continue
    emb = extract_birdnet(waves_bn)
    ta_emb_chunks.append(emb.astype(np.float16))
    ta_meta_rows.extend(chunk_meta)
    if (ci // CHUNK_A) % 5 == 0 or ci + CHUNK_A >= n_audio:
        print(f"  TA [{min(ci+CHUNK_A, n_audio)}/{n_audio}] windows={sum(c.shape[0] for c in ta_emb_chunks)}, {time.time()-t0:.0f}s")
    gc.collect()

ta_emb = np.concatenate(ta_emb_chunks, axis=0)
print(f"TA emb: {ta_emb.shape}, {ta_emb.nbytes/1e6:.0f} MB")
ta_meta_df = pd.DataFrame(ta_meta_rows)
np.savez_compressed(OUT_DIR / "birdnet_trainaudio_embeddings.npz", embeddings=ta_emb)
ta_meta_df.to_parquet(OUT_DIR / "birdnet_trainaudio_meta.parquet", index=False)
print(f"Saved birdnet_trainaudio_*  rows={len(ta_meta_df)}")"""))

cells.append(code_cell("summary", r"""for p in sorted(OUT_DIR.glob("birdnet_*")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")"""))

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}},
    "nbformat": 4, "nbformat_minor": 5}
out = HERE / "nb1g_birdnet_embed.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out} ({len(cells)} cells)")
