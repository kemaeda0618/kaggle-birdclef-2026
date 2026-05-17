"""Generate exp010 NB1f: AVES (wav2vec2-base) embedding extraction.

Extracts 768-d wav2vec2 embeddings for:
- train_soundscapes (10,658 files × 12 windows)
- train_audio (46k files × variable windows)

Output: aves_sc_embeddings.npz, aves_sc_meta.parquet,
        aves_trainaudio_embeddings.npz, aves_trainaudio_meta.parquet

Runs on T4 GPU, ~1-2 hours.
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
    "# exp010 NB1f: AVES (wav2vec2-base) embedding extraction\n"
    "\n"
    "Extract 768-d wav2vec2-base embeddings for BC2026 train_soundscapes + train_audio.\n"
    "EDA result: Spearman vs Perch = 0.417 (most independent), gap_normalized = 0.785 (high signal).\n"
    "\n"
    "Output: 4 files for use as 3rd blend axis."))

cells.append(code_cell("install",
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'transformers', 'torchaudio', 'librosa', 'soundfile'], check=False)\n"
    "print('Install attempted')"))

cells.append(code_cell("imports", r"""import os, gc, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__}, device={DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")"""))

cells.append(code_cell("config", r"""# Paths and config
BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")
TRAIN_SC_DIR = BASE / "train_soundscapes"
AUDIO_DIR    = BASE / "train_audio"
TRAIN_CSV    = BASE / "train.csv"

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR_BC = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR_BC * WINDOW_SEC
N_WINDOWS_SC = 12

# AVES = wav2vec2-base (fallback if `earthspecies/aves-base-bio` unavailable)
AVES_MODEL = "facebook/wav2vec2-base"
AVES_SR = 16_000
BATCH = 16

print(f"BASE={BASE}")"""))

cells.append(code_cell("model", r"""# Load model
from transformers import AutoFeatureExtractor, AutoModel

# Try real AVES first, then fallback
fallbacks = ["earthspecies/aves-base-bio", "facebook/wav2vec2-base"]
fe = None; model = None
for name in fallbacks:
    try:
        fe = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(DEVICE).eval()
        AVES_MODEL = name
        print(f"Loaded: {name}, SR={fe.sampling_rate}")
        break
    except Exception as e:
        print(f"  {name}: failed ({type(e).__name__})")
assert model is not None
AVES_SR = fe.sampling_rate

# Smoke test
with torch.no_grad():
    dummy = [np.random.randn(AVES_SR * 5).astype(np.float32)]
    inp = fe(dummy, sampling_rate=AVES_SR, return_tensors="pt").to(DEVICE)
    out = model(**inp)
    EMB_DIM = out.last_hidden_state.shape[-1]
print(f"emb_dim = {EMB_DIM}")"""))

cells.append(code_cell("inference-fn", r"""# Inference helper
def read_audio_full(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR_BC:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR_BC)
    return y.astype(np.float32)


def to_aves_sr(y_32k):
    return librosa.resample(y_32k, orig_sr=SR_BC, target_sr=AVES_SR).astype(np.float32)


@torch.no_grad()
def extract_embeddings(waves_at_aves_sr, batch=BATCH):
    # Returns (N, EMB_DIM) numpy float32
    out = []
    for i in range(0, len(waves_at_aves_sr), batch):
        b = waves_at_aves_sr[i:i + batch]
        inp = fe(b, sampling_rate=AVES_SR, return_tensors="pt", padding=True).to(DEVICE)
        h = model(**inp).last_hidden_state.mean(dim=1)
        out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


print("Helpers ready.")"""))

cells.append(code_cell("soundscapes", r"""# === Process train_soundscapes ===
import gc as _gc
sc_files = sorted(TRAIN_SC_DIR.glob("*.ogg"))
n_sc = len(sc_files)
print(f"Soundscape files: {n_sc}")

CHUNK = 30  # files per chunk to manage memory

n_total_windows = n_sc * N_WINDOWS_SC
sc_emb = np.zeros((n_total_windows, EMB_DIM), dtype=np.float16)
sc_meta_rows = []

t0 = time.time()
for ci in range(0, n_sc, CHUNK):
    chunk_paths = sc_files[ci:ci + CHUNK]
    waves_aves = []
    for fp in chunk_paths:
        y = read_audio_full(fp)
        target = SR_BC * 60
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        # 12 windows of 5sec, then resample each to AVES_SR
        windows_32k = y.reshape(N_WINDOWS_SC, WINDOW_SAMPLES)
        for wi in range(N_WINDOWS_SC):
            w = to_aves_sr(windows_32k[wi])
            waves_aves.append(w)
    emb = extract_embeddings(waves_aves)
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
    if (ci // CHUNK) % 10 == 0 or ci + CHUNK >= n_sc:
        elapsed = time.time() - t0
        print(f"  SC [{min(ci+CHUNK, n_sc)}/{n_sc}] {elapsed:.0f}s")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _gc.collect()

print(f"SC emb: {sc_emb.shape}, {sc_emb.nbytes/1e6:.0f} MB ({time.time()-t0:.0f}s)")
sc_meta_df = pd.DataFrame(sc_meta_rows)

np.savez_compressed(OUT_DIR / "aves_sc_embeddings.npz", embeddings=sc_emb)
sc_meta_df.to_parquet(OUT_DIR / "aves_sc_meta.parquet", index=False)
print(f"Saved aves_sc_*  rows={len(sc_meta_df)}")
del sc_emb; _gc.collect()"""))

cells.append(code_cell("trainaudio", r"""# === Process train_audio ===
train_df = pd.read_csv(TRAIN_CSV)
audio_files = []
for _, row in train_df.iterrows():
    fp = AUDIO_DIR / row["filename"]
    if fp.exists():
        audio_files.append({
            "path": fp, "filename": row["filename"],
            "primary_label": str(row["primary_label"]),
        })
n_audio = len(audio_files)
print(f"Train audio files: {n_audio}")

CHUNK_A = 200
ta_emb_chunks = []
ta_meta_rows = []
t0 = time.time()

for ci in range(0, n_audio, CHUNK_A):
    chunk = audio_files[ci:ci + CHUNK_A]
    waves_aves = []
    chunk_meta = []
    for finfo in chunk:
        try:
            y = read_audio_full(finfo["path"])
        except Exception as e:
            print(f"  SKIP {finfo['filename']}: {e}")
            continue
        n_full = len(y) // WINDOW_SAMPLES
        rem = len(y) % WINDOW_SAMPLES
        windows_32k = []
        if n_full > 0:
            windows_32k.append(y[:n_full * WINDOW_SAMPLES].reshape(n_full, WINDOW_SAMPLES))
        if rem > 0:
            last = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            last[:rem] = y[n_full * WINDOW_SAMPLES:]
            windows_32k.append(last.reshape(1, WINDOW_SAMPLES))
        if not windows_32k:
            windows_32k.append(np.zeros((1, WINDOW_SAMPLES), dtype=np.float32))
        windows_32k = np.concatenate(windows_32k, axis=0)
        n_win = windows_32k.shape[0]
        for wi in range(n_win):
            w = to_aves_sr(windows_32k[wi])
            waves_aves.append(w)
            chunk_meta.append({
                "filename": finfo["filename"],
                "primary_label": finfo["primary_label"],
                "window_idx": wi,
                "n_windows": n_win,
            })
    if not waves_aves:
        continue
    emb = extract_embeddings(waves_aves)
    ta_emb_chunks.append(emb.astype(np.float16))
    ta_meta_rows.extend(chunk_meta)
    if (ci // CHUNK_A) % 5 == 0 or ci + CHUNK_A >= n_audio:
        elapsed = time.time() - t0
        print(f"  TA [{min(ci+CHUNK_A, n_audio)}/{n_audio}] windows={sum(c.shape[0] for c in ta_emb_chunks)}, {elapsed/60:.1f}min")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    import gc as _gc; _gc.collect()

ta_emb = np.concatenate(ta_emb_chunks, axis=0)
print(f"TA emb: {ta_emb.shape}, {ta_emb.nbytes/1e6:.0f} MB")
ta_meta_df = pd.DataFrame(ta_meta_rows)

np.savez_compressed(OUT_DIR / "aves_trainaudio_embeddings.npz", embeddings=ta_emb)
ta_meta_df.to_parquet(OUT_DIR / "aves_trainaudio_meta.parquet", index=False)
print(f"Saved aves_trainaudio_*  rows={len(ta_meta_df)}")"""))

cells.append(code_cell("summary", r"""# === Summary ===
for p in sorted(OUT_DIR.glob("aves_*")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = HERE / "nb1f_aves_embed.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out} ({len(cells)} cells)")
