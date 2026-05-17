"""Generate exp010 NB1: Perch ONNX Embedding Extraction.

Extracts Perch v2 embeddings + mapped scores for:
- train_soundscapes (10,658 files, 60s each)
- train_audio (46,207 files, variable length)

Output: npz + parquet saved to /kaggle/working/ -> upload as Kaggle Dataset.

Usage: python _gen_nb1_embedding.py
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
cells.append(md_cell("hdr", "# exp010 NB1: Perch v2 ONNX Embedding Extraction\n"
    "\n"
    "Extract Perch v2 embeddings (1536d) + mapped logits (234 classes) for:\n"
    "- **train_soundscapes**: 10,658 files x 12 windows = ~128K windows\n"
    "- **train_audio**: 46,207 files x variable windows = ~139K windows\n"
    "\n"
    "Output files (upload as Kaggle Dataset):\n"
    "- `soundscape_embeddings.npz` -- embeddings + scores\n"
    "- `soundscape_meta.parquet` -- filename, site, hour, row_id\n"
    "- `trainaudio_embeddings.npz` -- embeddings + scores\n"
    "- `trainaudio_meta.parquet` -- filename, primary_label, window_idx"))

# ── Install ──
INSTALL_CODE = (
    "# Install ONNX Runtime GPU + HuggingFace Hub\n"
    "import subprocess, sys, os, time\n"
    "\n"
    "START = time.time()\n"
    "\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                       'onnxruntime-gpu', 'huggingface_hub'])\n"
    'print("Installed onnxruntime-gpu, huggingface_hub")'
)
cells.append(code_cell("install", INSTALL_CODE))

# ── Imports ──
IMPORTS_CODE = (
    "import gc, re, warnings\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import soundfile as sf\n"
    "import onnxruntime as ort\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    'print(f"onnxruntime {ort.__version__}")\n'
    'print(f"Providers: {ort.get_available_providers()}")'
)
cells.append(code_cell("imports", IMPORTS_CODE))

# ── Config ──
CONFIG_CODE = """\
# CONFIG
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160,000
FILE_DURATION = 60
N_WINDOWS_SC = FILE_DURATION // WINDOW_SEC  # 12

# Paths
BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")

ONNX_MODEL = None  # Downloaded from HuggingFace below

# labels.csv - try multiple paths
LABELS_CSV = None
for _p in [
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1/assets/labels.csv",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1/assets/labels.csv",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2/assets/labels.csv",
]:
    if Path(_p).exists():
        LABELS_CSV = _p
        break
assert LABELS_CSV is not None, "labels.csv not found!"

OUT_DIR = Path("/kaggle/working")
SC_DIR = BASE / "train_soundscapes"
AUDIO_DIR = BASE / "train_audio"
TRAIN_CSV = BASE / "train.csv"
TAXONOMY_CSV = BASE / "taxonomy.csv"

BATCH_WINDOWS = 48  # windows per GPU batch (T4 16GB VRAM limit)
CHUNK_SC = 50  # soundscape files per chunk
CHUNK_AUDIO = 1000  # train_audio files per chunk

print(f"BASE: {BASE}")
print(f"ONNX model: {ONNX_MODEL}")
print(f"Labels CSV: {LABELS_CSV}")"""
cells.append(code_cell("config", CONFIG_CODE))

# ── ONNX Session ──
ONNX_CODE = """\
# DOWNLOAD ONNX MODEL & CREATE SESSION (GPU)
from huggingface_hub import hf_hub_download

print("Downloading Perch v2 ONNX from HuggingFace...")
t0 = time.time()
ONNX_MODEL = hf_hub_download(
    repo_id="justinchuby/Perch-onnx",
    filename="perch_v2.onnx",
)
print(f"Downloaded in {time.time()-t0:.1f}s: {ONNX_MODEL}")

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=providers)

active = session.get_providers()
print(f"Active providers: {active}")
assert "CUDAExecutionProvider" in active, "GPU not available!"

for inp in session.get_inputs():
    print(f"  Input:  {inp.name}, shape={inp.shape}, type={inp.type}")
for out in session.get_outputs():
    print(f"  Output: {out.name}, shape={out.shape}, type={out.type}")"""
cells.append(code_cell("onnx-session", ONNX_CODE))

# ── Taxonomy & Label Mapping ──
TAXONOMY_CODE = """\
# TAXONOMY & PERCH LABEL MAPPING
taxonomy = pd.read_csv(TAXONOMY_CSV)
PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

# Perch label mapping
bc_labels = (
    pd.read_csv(LABELS_CSV)
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
NO_LABEL_INDEX = len(bc_labels)

taxonomy_m = taxonomy.copy()
taxonomy_m["scientific_name_lookup"] = taxonomy_m["scientific_name"]
bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

mapping = taxonomy_m.merge(
    bc_lookup[["scientific_name_lookup", "bc_index"]],
    on="scientific_name_lookup", how="left",
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)
label_to_bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES = np.array([int(label_to_bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC = BC_INDICES[MAPPED_MASK].astype(np.int32)

# Genus proxies for unmapped species
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sono = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
]

proxy_map = {}
for _, row in unmapped_non_sono.iterrows():
    genus = str(row["scientific_name"]).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(
            rf"^{re.escape(genus)}\\s", na=False
        )
    ]
    if len(hits) > 0:
        proxy_map[label_to_idx[row["primary_label"]]] = (
            hits["bc_index"].astype(int).values
        )

print(f"Species: {N_CLASSES}")
print(f"  Mapped to Perch: {MAPPED_MASK.sum()}")
print(f"  Unmapped: {(~MAPPED_MASK).sum()}")
print(f"  Genus proxies: {len(proxy_map)}")"""
cells.append(code_cell("taxonomy", TAXONOMY_CODE))

# ── Inference Functions ──
INFER_CODE = """\
# INFERENCE FUNCTIONS

def read_audio(path, target_samples=None):
    # Read audio file, mono, pad/truncate to target_samples if given.
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import torchaudio, torch
        y = torch.from_numpy(y).unsqueeze(0)
        y = torchaudio.functional.resample(y, sr, SR).squeeze(0).numpy()
    if target_samples is not None:
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
    return y


def split_windows(y):
    # Split audio into WINDOW_SEC-second windows. Pad last partial window.
    n_full = len(y) // WINDOW_SAMPLES
    remainder = len(y) % WINDOW_SAMPLES
    windows = []
    if n_full > 0:
        windows.append(y[:n_full * WINDOW_SAMPLES].reshape(n_full, WINDOW_SAMPLES))
    if remainder > 0:
        last = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        last[:remainder] = y[n_full * WINDOW_SAMPLES:]
        windows.append(last.reshape(1, WINDOW_SAMPLES))
    if len(windows) == 0:
        return np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    return np.concatenate(windows, axis=0)


def map_logits_to_scores(logits):
    # Map Perch logits (N, 14795) to competition scores (N, 234).
    scores = np.zeros((logits.shape[0], N_CLASSES), dtype=np.float32)
    scores[:, MAPPED_POS] = logits[:, MAPPED_BC]
    for pos, bc_idx_arr in proxy_map.items():
        scores[:, pos] = logits[:, bc_idx_arr].max(axis=1)
    return scores


def infer_batch(windows):
    # Run ONNX inference on windows. Returns (embeddings, scores).
    n = windows.shape[0]
    all_emb = []
    all_scores = []

    for i in range(0, n, BATCH_WINDOWS):
        batch = windows[i:i + BATCH_WINDOWS]
        outputs = session.run(None, {"inputs": batch})
        out_dict = {o.name: v for o, v in zip(session.get_outputs(), outputs)}

        emb = out_dict["embedding"].astype(np.float32)
        logits = out_dict["label"].astype(np.float32)

        all_emb.append(emb)
        all_scores.append(map_logits_to_scores(logits))

    return np.concatenate(all_emb), np.concatenate(all_scores)


print("Inference functions ready.")

# Quick test
test_input = np.random.randn(2, WINDOW_SAMPLES).astype(np.float32)
t0 = time.time()
test_emb, test_scores = infer_batch(test_input)
print(f"Test: {time.time()-t0:.3f}s, emb={test_emb.shape}, scores={test_scores.shape}")
del test_input, test_emb, test_scores"""
cells.append(code_cell("infer-fn", INFER_CODE))

# ── Process train_soundscapes ──
SC_CODE = """\
# PROCESS TRAIN_SOUNDSCAPES
def parse_sc_filename(fname):
    # Extract site and hour from soundscape filename.
    parts = Path(fname).stem.split("_")
    site = next((p for p in parts if p.startswith("S")), "UNK")
    time_str = parts[-1] if len(parts) >= 6 else "000000"
    hour = int(time_str[:2]) if len(time_str) >= 2 else 0
    return site, hour

sc_files = sorted(SC_DIR.glob("*.ogg"))
n_sc = len(sc_files)
n_rows_sc = n_sc * N_WINDOWS_SC

print(f"Soundscape files: {n_sc}")
print(f"Total windows: {n_rows_sc}")

# Pre-allocate
sc_emb = np.zeros((n_rows_sc, 1536), dtype=np.float16)
sc_scores = np.zeros((n_rows_sc, N_CLASSES), dtype=np.float16)
sc_meta_rows = []

t0 = time.time()

for ci in tqdm(range(0, n_sc, CHUNK_SC), desc="Soundscapes"):
    chunk_paths = sc_files[ci:ci + CHUNK_SC]
    chunk_n = len(chunk_paths)

    # Read all files in chunk
    all_windows = np.empty((chunk_n * N_WINDOWS_SC, WINDOW_SAMPLES), dtype=np.float32)
    for fi, fpath in enumerate(chunk_paths):
        y = read_audio(str(fpath), target_samples=SR * FILE_DURATION)
        offset = fi * N_WINDOWS_SC
        all_windows[offset:offset + N_WINDOWS_SC] = y.reshape(N_WINDOWS_SC, WINDOW_SAMPLES)

        site, hour = parse_sc_filename(fpath.name)
        stem = fpath.stem
        for wi in range(N_WINDOWS_SC):
            end_sec = (wi + 1) * WINDOW_SEC
            sc_meta_rows.append({
                "row_id": f"{stem}_{end_sec}",
                "filename": fpath.name,
                "site": site,
                "hour_utc": hour,
                "window_idx": wi,
            })

    # Inference
    emb, scores = infer_batch(all_windows)
    row_start = ci * N_WINDOWS_SC
    row_end = row_start + chunk_n * N_WINDOWS_SC
    sc_emb[row_start:row_end] = emb.astype(np.float16)
    sc_scores[row_start:row_end] = scores.astype(np.float16)

    del all_windows, emb, scores
    gc.collect()

elapsed = time.time() - t0
print(f"\\nSoundscapes done: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
print(f"  Embeddings: {sc_emb.shape}, {sc_emb.nbytes/1e6:.1f} MB")
print(f"  Scores: {sc_scores.shape}, {sc_scores.nbytes/1e6:.1f} MB")

# Save
sc_meta_df = pd.DataFrame(sc_meta_rows)
np.savez_compressed(
    OUT_DIR / "soundscape_embeddings.npz",
    embeddings=sc_emb, scores=sc_scores,
)
sc_meta_df.to_parquet(OUT_DIR / "soundscape_meta.parquet", index=False)
print(f"Saved: soundscape_embeddings.npz, soundscape_meta.parquet")

del sc_emb, sc_scores
gc.collect()"""
cells.append(code_cell("soundscapes", SC_CODE))

# ── Process train_audio ──
AUDIO_CODE = """\
# PROCESS TRAIN_AUDIO
train_df = pd.read_csv(TRAIN_CSV)
audio_files = []
for _, row in train_df.iterrows():
    fpath = AUDIO_DIR / row["filename"]
    if fpath.exists():
        audio_files.append({
            "path": fpath,
            "filename": row["filename"],
            "primary_label": str(row["primary_label"]),
        })

n_audio = len(audio_files)
print(f"Train audio files: {n_audio}")

# Process in chunks (variable length files)
ta_emb_list = []
ta_scores_list = []
ta_meta_rows = []

t0 = time.time()

for ci in tqdm(range(0, n_audio, CHUNK_AUDIO), desc="Train audio"):
    chunk = audio_files[ci:ci + CHUNK_AUDIO]

    # Read files, split into windows
    chunk_windows = []

    for fi, finfo in enumerate(chunk):
        try:
            y = read_audio(str(finfo["path"]))
            windows = split_windows(y)
        except Exception as e:
            print(f"  SKIP {finfo['filename']}: {e}")
            continue

        n_win = windows.shape[0]
        chunk_windows.append(windows)

        for wi in range(n_win):
            ta_meta_rows.append({
                "filename": finfo["filename"],
                "primary_label": finfo["primary_label"],
                "window_idx": wi,
                "n_windows": n_win,
            })

    if len(chunk_windows) == 0:
        continue

    all_windows = np.concatenate(chunk_windows, axis=0)

    # Inference
    emb, scores = infer_batch(all_windows)
    ta_emb_list.append(emb.astype(np.float16))
    ta_scores_list.append(scores.astype(np.float16))

    del all_windows, chunk_windows, emb, scores
    gc.collect()

elapsed = time.time() - t0
print(f"\\nTrain audio done: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")

# Concatenate
ta_emb = np.concatenate(ta_emb_list, axis=0)
ta_scores = np.concatenate(ta_scores_list, axis=0)
ta_meta_df = pd.DataFrame(ta_meta_rows)

print(f"  Total windows: {ta_emb.shape[0]}")
print(f"  Embeddings: {ta_emb.shape}, {ta_emb.nbytes/1e6:.1f} MB")
print(f"  Scores: {ta_scores.shape}, {ta_scores.nbytes/1e6:.1f} MB")

# Save
np.savez_compressed(
    OUT_DIR / "trainaudio_embeddings.npz",
    embeddings=ta_emb, scores=ta_scores,
)
ta_meta_df.to_parquet(OUT_DIR / "trainaudio_meta.parquet", index=False)
print(f"Saved: trainaudio_embeddings.npz, trainaudio_meta.parquet")

del ta_emb, ta_scores, ta_emb_list, ta_scores_list
gc.collect()"""
cells.append(code_cell("trainaudio", AUDIO_CODE))

# ── Summary ──
SUMMARY_CODE = """\
# SUMMARY
total_time = time.time() - START
print(f"\\n{'='*60}")
print(f"TOTAL TIME: {total_time/60:.1f} min ({total_time/3600:.2f} hr)")
print(f"{'='*60}")

print("\\nOutput files:")
for p in sorted(OUT_DIR.glob("*")):
    if p.is_file() and p.suffix in (".npz", ".parquet"):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name}: {size_mb:.1f} MB")

print("\\nVerification:")
for name in ["soundscape_embeddings.npz", "trainaudio_embeddings.npz"]:
    p = OUT_DIR / name
    if p.exists():
        arr = np.load(p)
        print(f"  {name}:")
        for k in arr.files:
            print(f"    {k}: {arr[k].shape}, dtype={arr[k].dtype}")
        arr.close()

for name in ["soundscape_meta.parquet", "trainaudio_meta.parquet"]:
    p = OUT_DIR / name
    if p.exists():
        df = pd.read_parquet(p)
        print(f"  {name}: {df.shape}, columns={list(df.columns)}")

print("\\nDone! Upload output as Kaggle Dataset for NB2/NB3.")"""
cells.append(code_cell("summary", SUMMARY_CODE))

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb1_embedding.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
