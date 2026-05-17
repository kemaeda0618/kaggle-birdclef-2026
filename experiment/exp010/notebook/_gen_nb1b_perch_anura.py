"""Generate exp010 NB1b: Perch embeddings for AnuraSet v2 (non-Aves boost).

Filter AnuraSet preprocessed (3-sec clips) to BirdCLEF 2026 Amphibia 17 species.
Pad 3s -> 5s for Perch input, extract embedding (1536d) per clip.

Multi-label aware: each clip may carry multiple species; meta.parquet stores
comma-joined `primary_labels` for downstream label-matrix build.

Output:
- anura_embeddings.npz : embeddings (N, 1536) float16
- anura_meta.parquet   : stem, site, primary_labels (csv), n_species
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
    "# exp010 NB1b: Perch embeddings for AnuraSet (non-Aves retrieval boost)\n"
    "\n"
    "AnuraSet v2 preprocessed (3-sec clips) のうち BirdCLEF 2026 Amphibia 17 種を含むクリップを抽出し、\n"
    "5秒 zero-pad で Perch v2 ONNX embedding (1536d) を計算する。\n"
    "\n"
    "Multi-label 対応: 1 クリップに複数種が含まれうるため `primary_labels` を csv で保存。\n"
    "\n"
    "Output (Kaggle Dataset 化想定):\n"
    "- `anura_embeddings.npz` -- embeddings (N, 1536) float16\n"
    "- `anura_meta.parquet`   -- stem, site, clip_name, primary_labels, n_species"))

cells.append(code_cell("install",
    "import subprocess, sys, time\n"
    "START = time.time()\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                       'onnxruntime', 'huggingface_hub'])\n"
    "print('Installed onnxruntime (CPU), huggingface_hub')"))

cells.append(code_cell("imports",
    "import gc, re, warnings, json\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import soundfile as sf\n"
    "import onnxruntime as ort\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "print(f'onnxruntime {ort.__version__}')\n"
    "print(f'Providers: {ort.get_available_providers()}')"))

cells.append(code_cell("config", """\
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC

BC_BASE = Path('/kaggle/input/competitions/birdclef-2026')
if not BC_BASE.exists():
    BC_BASE = Path('/kaggle/input/birdclef-2026')
TAXONOMY_CSV = BC_BASE / 'taxonomy.csv'

ANURA_CANDIDATES = [
    Path('/kaggle/input/anuraset-preprocessed/anuraset'),
    Path('/kaggle/input/anuraset-preprocessed'),
]
ANURA_ROOT = None
for p in ANURA_CANDIDATES:
    if p.exists() and (p / 'metadata.csv').exists():
        ANURA_ROOT = p
        break
if ANURA_ROOT is None:
    for p in Path('/kaggle/input').rglob('metadata.csv'):
        if 'anura' in str(p).lower():
            ANURA_ROOT = p.parent
            break
assert ANURA_ROOT is not None, 'AnuraSet metadata.csv not found'

ANURA_AUDIO_DIR = ANURA_ROOT / 'audio'
META_CSV = ANURA_ROOT / 'metadata.csv'
OUT_DIR = Path('/kaggle/working')

BATCH_WINDOWS = 64
print(f'BC_BASE: {BC_BASE}')
print(f'ANURA_ROOT: {ANURA_ROOT}')"""))

cells.append(code_cell("onnx-session", """\
from huggingface_hub import hf_hub_download
print('Downloading Perch v2 ONNX...')
t0 = time.time()
ONNX_MODEL = hf_hub_download(repo_id='justinchuby/Perch-onnx', filename='perch_v2.onnx')
print(f'Downloaded in {time.time()-t0:.1f}s')

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4
session = ort.InferenceSession(
    ONNX_MODEL, sess_opts,
    providers=['CPUExecutionProvider'],
)
print(f'Active providers: {session.get_providers()}')

EMB_OUT = None
for o in session.get_outputs():
    if o.name == 'embedding':
        EMB_OUT = o.name
assert EMB_OUT is not None"""))

cells.append(code_cell("species-match", """\
# AnuraSet 3+3 大文字コード <-> BirdCLEF Amphibia 17 種の照合
tax = pd.read_csv(TAXONOMY_CSV)
amph = tax[tax['class_name'] == 'Amphibia'].copy()
print(f'BirdCLEF Amphibia species: {len(amph)}')

def sci_to_code(sci):
    parts = str(sci).strip().split()
    if len(parts) < 2:
        return None
    return (parts[0][:3] + parts[1][:3]).upper()

amph['anura_code'] = amph['scientific_name'].apply(sci_to_code)

meta = pd.read_csv(META_CSV)
print(f'AnuraSet metadata: {meta.shape}')

non_species_keywords = {
    'fname', 'filename', 'audio_path', 'path', 'site', 'date', 'time',
    'duration', 'start', 'end', 'id', 'split', 'label', 'sr', 'fs',
    'min_t', 'max_t', 'quality', 'week', 'unnamed', 'species_number',
    'sample_name',
}
species_cols = []
for c in meta.columns:
    cl = c.lower()
    if cl in non_species_keywords:
        continue
    if 'number' in cl or 'count' in cl:
        continue
    if meta[c].dtype.kind in ('i', 'f', 'b'):
        species_cols.append(c)
print(f'Species candidate columns: {len(species_cols)}')

code_set = set(species_cols)
matched = amph[amph['anura_code'].isin(code_set)].copy()
print(f'Matched: {len(matched)} (expected ~17)')
print(matched[['scientific_name', 'common_name', 'primary_label', 'anura_code']].to_string(index=False))

ANURA_CODE_TO_PL = dict(zip(matched['anura_code'], matched['primary_label']))
TARGET_COLS = list(matched['anura_code'])"""))

cells.append(code_cell("filter-clips", """\
# 17種 のいずれかが positive な clip だけ抽出
mask = (meta[TARGET_COLS] > 0).any(axis=1)
clips = meta[mask].copy().reset_index(drop=True)
print(f'Filtered clips: {len(clips):,} / {len(meta):,}')

for col in ('sample_name', 'fname', 'min_t', 'max_t', 'site'):
    assert col in clips.columns, f'{col} missing'

clips['clip_name'] = clips.apply(
    lambda r: f"{r['fname']}_{int(r['min_t'])}_{int(r['max_t'])}.wav",
    axis=1,
)
clips['stem'] = clips['sample_name'].astype(str).str.replace('.wav', '', regex=False)

def row_to_labels_csv(row):
    labels = [ANURA_CODE_TO_PL[c] for c in TARGET_COLS if row[c] > 0]
    return ','.join(labels)

clips['primary_labels'] = clips.apply(row_to_labels_csv, axis=1)
clips['n_species'] = clips['primary_labels'].apply(lambda s: len(s.split(',')))
print(clips[['stem', 'site', 'primary_labels', 'n_species']].head())
print(f'site value counts:\\n{clips["site"].value_counts().to_string()}')"""))

cells.append(code_cell("infer-fn", """\
def read_audio_padded(path, target_samples=WINDOW_SAMPLES):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import torchaudio, torch
        y = torch.from_numpy(y).unsqueeze(0)
        y = torchaudio.functional.resample(y, sr, SR).squeeze(0).numpy()
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        y = y[:target_samples]
    return y.astype(np.float32)


def resolve_audio(clip_name, site):
    cand = ANURA_AUDIO_DIR / site / clip_name
    if cand.exists():
        return cand
    matches = list(ANURA_AUDIO_DIR.rglob(clip_name))
    return matches[0] if matches else None


def infer_batch(windows):
    n = windows.shape[0]
    embs = []
    for i in range(0, n, BATCH_WINDOWS):
        batch = windows[i:i + BATCH_WINDOWS]
        outputs = session.run([EMB_OUT], {'inputs': batch})
        embs.append(outputs[0].astype(np.float32))
    return np.concatenate(embs, axis=0)


# Quick test
test_path = resolve_audio(clips['clip_name'].iloc[0], clips['site'].iloc[0])
print(f'Test: {test_path}')
assert test_path is not None and test_path.exists()
test_y = read_audio_padded(str(test_path))
test_emb = infer_batch(test_y[None, :])
print(f'Test emb shape: {test_emb.shape}')"""))

cells.append(code_cell("process", """\
N = len(clips)
all_emb = np.zeros((N, 1536), dtype=np.float16)
keep = np.zeros(N, dtype=bool)

CHUNK = 256
t0 = time.time()
for ci in tqdm(range(0, N, CHUNK), desc='AnuraSet'):
    sub = clips.iloc[ci:ci + CHUNK]
    batch_y = []
    batch_idx = []
    for j, row in sub.iterrows():
        src = resolve_audio(row['clip_name'], row['site'])
        if src is None:
            continue
        try:
            y = read_audio_padded(str(src))
        except Exception as e:
            print(f'  SKIP {row["clip_name"]}: {e}')
            continue
        batch_y.append(y)
        batch_idx.append(j)
    if not batch_y:
        continue
    batch_arr = np.stack(batch_y, axis=0)
    emb = infer_batch(batch_arr)
    for k, j in enumerate(batch_idx):
        all_emb[j] = emb[k].astype(np.float16)
        keep[j] = True
    if (ci // CHUNK) % 10 == 0:
        gc.collect()

elapsed = time.time() - t0
print(f'\\nDone: {keep.sum():,}/{N:,} clips in {elapsed/60:.1f} min')

clips_kept = clips[keep].reset_index(drop=True)
emb_kept = all_emb[keep]
print(f'Embeddings kept: {emb_kept.shape}')"""))

cells.append(code_cell("save", """\
np.savez_compressed(
    OUT_DIR / 'anura_embeddings.npz',
    embeddings=emb_kept,
)
meta_out = clips_kept[['stem', 'site', 'clip_name', 'primary_labels', 'n_species']].copy()
meta_out.to_parquet(OUT_DIR / 'anura_meta.parquet', index=False)

print('Saved:')
for p in sorted(OUT_DIR.glob('anura_*')):
    print(f'  {p.name}: {p.stat().st_size/1e6:.1f} MB')

# Verify
arr = np.load(OUT_DIR / 'anura_embeddings.npz')
print(f'embeddings: {arr["embeddings"].shape}, dtype={arr["embeddings"].dtype}')
arr.close()
df = pd.read_parquet(OUT_DIR / 'anura_meta.parquet')
print(f'meta: {df.shape}, columns={list(df.columns)}')
print(df.head().to_string())

print(f'\\nTotal time: {(time.time()-START)/60:.1f} min')"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = HERE / "nb1b_perch_anura.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
