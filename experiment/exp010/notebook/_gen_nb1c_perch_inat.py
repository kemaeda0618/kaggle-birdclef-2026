"""Generate exp010 NB1c: Perch embeddings for iNatSounds non-Aves subset.

Filter iNatSounds 2024 to BirdCLEF 2026 non-Aves species
(Amphibia 24 + Insecta 3 + Mammalia 4 = ~31 species, 1-14 samples/species).
Split each audio into 5s windows (zero-pad last partial), Perch embedding.

Output:
- inat_nonaves_embeddings.npz : embeddings (M, 1536) float16
- inat_nonaves_meta.parquet   : file_name, primary_label, class_name, window_idx, n_windows
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
    "# exp010 NB1c: Perch embeddings for iNatSounds non-Aves subset\n"
    "\n"
    "iNatSounds 2024 のうち BirdCLEF 2026 non-Aves (Amphibia/Insecta/Mammalia) と一致する種を抽出。\n"
    "Aves は除外 (BC2026 train_audio で十分カバー済み)。\n"
    "\n"
    "5秒窓に分割 → Perch v2 ONNX embedding (1536d) を計算。\n"
    "\n"
    "Output:\n"
    "- `inat_nonaves_embeddings.npz` -- embeddings (M, 1536) float16\n"
    "- `inat_nonaves_meta.parquet`   -- file_name, primary_label, class_name, window_idx, n_windows"))

cells.append(code_cell("install",
    "import subprocess, sys, time\n"
    "START = time.time()\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                       'onnxruntime', 'huggingface_hub'])\n"
    "print('Installed (CPU)')"))

cells.append(code_cell("imports",
    "import gc, re, warnings, json, tarfile, urllib.request\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import soundfile as sf\n"
    "import onnxruntime as ort\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "print(f'onnxruntime {ort.__version__}')"))

cells.append(code_cell("config", """\
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC

BC_BASE = Path('/kaggle/input/competitions/birdclef-2026')
if not BC_BASE.exists():
    BC_BASE = Path('/kaggle/input/birdclef-2026')
TAXONOMY_CSV = BC_BASE / 'taxonomy.csv'

import os as _os
print('=== /kaggle/input contents ===')
_root = Path('/kaggle/input')
if _root.exists():
    for p in sorted(_root.iterdir()):
        print(f'  {p}')
        if p.is_dir():
            for sub in sorted(p.iterdir())[:5]:
                print(f'    -> {sub.name}')
else:
    print('  /kaggle/input does not exist')

INAT_CANDIDATES = [
    Path('/kaggle/input/datasets/shadowdude/train-recordings'),
    Path('/kaggle/input/train-recordings'),
    Path('/kaggle/input/inatsounds-2024'),
    Path('/kaggle/input/inatsounds'),
]
INAT_ROOT = None
for p in INAT_CANDIDATES:
    if p.exists():
        INAT_ROOT = p
        break
# Fallback: search for any directory containing a 'train' subdir
if INAT_ROOT is None:
    for p in Path('/kaggle/input').rglob('train'):
        if p.is_dir() and not p.parent.name.startswith('.'):
            # heuristic: train subdir with .wav files inside
            try:
                first_sub = next(p.iterdir(), None)
                if first_sub is not None and first_sub.is_dir():
                    INAT_ROOT = p.parent
                    break
            except (PermissionError, StopIteration):
                continue
assert INAT_ROOT is not None, f'iNat dataset not found. /kaggle/input tree: {list(Path("/kaggle/input").rglob("*"))[:30]}'
INAT_AUDIO_DIR = INAT_ROOT / 'train' if (INAT_ROOT / 'train').exists() else INAT_ROOT

OUT_DIR = Path('/kaggle/working')
BATCH_WINDOWS = 64

print(f'INAT_ROOT: {INAT_ROOT}')
print(f'INAT_AUDIO_DIR: {INAT_AUDIO_DIR}')"""))

cells.append(code_cell("onnx-session", """\
from huggingface_hub import hf_hub_download
ONNX_MODEL = hf_hub_download(repo_id='justinchuby/Perch-onnx', filename='perch_v2.onnx')

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4
session = ort.InferenceSession(
    ONNX_MODEL, sess_opts,
    providers=['CPUExecutionProvider'],
)
print(f'Providers: {session.get_providers()}')

EMB_OUT = None
for o in session.get_outputs():
    if o.name == 'embedding':
        EMB_OUT = o.name
assert EMB_OUT is not None"""))

cells.append(code_cell("locate-train-json", """\
# train.json を探す。無ければ S3 から取得
TRAIN_JSON = None
for name in ['train.json', 'train_annotations.json']:
    for parent in [INAT_ROOT, INAT_ROOT.parent]:
        cand = parent / name
        if cand.exists():
            TRAIN_JSON = cand
            break
    if TRAIN_JSON:
        break

if TRAIN_JSON is None:
    print('train.json not found locally, downloading from S3...')
    url = 'https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/train.json.tar.gz'
    tar_path = Path('/kaggle/working/train.json.tar.gz')
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall('/kaggle/working')
    TRAIN_JSON = Path('/kaggle/working/train.json')
    tar_path.unlink()
print(f'TRAIN_JSON: {TRAIN_JSON}')"""))

cells.append(code_cell("build-mapping", """\
# BirdCLEF non-Aves <-> iNat 2024 species 照合
tax = pd.read_csv(TAXONOMY_CSV)
tax_nonaves = tax[tax['class_name'] != 'Aves'].copy()
tax_nonaves['sci_l'] = tax_nonaves['scientific_name'].str.strip().str.lower()
print(f'BirdCLEF non-Aves species: {len(tax_nonaves)}')
print(tax_nonaves['class_name'].value_counts().to_dict())

t0 = time.time()
with open(TRAIN_JSON, encoding='utf-8') as f:
    d = json.load(f)
print(f'train.json loaded in {time.time()-t0:.1f}s')

cats = pd.DataFrame(d['categories'])
anns = pd.DataFrame(d['annotations'])
audio = pd.DataFrame(d['audio'])
cats['sci_l'] = cats['name'].str.strip().str.lower()

matched = tax_nonaves.merge(
    cats[['id', 'sci_l', 'class', 'audio_dir_name']].rename(
        columns={'id': 'inat_category_id', 'class': 'inat_class'}
    ),
    on='sci_l', how='inner',
)
ann_counts = anns['category_id'].value_counts().to_dict()
matched['n_audio'] = matched['inat_category_id'].map(ann_counts).fillna(0).astype(int)
print(f'\\nMatched non-Aves species: {len(matched)}')
print(matched[['scientific_name', 'common_name', 'class_name', 'n_audio']].sort_values(
    ['class_name', 'n_audio'], ascending=[True, False]
).to_string(index=False))

# Build filelist
matched_ids = set(matched['inat_category_id'])
anns_sub = anns[anns['category_id'].isin(matched_ids)].copy()
audio_fname = dict(zip(audio['id'], audio['file_name']))
anns_sub['file_name'] = anns_sub['audio_id'].map(audio_fname)
cat_to_pl = dict(zip(matched['inat_category_id'], matched['primary_label']))
cat_to_cls = dict(zip(matched['inat_category_id'], matched['class_name']))

files_df = pd.DataFrame({
    'file_name': anns_sub['file_name'],
    'primary_label': anns_sub['category_id'].map(cat_to_pl),
    'class_name': anns_sub['category_id'].map(cat_to_cls),
}).dropna().sort_values(['class_name', 'primary_label']).reset_index(drop=True)
print(f'\\nFilelist: {len(files_df):,} audio files')"""))

cells.append(code_cell("infer-fn", """\
def read_audio_full(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import torchaudio, torch
        y = torch.from_numpy(y).unsqueeze(0)
        y = torchaudio.functional.resample(y, sr, SR).squeeze(0).numpy()
    return y.astype(np.float32)


def split_windows(y):
    if len(y) < WINDOW_SAMPLES:
        out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        out[:len(y)] = y
        return out.reshape(1, WINDOW_SAMPLES)
    n_full = len(y) // WINDOW_SAMPLES
    rem = len(y) % WINDOW_SAMPLES
    parts = [y[:n_full * WINDOW_SAMPLES].reshape(n_full, WINDOW_SAMPLES)]
    if rem > 0:
        last = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        last[:rem] = y[n_full * WINDOW_SAMPLES:]
        parts.append(last.reshape(1, WINDOW_SAMPLES))
    return np.concatenate(parts, axis=0)


def resolve_audio(fname):
    cand1 = INAT_ROOT / fname
    if cand1.exists():
        return cand1
    if fname.startswith('train/'):
        cand2 = INAT_AUDIO_DIR / fname[len('train/'):]
        if cand2.exists():
            return cand2
    cand3 = INAT_AUDIO_DIR / fname
    return cand3 if cand3.exists() else None


def infer_batch(windows):
    n = windows.shape[0]
    embs = []
    for i in range(0, n, BATCH_WINDOWS):
        batch = windows[i:i + BATCH_WINDOWS]
        outputs = session.run([EMB_OUT], {'inputs': batch})
        embs.append(outputs[0].astype(np.float32))
    return np.concatenate(embs, axis=0)"""))

cells.append(code_cell("process", """\
all_emb_chunks = []
meta_rows = []
errors = []

t0 = time.time()
for i in tqdm(range(len(files_df)), desc='iNat non-Aves'):
    row = files_df.iloc[i]
    fname = row['file_name']
    src = resolve_audio(fname)
    if src is None:
        errors.append({'file_name': fname, 'error': 'not_found'})
        continue
    try:
        y = read_audio_full(str(src))
        windows = split_windows(y)
    except Exception as e:
        errors.append({'file_name': fname, 'error': str(e)})
        continue

    emb = infer_batch(windows)
    n_win = windows.shape[0]
    all_emb_chunks.append(emb.astype(np.float16))
    for wi in range(n_win):
        meta_rows.append({
            'file_name': fname,
            'primary_label': row['primary_label'],
            'class_name': row['class_name'],
            'window_idx': wi,
            'n_windows': n_win,
        })

    if (i + 1) % 100 == 0:
        gc.collect()

elapsed = time.time() - t0
print(f'\\nDone: {len(meta_rows):,} windows from {len(all_emb_chunks):,} files in {elapsed/60:.1f} min')
print(f'Errors: {len(errors)}')
if errors[:5]:
    for e in errors[:5]:
        print(f'  {e}')"""))

cells.append(code_cell("save", """\
emb_all = np.concatenate(all_emb_chunks, axis=0) if all_emb_chunks else np.zeros((0, 1536), dtype=np.float16)
meta_df = pd.DataFrame(meta_rows)

np.savez_compressed(OUT_DIR / 'inat_nonaves_embeddings.npz', embeddings=emb_all)
meta_df.to_parquet(OUT_DIR / 'inat_nonaves_meta.parquet', index=False)

print('Saved:')
for p in sorted(OUT_DIR.glob('inat_nonaves_*')):
    print(f'  {p.name}: {p.stat().st_size/1e6:.1f} MB')

print(f'\\nemb shape: {emb_all.shape}')
print(f'meta shape: {meta_df.shape}')
if len(meta_df):
    print('\\nPer-class window counts:')
    print(meta_df.groupby('class_name').size().to_dict())
    print('\\nPer-species window counts (top 20):')
    print(meta_df.groupby(['class_name', 'primary_label']).size().sort_values(ascending=False).head(20).to_string())

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

out_path = HERE / "nb1c_perch_inat.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
