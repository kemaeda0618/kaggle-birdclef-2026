"""
Build pseudo-labeling notebook for exp004.
Generates pseudo_label.ipynb from exp004 submission.ipynb cells + ONNX Perch + pseudo-label loop.

Usage: python build_pseudo_label_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SUB_NB = HERE / "submission.ipynb"

# Read submission notebook cells
with open(SUB_NB, encoding="utf-8") as f:
    sub_nb = json.load(f)

def get_cell_source(idx):
    return "".join(sub_nb["cells"][idx]["source"])

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.split("\n")}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

# Helper: fix source to list-of-lines format for ipynb
def fix_source(source):
    lines = source.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": fix_source(source),
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": fix_source(source),
    }

cells = []

# ============================================================
# Cell 0: Title
# ============================================================
cells.append(make_md_cell("""# exp004 Pseudo-Labeling Pipeline (ONNX + soundfile)

GPU ノートブック。ONNX Perch v2 で全 train_soundscapes の埋め込みを計算し、
疑似ラベリングによる ProtoSSM + MLP の学習を行う。

出力:
- `all_perch_embeddings.npz` — 全 train_soundscapes の Perch 埋め込みキャッシュ
- `pseudo_label_weights.pt` — 疑似ラベリング済み ProtoSSM 重み
- `pseudo_label_probes.pkl` — 疑似ラベリング済み MLP Probe
"""))

# ============================================================
# Cell 1: Install
# ============================================================
cells.append(make_code_cell("""# Cell 1 — Install dependencies
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'onnxruntime-gpu', 'huggingface_hub'])"""))

# ============================================================
# Cell 2: Imports
# ============================================================
cells.append(make_code_cell("""# Cell 2 — Imports
import os, gc, json, re, time, warnings, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

import onnxruntime as ort

warnings.filterwarnings("ignore")

DEVICE = "cpu"  # ProtoSSM is lightweight, runs on CPU
print("onnxruntime version:", ort.__version__)
print("CUDA available:", "CUDAExecutionProvider" in ort.get_available_providers())"""))

# ============================================================
# Cell 3: Config
# ============================================================
cells.append(make_code_cell("""# Cell 3 — Config
BASE = Path("/kaggle/input/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/competitions/birdclef-2026")

SR = 32000
DURATION = 60
FILE_SAMPLES = SR * DURATION
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160000
N_WINDOWS = DURATION // WINDOW_SEC  # 12

CFG = {
    "mode": "train",
    "verbose": True,
    "batch_files": 12,
    "proxy_reduce": "max",
    "n_splits": 5,
    # ProtoSSM
    "proto_ssm": {
        "d_model": 128,
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.1,
    },
    "proto_ssm_train": {
        "n_epochs": 80,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "patience": 15,
        "bce_weight": 1.0,
        "distill_weight": 0.3,
        "family_weight": 0.1,
    },
    # MLP
    "mlp_params": {
        "hidden_layer_sizes": (128,),
        "activation": "relu",
        "max_iter": 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": 42,
    },
    "frozen_best_probe": {
        "alpha": 0.6,
        "min_pos": 2,
    },
    # PCA
    "pca_dim": 64,
    # Pseudo-labeling
    "pseudo_rounds": 3,
    "pseudo_power": [1.0, 1.54, 1.82],  # BirdCLEF 2025 1st: 1, 1/0.65, 1/0.55
    "pseudo_threshold": 0.5,  # minimum probability to include as pseudo-label
}

# ONNX model path (will be downloaded)
ONNX_MODEL_PATH = None

# Output directory
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True)

print("BASE:", BASE)
print("Config ready.")"""))

# ============================================================
# Cell 4: Download ONNX model
# ============================================================
cells.append(make_code_cell("""# Cell 4 — Download ONNX Perch v2 model
from huggingface_hub import hf_hub_download

print("Downloading Perch v2 ONNX model from HuggingFace...")
t0 = time.time()
ONNX_MODEL_PATH = hf_hub_download(
    repo_id="justinchuby/Perch-onnx",
    filename="perch_v2.onnx",
)
print(f"Downloaded in {time.time()-t0:.1f}s: {ONNX_MODEL_PATH}")

# Load ONNX session with GPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
print("Active providers:", onnx_session.get_providers())

# Verify input/output
for inp in onnx_session.get_inputs():
    print(f"Input: {inp.name}, shape={inp.shape}")
for out in onnx_session.get_outputs():
    print(f"Output: {out.name}, shape={out.shape}")"""))

# ============================================================
# Cell 5: Load taxonomy and labels
# ============================================================
cells.append(make_code_cell("""# Cell 5 — Load taxonomy and labels
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sorted(taxonomy["primary_label"].tolist())
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

# Parse soundscape labels into multi-hot matrix
ss_files = sorted(soundscape_labels["filename"].unique().tolist())
print(f"Labeled soundscape files: {len(ss_files)}")

# Build label matrix for labeled files
all_ss_dir = BASE / "train_soundscapes"
all_ss_files = sorted([f.name for f in all_ss_dir.glob("*.ogg")])
print(f"Total soundscape files: {len(all_ss_files)}")
print(f"Unlabeled soundscape files: {len(all_ss_files) - len(ss_files)}")

# Build row-level labels for labeled soundscapes
def build_label_rows(label_df, primary_labels, label_to_idx):
    rows = []
    for _, r in label_df.iterrows():
        fn = r["filename"]
        start_sec = int(pd.Timedelta(r["start"]).total_seconds())
        end_sec = int(pd.Timedelta(r["end"]).total_seconds())
        labels = str(r["primary_label"]).split(";")
        row_id = f"{Path(fn).stem}_{end_sec}"
        y = np.zeros(len(primary_labels), dtype=np.float32)
        for lbl in labels:
            lbl = lbl.strip()
            if lbl in label_to_idx:
                y[label_to_idx[lbl]] = 1.0
        rows.append({"row_id": row_id, "filename": fn, "end_sec": end_sec, "y": y})
    return rows

label_rows = build_label_rows(soundscape_labels, PRIMARY_LABELS, label_to_idx)
print(f"Label rows: {len(label_rows)}")"""))

# ============================================================
# Cell 6: Perch label mapping and proxies
# ============================================================
# We need the labels.csv from Perch model to map species
cells.append(make_code_cell("""# Cell 6 — Perch label mapping and genus proxies
# Load Perch labels from the ONNX model's associated labels
# We need labels.csv - try to find it from Kaggle model input or download
import csv

# Try to find labels.csv from Perch model input
LABELS_CSV = None
for candidate in [
    Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1/assets/labels.csv"),
    Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1/assets/labels.csv"),
    Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/2/assets/labels.csv"),
]:
    if candidate.exists():
        LABELS_CSV = candidate
        break

if LABELS_CSV is None:
    # Download from HuggingFace
    LABELS_CSV = hf_hub_download(
        repo_id="google/bird-vocalization-classifier",
        filename="assets/labels.csv",
        revision="main",
    )
    # If that fails, we'll handle it
    if LABELS_CSV is None:
        raise FileNotFoundError("Cannot find Perch labels.csv")

print(f"Perch labels: {LABELS_CSV}")

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
    on="scientific_name_lookup",
    how="left"
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)
label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

# Genus proxies for unmapped species
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
].copy()

def get_genus_hits(scientific_name):
    genus = str(scientific_name).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\\s", na=False)
    ].copy()
    return genus, hits

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    target = row["primary_label"]
    sci = row["scientific_name"]
    genus, hits = get_genus_hits(sci)
    if len(hits) > 0:
        proxy_map[target] = {
            "genus": genus,
            "bc_indices": hits["bc_index"].astype(int).tolist(),
        }

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
SELECTED_PROXY_TARGETS = sorted([
    t for t in proxy_map.keys()
    if CLASS_NAME_MAP.get(t) in PROXY_TAXA
])

selected_proxy_pos_to_bc = {
    label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
    for target in SELECTED_PROXY_TARGETS
}

print(f"Mapped: {MAPPED_MASK.sum()}/{N_CLASSES}, Proxies: {len(SELECTED_PROXY_TARGETS)}")"""))

# ============================================================
# Cell 7: ONNX Perch inference function
# ============================================================
cells.append(make_code_cell("""# Cell 7 — ONNX Perch inference (soundfile + batched)

def parse_soundscape_filename(fname):
    parts = Path(fname).stem.split("_")
    site = [p for p in parts if p.startswith("S")][0] if any(p.startswith("S") for p in parts) else "UNK"
    # Extract hour from time part (last element, HHMMSS format)
    time_str = parts[-1] if len(parts) >= 6 else "000000"
    hour_utc = int(time_str[:2]) if len(time_str) >= 2 else 0
    return {"site": site, "hour_utc": hour_utc}

def read_soundscape_sf(path):
    \"\"\"Read 60s soundscape using soundfile (fast, no resampling).\"\"\"
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y

def infer_perch_onnx(paths, batch_files=12, verbose=True):
    \"\"\"Compute Perch embeddings + logits for soundscape files using ONNX.\"\"\"
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files,
                       desc="ONNX Perch")

    for start in iterator:
        batch_paths = paths[start:start + batch_files]
        batch_n = len(batch_paths)

        x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_row_start = write_row
        x_pos = 0

        for path in batch_paths:
            y = read_soundscape_sf(path)
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

            meta = parse_soundscape_filename(path.name)
            stem = path.stem

            row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row:write_row + N_WINDOWS] = path.name
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

            x_pos += N_WINDOWS
            write_row += N_WINDOWS

        # ONNX inference
        outputs = onnx_session.run(None, {"inputs": x})
        output_names = [o.name for o in onnx_session.get_outputs()]
        out_dict = dict(zip(output_names, outputs))

        logits = out_dict["label"].astype(np.float32)
        emb = out_dict["embedding"].astype(np.float32)

        scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        embeddings[batch_row_start:write_row] = emb

        # Genus proxies
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            scores[batch_row_start:write_row, pos] = sub.max(axis=1).astype(np.float32)

        del x, outputs, logits, emb
        gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings

print("ONNX Perch inference function ready.")"""))

# ============================================================
# Cell 8: Compute ALL embeddings
# ============================================================
cells.append(make_code_cell("""# Cell 8 — Compute Perch embeddings for ALL train_soundscapes

CACHE_PATH = OUT_DIR / "all_perch_embeddings.npz"
CACHE_META_PATH = OUT_DIR / "all_perch_meta.parquet"

if CACHE_PATH.exists() and CACHE_META_PATH.exists():
    print("Loading cached embeddings...")
    arr = np.load(CACHE_PATH)
    all_scores = arr["scores"]
    all_embeddings = arr["embeddings"]
    all_meta = pd.read_parquet(CACHE_META_PATH)
else:
    all_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))
    print(f"Computing embeddings for {len(all_paths)} files...")

    t0 = time.time()
    all_meta, all_scores, all_embeddings = infer_perch_onnx(
        all_paths, batch_files=CFG["batch_files"], verbose=True
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")

    # Save cache
    np.savez_compressed(CACHE_PATH, scores=all_scores, embeddings=all_embeddings)
    all_meta.to_parquet(CACHE_META_PATH, index=False)
    print(f"Saved cache: {CACHE_PATH} ({CACHE_PATH.stat().st_size/1e6:.1f} MB)")

print(f"All meta: {all_meta.shape}")
print(f"All scores: {all_scores.shape}")
print(f"All embeddings: {all_embeddings.shape}")"""))

# ============================================================
# Cell 9: Split labeled vs unlabeled
# ============================================================
cells.append(make_code_cell("""# Cell 9 — Split labeled vs unlabeled data

labeled_filenames = set(ss_files)
is_labeled = all_meta["filename"].isin(labeled_filenames).values

# Labeled data
meta_labeled = all_meta[is_labeled].reset_index(drop=True)
scores_labeled = all_scores[is_labeled]
emb_labeled = all_embeddings[is_labeled]

# Build Y_labeled (ground truth labels)
label_row_map = {}
for row in label_rows:
    label_row_map[row["row_id"]] = row["y"]

Y_labeled = np.zeros((len(meta_labeled), N_CLASSES), dtype=np.float32)
for i, row_id in enumerate(meta_labeled["row_id"]):
    if row_id in label_row_map:
        Y_labeled[i] = label_row_map[row_id]

# Unlabeled data
meta_unlabeled = all_meta[~is_labeled].reset_index(drop=True)
scores_unlabeled = all_scores[~is_labeled]
emb_unlabeled = all_embeddings[~is_labeled]

print(f"Labeled: {len(meta_labeled)} windows ({len(labeled_filenames)} files)")
print(f"Unlabeled: {len(meta_unlabeled)} windows ({meta_unlabeled['filename'].nunique()} files)")
print(f"Active classes in labeled: {int((Y_labeled.sum(axis=0) > 0).sum())}")"""))

# ============================================================
# Cell 10: Helper functions (from exp004)
# ============================================================
cells.append(make_code_cell(get_cell_source(6)))  # metrics/helpers
cells.append(make_code_cell(get_cell_source(9)))  # prior tables
cells.append(make_code_cell(get_cell_source(11)))  # classwise probe helpers

# ============================================================
# Cell 13: ProtoSSM Architecture (from exp004)
# ============================================================
cells.append(make_code_cell(get_cell_source(12)))  # ProtoSSM class

# ============================================================
# Cell 14: ProtoSSM Training (from exp004)
# ============================================================
cells.append(make_code_cell(get_cell_source(13)))  # training loop

# ============================================================
# Cell 15: Pseudo-labeling loop
# ============================================================
cells.append(make_code_cell("""# Cell 15 — Pseudo-labeling loop

def reshape_to_files(arr, meta_df):
    \"\"\"Reshape flat (n_windows_total, D) to (n_files, N_WINDOWS, D).\"\"\"
    fnames = meta_df["filename"].values
    unique_files = list(dict.fromkeys(fnames))  # preserve order
    n_files = len(unique_files)
    D = arr.shape[1] if arr.ndim == 2 else arr.shape[1:]

    if arr.ndim == 2:
        out = np.zeros((n_files, N_WINDOWS, arr.shape[1]), dtype=arr.dtype)
    else:
        out = np.zeros((n_files, N_WINDOWS) + arr.shape[1:], dtype=arr.dtype)

    file_to_idx = {f: i for i, f in enumerate(unique_files)}
    window_counters = np.zeros(n_files, dtype=int)
    for row_idx, fn in enumerate(fnames):
        fi = file_to_idx[fn]
        wi = window_counters[fi]
        if wi < N_WINDOWS:
            out[fi, wi] = arr[row_idx]
            window_counters[fi] += 1

    return out, unique_files

def train_full_pipeline(emb_data, scores_data, Y_data, meta_data, cfg, round_name=""):
    \"\"\"Train ProtoSSM + MLP probes on given data. Returns model, probes, and fitted transformers.\"\"\"
    print(f"\\n{'='*60}")
    print(f"Training pipeline: {round_name}")
    print(f"Data: {len(meta_data)} windows, {meta_data['filename'].nunique()} files")
    print(f"{'='*60}")

    # Fit PCA on embeddings
    emb_scaler = StandardScaler()
    emb_scaled = emb_scaler.fit_transform(emb_data)
    emb_pca = PCA(n_components=min(cfg["pca_dim"], emb_data.shape[1]), random_state=42)
    Z = emb_pca.fit_transform(emb_scaled).astype(np.float32)

    # Reshape to file-level
    emb_files, file_list = reshape_to_files(emb_data, meta_data)
    logits_files, _ = reshape_to_files(scores_data, meta_data)
    labels_files, _ = reshape_to_files(Y_data, meta_data)

    # Build family mapping
    n_families, class_to_family, fam_to_idx = build_family_mapping(taxonomy, PRIMARY_LABELS)
    file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
    for fi in range(len(file_list)):
        active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
        for ci in active_classes:
            file_families[fi, class_to_family[ci]] = 1.0

    # ProtoSSM
    ssm_cfg = cfg["proto_ssm"]
    model = ProtoSSM(
        d_input=emb_data.shape[1],
        d_model=ssm_cfg["d_model"],
        d_state=ssm_cfg["d_state"],
        n_ssm_layers=ssm_cfg["n_ssm_layers"],
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        dropout=ssm_cfg["dropout"],
    ).to(DEVICE)

    emb_flat_tensor = torch.tensor(emb_data, dtype=torch.float32)
    labels_flat_tensor = torch.tensor(Y_data, dtype=torch.float32)
    model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
    model.init_family_head(n_families, class_to_family)

    t0 = time.time()
    model, history = train_proto_ssm(
        model,
        emb_files, logits_files, labels_files.astype(np.float32),
        file_families=file_families,
        cfg=cfg["proto_ssm_train"],
        verbose=True,
    )
    print(f"ProtoSSM training: {time.time()-t0:.1f}s")

    # OOF base/prior for MLP features
    sc_meta = meta_data.copy()
    groups = sc_meta["filename"].to_numpy()

    # Fit prior tables
    prior_tables = fit_prior_tables(sc_meta.reset_index(drop=True), Y_data)

    # Build base scores with prior fusion
    base_scores, prior_scores = fuse_scores_with_tables(
        scores_data,
        sites=sc_meta["site"].to_numpy(),
        hours=sc_meta["hour_utc"].to_numpy(),
        tables=prior_tables,
    )

    # Train MLP probes
    min_pos = int(cfg["frozen_best_probe"]["min_pos"])
    PROBE_CLASS_IDX = np.where(Y_data.sum(axis=0) >= min_pos)[0].astype(np.int32)

    probe_models = {}
    for cls_idx in tqdm(PROBE_CLASS_IDX, desc="MLP probes", disable=not cfg["verbose"]):
        y = Y_data[:, cls_idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        X_cls = build_class_features(
            Z,
            raw_col=scores_data[:, cls_idx],
            prior_col=prior_scores[:, cls_idx],
            base_col=base_scores[:, cls_idx],
        )

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > n_pos:
            repeat = max(1, n_neg // n_pos)
            pos_idx = np.where(y == 1)[0]
            X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
            y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
        else:
            X_bal, y_bal = X_cls, y

        clf = MLPClassifier(**cfg["mlp_params"])
        clf.fit(X_bal, y_bal)
        probe_models[cls_idx] = clf

    print(f"MLP probes trained: {len(probe_models)}")

    return {
        "model": model,
        "probes": probe_models,
        "emb_scaler": emb_scaler,
        "emb_pca": emb_pca,
        "prior_tables": prior_tables,
    }

def predict_scores(pipeline, emb_data, scores_data, meta_data):
    \"\"\"Run full prediction pipeline (ProtoSSM + MLP + ensemble).\"\"\"
    model = pipeline["model"]
    probes = pipeline["probes"]
    emb_scaler = pipeline["emb_scaler"]
    emb_pca = pipeline["emb_pca"]
    prior_tables = pipeline["prior_tables"]

    # ProtoSSM inference
    emb_files, file_list = reshape_to_files(emb_data, meta_data)
    logits_files, _ = reshape_to_files(scores_data, meta_data)

    model.eval()
    with torch.no_grad():
        proto_out, _, _ = model(
            torch.tensor(emb_files, dtype=torch.float32),
            torch.tensor(logits_files, dtype=torch.float32),
        )
        proto_scores_flat = proto_out.numpy().reshape(-1, N_CLASSES).astype(np.float32)

    # Prior fusion
    base_scores, prior_scores = fuse_scores_with_tables(
        scores_data,
        sites=meta_data["site"].to_numpy(),
        hours=meta_data["hour_utc"].to_numpy(),
        tables=prior_tables,
    )

    # MLP probe scores
    emb_scaled = emb_scaler.transform(emb_data)
    Z = emb_pca.transform(emb_scaled).astype(np.float32)

    mlp_scores = base_scores.copy()
    alpha = float(CFG["frozen_best_probe"]["alpha"])
    for cls_idx, clf in probes.items():
        X_cls = build_class_features(
            Z,
            raw_col=scores_data[:, cls_idx],
            prior_col=prior_scores[:, cls_idx],
            base_col=base_scores[:, cls_idx],
        )
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        else:
            pred = clf.decision_function(X_cls).astype(np.float32)
        mlp_scores[:, cls_idx] = (1.0 - alpha) * base_scores[:, cls_idx] + alpha * pred

    # Ensemble
    final_scores = 0.5 * proto_scores_flat + 0.5 * mlp_scores
    return final_scores.astype(np.float32)

print("Pseudo-labeling functions ready.")"""))

# ============================================================
# Cell 16: Run pseudo-labeling
# ============================================================
cells.append(make_code_cell("""# Cell 16 — Run pseudo-labeling rounds

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# Round 0: Train on labeled data only
pipeline = train_full_pipeline(
    emb_labeled, scores_labeled, Y_labeled, meta_labeled,
    cfg=CFG, round_name="Round 0 (labeled only)"
)

for round_idx in range(CFG["pseudo_rounds"]):
    power = CFG["pseudo_power"][round_idx] if round_idx < len(CFG["pseudo_power"]) else CFG["pseudo_power"][-1]
    print(f"\\n{'#'*60}")
    print(f"# PSEUDO-LABELING ROUND {round_idx + 1} (power={power:.2f})")
    print(f"{'#'*60}")

    # 1. Predict on unlabeled data
    t0 = time.time()
    unlabeled_scores = predict_scores(
        pipeline, emb_unlabeled, scores_unlabeled, meta_unlabeled
    )
    print(f"Prediction on {len(meta_unlabeled)} unlabeled windows: {time.time()-t0:.1f}s")

    # 2. Convert to probabilities and apply Power Transform
    unlabeled_probs = sigmoid(unlabeled_scores)
    unlabeled_pseudo = unlabeled_probs ** power
    print(f"Power transform (power={power:.2f}): max={unlabeled_pseudo.max():.3f}, "
          f"mean={unlabeled_pseudo.mean():.5f}")

    # 3. Create pseudo-labels (soft labels above threshold)
    threshold = CFG["pseudo_threshold"]
    mask = unlabeled_pseudo.max(axis=1) > threshold
    n_pseudo = mask.sum()
    print(f"Pseudo-labeled windows (max prob > {threshold}): {n_pseudo}/{len(mask)} "
          f"({100*n_pseudo/len(mask):.1f}%)")

    if n_pseudo == 0:
        print("No pseudo-labels generated. Stopping.")
        break

    # 4. Combine labeled + pseudo-labeled data
    pseudo_meta = meta_unlabeled[mask].reset_index(drop=True)
    pseudo_emb = emb_unlabeled[mask]
    pseudo_scores = scores_unlabeled[mask]
    pseudo_Y = unlabeled_pseudo[mask]  # soft pseudo-labels

    combined_meta = pd.concat([meta_labeled, pseudo_meta], ignore_index=True)
    combined_emb = np.vstack([emb_labeled, pseudo_emb])
    combined_scores = np.vstack([scores_labeled, pseudo_scores])
    combined_Y = np.vstack([Y_labeled, pseudo_Y])

    print(f"Combined data: {len(combined_meta)} windows "
          f"({len(meta_labeled)} labeled + {n_pseudo} pseudo)")

    # 5. Retrain
    pipeline = train_full_pipeline(
        combined_emb, combined_scores, combined_Y, combined_meta,
        cfg=CFG, round_name=f"Round {round_idx + 1} (labeled + pseudo)"
    )

    # 6. Evaluate on labeled data (sanity check)
    labeled_preds = predict_scores(
        pipeline, emb_labeled, scores_labeled, meta_labeled
    )
    labeled_probs = sigmoid(labeled_preds)
    auc = macro_auc_skip_empty(Y_labeled, labeled_probs)
    print(f"\\nLabeled data AUC after round {round_idx + 1}: {auc:.4f}")

print("\\nPseudo-labeling complete.")"""))

# ============================================================
# Cell 17: Save results
# ============================================================
cells.append(make_code_cell("""# Cell 17 — Save trained weights and models

# Save ProtoSSM weights
proto_path = OUT_DIR / "pseudo_label_proto_ssm.pt"
torch.save({
    "model_state_dict": pipeline["model"].state_dict(),
    "config": CFG["proto_ssm"],
    "n_classes": N_CLASSES,
    "n_windows": N_WINDOWS,
}, proto_path)
print(f"Saved ProtoSSM: {proto_path}")

# Save MLP probes
probe_path = OUT_DIR / "pseudo_label_probes.pkl"
with open(probe_path, "wb") as f:
    pickle.dump(pipeline["probes"], f)
print(f"Saved MLP probes: {probe_path}")

# Save transformers (scaler, PCA, prior tables)
aux_path = OUT_DIR / "pseudo_label_aux.pkl"
with open(aux_path, "wb") as f:
    pickle.dump({
        "emb_scaler": pipeline["emb_scaler"],
        "emb_pca": pipeline["emb_pca"],
        "prior_tables": pipeline["prior_tables"],
        "cfg": CFG,
    }, f)
print(f"Saved auxiliary: {aux_path}")

# Save embeddings cache (for future use)
print(f"\\nEmbeddings cache: {CACHE_PATH}")
print(f"\\nAll output files:")
for p in sorted(OUT_DIR.glob("*")):
    if p.is_file():
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")

print("\\nDone! Download output files and upload as Kaggle Dataset for submission notebook.")"""))

# ============================================================
# Build notebook
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = HERE / "pseudo_label.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Built: {out_path}")
print(f"Cells: {len(cells)}")
print(f"Size: {out_path.stat().st_size:,} bytes")
