"""Generate exp010 EDA NB: cross-model embedding correlation vs Perch.

Compares 5 audio embedding models (BirdNET / AST / AVES / YAMNet / CLAP) with
Perch v2 on the 792 labeled SS windows. Uses pairwise cosine similarity matrices
and Spearman correlation to measure structural agreement (no need to map species).

Output: correlation table + heatmap + saved npz of all sim matrices.
Runs on Kaggle T4 GPU with internet ON, ~30-60 min.
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
    "# exp010 EDA: Audio embedding model correlation vs Perch\n"
    "\n"
    "Compare {BirdNET, AST, AVES, YAMNet, CLAP} embeddings against Perch v2\n"
    "on the 792 labeled SS windows. Determines which model is most independent\n"
    "from Perch and worth adding as a 3rd blend axis.\n"
    "\n"
    "**Method**: pairwise cosine similarity matrix → Spearman correlation vs Perch's matrix\n"
    "**Lower correlation = more independent = better blend candidate**\n"
    "\n"
    "Runs on T4 GPU with internet ON, ~30-60 min."))

# 1. Install
INSTALL = r"""# Install dependencies
import subprocess, sys, os
PKGS = [
    "onnxruntime-gpu",
    "transformers>=4.35",
    "torchaudio",
    "librosa",
    "soundfile",
    "scipy",
    "scikit-learn",
    "fairseq2",
]
# `transformers` already on Kaggle, just upgrade if needed
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "onnxruntime-gpu", "transformers", "torchaudio",
                "librosa", "soundfile", "scipy", "scikit-learn"],
               check=False)
print("Install attempted")"""
cells.append(code_cell("install", INSTALL))

# 2. Imports / paths / config
IMPORTS = r"""# Imports + config
import os, gc, time, warnings, glob, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
from scipy.stats import spearmanr
import onnxruntime as ort

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}, device={DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}, "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

BASE = Path("/kaggle/input/competitions/birdclef-2026")
if not BASE.exists():
    BASE = Path("/kaggle/input/birdclef-2026")
TRAIN_SC_DIR = BASE / "train_soundscapes"
SC_LABELS_CSV = BASE / "train_soundscapes_labels.csv"

EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 32_000
WINDOW_SEC = 5
N_WINDOWS_PER_FILE = 12
WINDOW_SAMPLES = SR * WINDOW_SEC

# Pairwise sim matrix size: 792 x 792 if we use 792 windows.
# To keep memory/time small, optionally subsample.
MAX_WINDOWS = 792

print(f"BASE={BASE}, EMB_DIR={EMB_DIR}")"""
cells.append(code_cell("imports", IMPORTS))

# 3. Build Perch embeddings + audio cache for labeled SS
LOAD_PERCH = r"""# Load Perch embeddings (from NB1) for labeled SS, plus raw audio cache
sc_data = np.load(EMB_DIR / "soundscape_embeddings.npz")
sc_emb = sc_data["embeddings"].astype(np.float32)   # (~128k, 1536)
sc_meta = pd.read_parquet(EMB_DIR / "soundscape_meta.parquet")
print(f"All SS embeddings: {sc_emb.shape}, meta={sc_meta.shape}")

sc_labels_df = pd.read_csv(SC_LABELS_CSV)
labeled_files = set(sc_labels_df["filename"].unique())
is_labeled = sc_meta["filename"].isin(labeled_files).values

lab_emb = sc_emb[is_labeled]                                # (792, 1536)
lab_meta = sc_meta[is_labeled].reset_index(drop=True)
print(f"Labeled SS windows: {lab_emb.shape}")

# Cap to MAX_WINDOWS
if lab_emb.shape[0] > MAX_WINDOWS:
    rng = np.random.default_rng(42)
    sel = rng.choice(lab_emb.shape[0], MAX_WINDOWS, replace=False)
    sel.sort()
    lab_emb = lab_emb[sel]
    lab_meta = lab_meta.iloc[sel].reset_index(drop=True)
    print(f"Subsampled to {lab_emb.shape}")

N = lab_emb.shape[0]

# Load raw waveforms for these 66 files (cache by filename)
unique_files = list(dict.fromkeys(lab_meta["filename"].values))
print(f"Unique files to load: {len(unique_files)}")

t0 = time.time()
file_audio = {}
for fn in unique_files:
    fp = TRAIN_SC_DIR / fn
    y, sr0 = sf.read(str(fp), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    target = SR * 60
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    elif len(y) > target:
        y = y[:target]
    file_audio[fn] = y.astype(np.float32)
print(f"Audio loaded in {time.time()-t0:.0f}s")

# Build (N, WINDOW_SAMPLES) waveform array aligned with lab_emb / lab_meta
# Each row = the matching 5sec window from raw audio
def get_window(fname, window_idx):
    y = file_audio[fname]
    start = window_idx * WINDOW_SAMPLES
    return y[start:start + WINDOW_SAMPLES]

waves_32k = np.zeros((N, WINDOW_SAMPLES), dtype=np.float32)
for i in range(N):
    fn = lab_meta["filename"].iloc[i]
    wi = int(lab_meta["window_idx"].iloc[i]) if "window_idx" in lab_meta.columns else (i % N_WINDOWS_PER_FILE)
    waves_32k[i] = get_window(fn, wi)

print(f"waves_32k: {waves_32k.shape}, dtype={waves_32k.dtype}, "
      f"size={waves_32k.nbytes/1e6:.1f} MB")

# Perch embedding already done — just normalize
def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

embeddings = {"perch": l2_normalize(lab_emb)}
print(f"Perch emb shape: {embeddings['perch'].shape}")"""
cells.append(code_cell("load-perch", LOAD_PERCH))

# 4. BirdNET ONNX inference
BIRDNET = r"""# === BirdNET v2.4 ONNX ===
print("=== BirdNET ===")
BN_MODEL = None
# Broad search: any .onnx file containing 'birdnet' in path
for f in Path("/kaggle/input").rglob("*.onnx"):
    if "birdnet" in str(f).lower():
        BN_MODEL = f; break
# Fallback: any .tflite with birdnet
if BN_MODEL is None:
    for f in Path("/kaggle/input").rglob("*.tflite"):
        if "birdnet" in str(f).lower():
            BN_MODEL = f; break
# Debug listing
if BN_MODEL is None:
    print("Searched paths under /kaggle/input/ for birdnet:")
    for p in Path("/kaggle/input").iterdir():
        print(f"  {p.name}")
        if p.is_dir():
            for sub in list(p.rglob("*"))[:5]:
                print(f"    -> {sub.name}")

if BN_MODEL is None:
    print("BirdNET ONNX not found, skipping")
else:
    print(f"BirdNET model: {BN_MODEL}")
    sess = ort.InferenceSession(str(BN_MODEL),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"Providers: {sess.get_providers()}")
    inp = sess.get_inputs()[0]
    print(f"  Input: name={inp.name}, shape={inp.shape}, dtype={inp.type}")
    for o in sess.get_outputs():
        print(f"  Output: name={o.name}, shape={o.shape}, dtype={o.type}")

    # BirdNET expects 48kHz, 3-second windows
    SR_BN = 48_000
    WIN_BN = 3 * SR_BN   # 144000
    # Resample 5sec @ 32kHz → 5sec @ 48kHz, then take center 3sec
    t0 = time.time()
    waves_bn = []
    for x32 in waves_32k:
        x48 = librosa.resample(x32, orig_sr=SR, target_sr=SR_BN)
        # center 3sec
        start = max(0, (len(x48) - WIN_BN) // 2)
        x = x48[start:start + WIN_BN]
        if len(x) < WIN_BN:
            x = np.pad(x, (0, WIN_BN - len(x)))
        waves_bn.append(x)
    waves_bn = np.stack(waves_bn).astype(np.float32)
    print(f"  Resampled to 48k 3sec: {waves_bn.shape}")

    BATCH = 32
    bn_emb = []
    bn_logit = []
    for i in range(0, N, BATCH):
        batch = waves_bn[i:i + BATCH]
        outs = sess.run(None, {inp.name: batch})
        # Output shape: usually 2 outputs (embedding, logits) but order varies
        # heuristic: pick output with dim ~1024 as embedding, others = logit
        for arr in outs:
            if arr.ndim == 2:
                if arr.shape[1] in (1024, 2048):
                    bn_emb.append(arr.astype(np.float32))
                else:
                    bn_logit.append(arr.astype(np.float32))
    if bn_emb:
        bn_emb = np.concatenate(bn_emb, axis=0)
        embeddings["birdnet"] = l2_normalize(bn_emb)
        print(f"  BirdNET emb: {bn_emb.shape} ({time.time()-t0:.0f}s)")
    else:
        print("  No embedding output found, taking first output as proxy")
        # fallback: take first output regardless
        outs0 = []
        for i in range(0, N, BATCH):
            batch = waves_bn[i:i+BATCH]
            o = sess.run(None, {inp.name: batch})[0]
            if o.ndim == 2:
                outs0.append(o.astype(np.float32))
        if outs0:
            arr = np.concatenate(outs0, axis=0)
            embeddings["birdnet"] = l2_normalize(arr)
            print(f"  BirdNET first-output as emb: {arr.shape}")

    del sess
    if "waves_bn" in dir():
        del waves_bn
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()"""
cells.append(code_cell("birdnet", BIRDNET))

# 5. AST (HuggingFace)
AST_CELL = r"""# === AST (Audio Spectrogram Transformer) ===
print("=== AST ===")
try:
    from transformers import ASTFeatureExtractor, ASTModel
    AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(AST_NAME)
    model = ASTModel.from_pretrained(AST_NAME).to(DEVICE).eval()
    AST_SR = fe.sampling_rate
    print(f"  AST loaded, target SR={AST_SR}")
    BATCH = 8
    out_emb = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, BATCH):
            batch = []
            for x32 in waves_32k[i:i + BATCH]:
                if SR != AST_SR:
                    x = librosa.resample(x32, orig_sr=SR, target_sr=AST_SR)
                else:
                    x = x32
                batch.append(x)
            inputs = fe(batch, sampling_rate=AST_SR, return_tensors="pt").to(DEVICE)
            out = model(**inputs)
            # last_hidden_state: (B, T, D); pool mean over T
            emb = out.last_hidden_state.mean(dim=1)
            out_emb.append(emb.cpu().numpy().astype(np.float32))
    out_emb = np.concatenate(out_emb, axis=0)
    embeddings["ast"] = l2_normalize(out_emb)
    print(f"  AST emb: {out_emb.shape} ({time.time()-t0:.0f}s)")
    del model, fe; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    print(f"  AST failed: {type(e).__name__}: {e}")"""
cells.append(code_cell("ast", AST_CELL))

# 6. AVES (HuggingFace via fairseq2 or torch hub)
AVES = r"""# === AVES (wav2vec2-style for animal sounds) ===
print("=== AVES ===")
try:
    # AVES is on HuggingFace as `m-a-p/MERT-v1-95M` style or via earthspecies
    # Try the public earth-species AVES model
    from transformers import AutoModel, AutoFeatureExtractor
    AVES_NAME = "earthspecies/aves-base-bio"  # fallback if not available
    fallbacks = [AVES_NAME, "facebook/wav2vec2-base"]
    model = None
    fe = None
    for name in fallbacks:
        try:
            fe = AutoFeatureExtractor.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(DEVICE).eval()
            print(f"  Loaded: {name}, SR={fe.sampling_rate}")
            break
        except Exception as e:
            print(f"    {name} failed: {type(e).__name__}")
    if model is None:
        raise RuntimeError("No wav2vec2-style model available")

    AVES_SR = fe.sampling_rate
    BATCH = 8
    out_emb = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, BATCH):
            batch = []
            for x32 in waves_32k[i:i + BATCH]:
                if SR != AVES_SR:
                    x = librosa.resample(x32, orig_sr=SR, target_sr=AVES_SR)
                else:
                    x = x32
                batch.append(x)
            inputs = fe(batch, sampling_rate=AVES_SR, return_tensors="pt", padding=True).to(DEVICE)
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1)
            out_emb.append(emb.cpu().numpy().astype(np.float32))
    out_emb = np.concatenate(out_emb, axis=0)
    embeddings["aves"] = l2_normalize(out_emb)
    print(f"  AVES/wav2vec2 emb: {out_emb.shape} ({time.time()-t0:.0f}s)")
    del model, fe; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    print(f"  AVES failed: {type(e).__name__}: {e}")"""
cells.append(code_cell("aves", AVES))

# 7. YAMNet (TF Hub or ONNX)
YAMNET = r"""# === YAMNet (AudioSet) ===
print("=== YAMNet ===")
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    yam = hub.load("https://tfhub.dev/google/yamnet/1")
    YAM_SR = 16_000
    print(f"  YAMNet loaded, SR={YAM_SR}")

    BATCH_FILES = 32
    out_emb = []
    t0 = time.time()
    for i in range(N):
        x32 = waves_32k[i]
        if SR != YAM_SR:
            x = librosa.resample(x32, orig_sr=SR, target_sr=YAM_SR)
        else:
            x = x32
        scores, embeddings_yam, spec = yam(tf.constant(x.astype(np.float32)))
        # embeddings_yam shape: (n_frames, 1024); avg over time
        e = embeddings_yam.numpy().mean(axis=0)
        out_emb.append(e)
    out_emb = np.stack(out_emb, axis=0).astype(np.float32)
    embeddings["yamnet"] = l2_normalize(out_emb)
    print(f"  YAMNet emb: {out_emb.shape} ({time.time()-t0:.0f}s)")
    del yam; gc.collect()
except Exception as e:
    print(f"  YAMNet failed: {type(e).__name__}: {e}")"""
cells.append(code_cell("yamnet", YAMNET))

# 8. CLAP
CLAP = r"""# === CLAP (Contrastive Language-Audio Pretraining) ===
print("=== CLAP ===")
try:
    from transformers import ClapModel, ClapProcessor
    CLAP_NAME = "laion/clap-htsat-unfused"
    proc = ClapProcessor.from_pretrained(CLAP_NAME)
    model = ClapModel.from_pretrained(CLAP_NAME).to(DEVICE).eval()
    CLAP_SR = proc.feature_extractor.sampling_rate
    print(f"  CLAP loaded, SR={CLAP_SR}")
    BATCH = 8
    out_emb = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, BATCH):
            batch = []
            for x32 in waves_32k[i:i + BATCH]:
                if SR != CLAP_SR:
                    x = librosa.resample(x32, orig_sr=SR, target_sr=CLAP_SR)
                else:
                    x = x32
                batch.append(x)
            inputs = proc(audio=batch, sampling_rate=CLAP_SR, return_tensors="pt", padding=True).to(DEVICE)
            out = model.get_audio_features(**inputs)
            # transformers 5.0+: get_audio_features returns BaseModelOutputWithPooling
            # pooler_output is the 512d projected audio embedding
            if hasattr(out, "pooler_output"):
                emb = out.pooler_output
            elif hasattr(out, "audio_embeds"):
                emb = out.audio_embeds
            else:
                emb = out  # legacy: tensor directly
            out_emb.append(emb.cpu().numpy().astype(np.float32))
    out_emb = np.concatenate(out_emb, axis=0)
    embeddings["clap"] = l2_normalize(out_emb)
    print(f"  CLAP emb: {out_emb.shape} ({time.time()-t0:.0f}s)")
    del model, proc; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    print(f"  CLAP failed: {type(e).__name__}: {e}")"""
cells.append(code_cell("clap", CLAP))

# 9. Compute pairwise sim matrices and Spearman correlation vs Perch
ANALYZE = r"""# === Analysis: pairwise sim matrices + Spearman vs Perch ===
print(f"\n=== Available embeddings ===")
for k, v in embeddings.items():
    print(f"  {k}: shape={v.shape}")

# Compute pairwise cosine similarity matrix per model
sim_matrices = {}
for name, emb in embeddings.items():
    # emb already L2-normalized
    S = emb @ emb.T  # (N, N)
    np.fill_diagonal(S, 0.0)   # exclude self-sim
    sim_matrices[name] = S.astype(np.float32)
    print(f"  {name}: sim matrix {S.shape}, mean={S.mean():.4f}, std={S.std():.4f}")

# Spearman correlation between Perch's sim matrix and each other model's
print(f"\n=== Correlation with Perch (lower = more independent = better blend candidate) ===")
perch_flat = sim_matrices["perch"][np.triu_indices(N, k=1)]
results = []
for name, S in sim_matrices.items():
    if name == "perch":
        continue
    other_flat = S[np.triu_indices(N, k=1)]
    # Spearman is robust to scale
    rho, _ = spearmanr(perch_flat, other_flat)
    # Also Pearson for reference
    pearson = np.corrcoef(perch_flat, other_flat)[0, 1]
    results.append({"model": name, "spearman_vs_perch": rho, "pearson_vs_perch": pearson})
    print(f"  {name:<10} Spearman={rho:.4f}, Pearson={pearson:.4f}")

results_df = pd.DataFrame(results).sort_values("spearman_vs_perch")
print(f"\n=== Sorted by independence (low correlation first) ===")
print(results_df.to_string(index=False))

# Top-K nearest neighbor agreement (alternative metric)
print(f"\n=== Top-K nearest neighbor agreement (k=5) ===")
def topk_agreement(A, B, k=5):
    nnA = np.argsort(-A, axis=1)[:, :k]
    nnB = np.argsort(-B, axis=1)[:, :k]
    agree = np.mean([len(set(a) & set(b)) / k for a, b in zip(nnA, nnB)])
    return agree

S_perch = sim_matrices["perch"]
for name, S in sim_matrices.items():
    if name == "perch":
        continue
    agree = topk_agreement(S_perch, S, k=5)
    print(f"  {name:<10} top-5 NN overlap: {agree:.4f}")"""
cells.append(code_cell("analyze", ANALYZE))

# 10. Save outputs
CLASS_GAP = r"""# === F1 analysis: within-class vs between-class similarity gap ===
# Goal: distinguish "useful independent signal" from "noise"
# A model with high gap (same-class sim >> diff-class sim) carries species info.

# Build per-window label set (multi-label) for the 792 windows
print("Building per-window label sets...")
import ast as _ast
sc_labels_df_local = pd.read_csv(SC_LABELS_CSV)
row_to_labels = {}
for _, r in sc_labels_df_local.iterrows():
    fn = r["filename"]
    end_sec = int(pd.Timedelta(r["end"]).total_seconds())
    rid = f"{Path(fn).stem}_{end_sec}"
    labs = [s.strip() for s in str(r["primary_label"]).split(";") if s.strip()]
    row_to_labels[rid] = set(labs)

# Map window index → label set (using lab_meta row_id)
labels_per_window = []
for _, m in lab_meta.iterrows():
    rid = m["row_id"] if "row_id" in m else f"{Path(m['filename']).stem}_{(int(m.get('window_idx', 0)) + 1) * WINDOW_SEC}"
    labels_per_window.append(row_to_labels.get(rid, set()))
print(f"Windows with at least 1 label: {sum(1 for s in labels_per_window if s)} / {len(labels_per_window)}")

# Build same/diff masks (792x792)
n_with_lbl = sum(1 for s in labels_per_window if s)
print(f"Computing same/diff class masks (excluding label-less windows)...")
same_mask = np.zeros((N, N), dtype=bool)
diff_mask = np.zeros((N, N), dtype=bool)
for i in range(N):
    Li = labels_per_window[i]
    if not Li:
        continue
    for j in range(i + 1, N):
        Lj = labels_per_window[j]
        if not Lj:
            continue
        if Li & Lj:
            same_mask[i, j] = True
        else:
            diff_mask[i, j] = True
n_same = same_mask.sum()
n_diff = diff_mask.sum()
print(f"  same-class pairs: {n_same}")
print(f"  diff-class pairs: {n_diff}")

# Compute gap per model
print(f"\n=== Within-class vs Between-class similarity gap ===")
print(f"{'model':<10} {'same_mean':>10} {'diff_mean':>10} {'gap':>10} {'gap/std':>10}")
print("-" * 60)
gap_results = []
for name, S in sim_matrices.items():
    if n_same == 0 or n_diff == 0:
        continue
    same_vals = S[same_mask]
    diff_vals = S[diff_mask]
    s_mean = float(same_vals.mean())
    d_mean = float(diff_vals.mean())
    gap = s_mean - d_mean
    # Gap divided by overall std as effect size
    overall_std = float(np.concatenate([same_vals, diff_vals]).std())
    gap_norm = gap / max(overall_std, 1e-6)
    gap_results.append({"model": name, "same_mean": s_mean, "diff_mean": d_mean,
                         "gap": gap, "gap_normalized": gap_norm})
    print(f"{name:<10} {s_mean:>10.4f} {d_mean:>10.4f} {gap:>10.4f} {gap_norm:>10.3f}")

gap_df = pd.DataFrame(gap_results).sort_values("gap_normalized", ascending=False)
print(f"\n=== Sorted by gap (high = useful species signal) ===")
print(gap_df.to_string(index=False))

# Combined judgment: independent (low spearman) AND useful (high gap)
print(f"\n=== Combined: independence × usefulness ===")
combined = results_df.merge(
    gap_df[["model", "gap_normalized"]], on="model", how="outer"
)
# Add Perch row for reference (Perch has 0 self-correlation by definition)
perch_gap = next((g for g in gap_results if g["model"] == "perch"), None)
if perch_gap and "perch" not in combined["model"].values:
    perch_row = {"model": "perch", "spearman_vs_perch": 0.0,
                 "pearson_vs_perch": 0.0, "gap_normalized": perch_gap["gap_normalized"]}
    combined = pd.concat([combined, pd.DataFrame([perch_row])], ignore_index=True)
print(combined.to_string(index=False))
print()
print("**Best blend candidate: low spearman_vs_perch AND high gap_normalized**")
print("Useful but redundant: low gap, low spearman (= probably noise)")
print("Useful but not independent: high gap, high spearman (= duplicate of Perch)")
print("Useless: low gap (= no species signal regardless of correlation)")"""
cells.append(code_cell("class-gap", CLASS_GAP))

SAVE = r"""# === Save outputs ===
results_df.to_csv(OUT_DIR / "embed_correlation.csv", index=False)
gap_df.to_csv(OUT_DIR / "embed_class_gap.csv", index=False)
combined.to_csv(OUT_DIR / "embed_combined.csv", index=False)
np.savez_compressed(
    OUT_DIR / "sim_matrices.npz",
    **{k: v for k, v in sim_matrices.items()},
)
np.savez_compressed(
    OUT_DIR / "embeddings.npz",
    **{k: v for k, v in embeddings.items()},
)
print("Saved:")
for p in sorted(OUT_DIR.glob("*")):
    print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")

print("\n=== Recommendation ===")
print("blend candidate priority by Spearman (low first):")
print(results_df.to_string(index=False))
print()
print("- Spearman < 0.50: highly independent (大期待 +0.005-0.010)")
print("- 0.50-0.75: moderate (中期待 +0.002-0.005)")
print("- > 0.75: redundant (低期待 +0.000-0.001)")"""
cells.append(code_cell("save", SAVE))

# Assemble
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_eda_embed_corr.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
