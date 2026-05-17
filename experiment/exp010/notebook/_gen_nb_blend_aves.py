"""Generate exp010 3-way blend: NB4 v7 + Tucker SED + AVES head.

Adds AVES (wav2vec2-base) as 3rd axis to existing NB4 + Tucker blend.
- Test-time AVES inference: cached audio → 16kHz resample → wav2vec2 → 5 heads → 234d
- Ratios: 45 : 45 : 10 (NB4 : Tucker : AVES) — conservative AVES weight first
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from _gen_nb4_blend import (
    code_cell, md_cell,
    INSTALL, IMPORTS, CONFIG, TAXONOMY, LOAD, MODEL, TRAIN, ONNX_TEST,
)

cells = []

cells.append(md_cell("hdr",
    "# exp010 3-way blend: NB4 v7 + Tucker SED + AVES\n"
    "\n"
    "現 best (NB4 + Tucker 50:50 = 0.939) に AVES 軸 (Spearman 0.417 / gap 0.785 で独立確認済) を追加。\n"
    "Ratio: 45 : 45 : 10 (conservative)。"))

# Override INSTALL to also install transformers
INSTALL_PLUS = INSTALL + (
    "\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                       'transformers'])\n"
    'print("transformers installed")'
)

cells.append(code_cell("install", INSTALL_PLUS))
cells.append(code_cell("imports", IMPORTS))
cells.append(code_cell("config", CONFIG))
cells.append(code_cell("taxonomy", TAXONOMY))
cells.append(code_cell("load", LOAD))
cells.append(code_cell("model", MODEL))
cells.append(code_cell("train", TRAIN))
cells.append(code_cell("onnx-test", ONNX_TEST))

# Stage A: NB4 inference + cache audio
NB4_INFER = r"""# === Stage A: NB4 v7 inference + audio cache ===
test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))
if len(test_files) == 0:
    print("No test files, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))[:8]
print(f"Test files: {len(test_files)}")

all_row_ids = []
probs_exp010 = []
audio_cache = []

t0 = time.time()
for _m in proto_models + mlp_models:
    _m.eval()

from concurrent.futures import ThreadPoolExecutor
def _load_windows(fp):
    y = read_audio(fp, target_samples=SR * 60)
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES), y

PREFETCH = 4
executor = ThreadPoolExecutor(max_workers=4)
pending = {}
for _i in range(min(PREFETCH, len(test_files))):
    pending[_i] = executor.submit(_load_windows, test_files[_i])


def _ensemble_one(models_list, emb_t, scores_t, site_t, hour_t, prior_t):
    ens_probs = []
    for s in TTA_SHIFTS:
        if s == 0:
            e_shift, sc_shift = emb_t, scores_t
        else:
            e_shift = torch.roll(emb_t, shifts=s, dims=1)
            sc_shift = torch.roll(scores_t, shifts=s, dims=1)
        for m in models_list:
            p = m(e_shift, sc_shift, site_ids=site_t, hours=hour_t,
                  prior_logit=prior_t, lambda_prior=LAMBDA_PRIOR)
            if s != 0:
                p = torch.roll(p, shifts=-s, dims=1)
            ens_probs.append(p)
    return torch.stack(ens_probs, dim=0).mean(dim=0)


for fi, fpath in enumerate(test_files):
    stem = Path(fpath).stem
    _ni = fi + PREFETCH
    if _ni < len(test_files):
        pending[_ni] = executor.submit(_load_windows, test_files[_ni])
    windows, raw_60s = pending.pop(fi).result()
    audio_cache.append(raw_60s)

    emb, scores = infer_perch_cpu(windows)
    _m = META_PAT.search(fpath)
    site_id = int(_m.group(1)) if _m else 0
    hour = int(_m.group(3)) if _m else 0
    site_t = torch.tensor([site_id], dtype=torch.long)
    hour_t = torch.tensor([hour], dtype=torch.long)
    prior_vec = compute_prior_logit(site_id, hour)
    prior_t = torch.tensor(prior_vec, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        scores_t = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        p_proto = _ensemble_one(proto_models, emb_t, scores_t, site_t, hour_t, prior_t)
        p_mlp = _ensemble_one(mlp_models, emb_t, scores_t, site_t, hour_t, prior_t)
        p_proto_c = p_proto.clamp(min=1e-7, max=1 - 1e-7)
        p_mlp_c = p_mlp.clamp(min=1e-7, max=1 - 1e-7)
        l_proto = torch.log(p_proto_c) - torch.log1p(-p_proto_c)
        l_mlp = torch.log(p_mlp_c) - torch.log1p(-p_mlp_c)
        l_blend = W_PROTO * l_proto + W_MLP * l_mlp

        if LAMBDA_RETRIEVAL > 0:
            ret_logit = compute_retrieval_logit(emb, site_id, hour)
            ret_logit_t = torch.tensor(ret_logit, dtype=torch.float32).unsqueeze(0)
            l_blend = l_blend + LAMBDA_RETRIEVAL * ret_logit_t

        if USE_TA_RETRIEVAL and ta_pool_emb_norm is not None:
            ta_ret_logit = compute_retrieval_ta_logit(emb)
            ta_ret_logit_t = torch.tensor(ta_ret_logit, dtype=torch.float32).unsqueeze(0)
            lam_t = torch.tensor(LAMBDA_TA_VEC, dtype=torch.float32)
            l_blend = l_blend + lam_t * ta_ret_logit_t

        agg = torch.sigmoid(l_blend)
        probs_e10 = agg.squeeze(0).numpy()
    probs_exp010.append(probs_e10)

    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        all_row_ids.append(f"{stem}_{end_sec}")

    if (fi + 1) % 10 == 0 or fi == len(test_files) - 1:
        elapsed = time.time() - t0
        print(f"  exp010 [{fi+1}/{len(test_files)}] {elapsed:.0f}s")

executor.shutdown()
probs_exp010 = np.stack(probs_exp010)
print(f"exp010 done: {probs_exp010.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("nb4-infer", NB4_INFER))

# Stage B: Tucker SED 5-fold ONNX
SED_INFER = r"""# === Stage B: Tucker 5-fold SED ONNX ensemble ===
import librosa
from scipy.ndimage import gaussian_filter1d

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80


def find_sed_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    assert hits, "sed_fold0.onnx not found — attach tuckerarrants/bc2026-distilled-sed-public"
    return hits[0].parent


def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


sed_dir = find_sed_dir()
sed_fold_paths = sorted(sed_dir.glob("sed_fold*.onnx"),
                        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
sed_sessions = [make_sed_session(p) for p in sed_fold_paths]
print(f"SED dir: {sed_dir}")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")

t0 = time.time()
probs_sed = []
for fi, raw_60s in enumerate(audio_cache):
    chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
    mel = audio_to_mel(chunks)
    p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    p_mean = p_sum / len(sed_sessions)
    p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_sed.append(p_mean)
    if (fi + 1) % 10 == 0 or fi == len(audio_cache) - 1:
        elapsed = time.time() - t0
        print(f"  SED [{fi+1}/{len(audio_cache)}] {elapsed:.0f}s")

probs_sed = np.stack(probs_sed)
print(f"SED done: {probs_sed.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("sed-infer", SED_INFER))

# Stage C: AVES (wav2vec2) ONNX inference
AVES_INFER = r"""# === Stage C: AVES wav2vec2 ONNX inference (~3x faster than torch CPU) ===
# Load 5 head weights from `maekeso/birdclef2026-aves-head`
AVES_HEAD_DIR = None
for cand in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-aves-head"),
    Path("/kaggle/input/birdclef2026-aves-head"),
]:
    if cand.exists():
        AVES_HEAD_DIR = cand
        break
if AVES_HEAD_DIR is None:
    for p in Path("/kaggle/input").rglob("aves_head_seed42.pt"):
        AVES_HEAD_DIR = p.parent
        break
assert AVES_HEAD_DIR is not None, "AVES head dataset not attached"
print(f"AVES_HEAD_DIR: {AVES_HEAD_DIR}")

# Load wav2vec2 ONNX from `maekeso/birdclef2026-aves-onnx-export`
# Files are inside `wav2vec2_onnx/` subdirectory in the dataset
AVES_ONNX_DIR = None
for p in Path("/kaggle/input").rglob("preprocessor_config.json"):
    if (p.parent / "model.onnx").exists():
        AVES_ONNX_DIR = p.parent
        break
assert AVES_ONNX_DIR is not None, "AVES ONNX dir (with model.onnx + preprocessor_config.json) not found"
print(f"AVES_ONNX_DIR: {AVES_ONNX_DIR}")

from transformers import AutoFeatureExtractor
aves_fe = AutoFeatureExtractor.from_pretrained(str(AVES_ONNX_DIR))
AVES_SR = aves_fe.sampling_rate
EMB_DIM_AVES = 768

# ONNX session
_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
aves_sess = ort.InferenceSession(str(AVES_ONNX_DIR / "model.onnx"),
                                  sess_options=_so, providers=["CPUExecutionProvider"])
AVES_INPUT_NAME = aves_sess.get_inputs()[0].name
print(f"AVES ONNX loaded, SR={AVES_SR}, input={AVES_INPUT_NAME}")


# Reconstruct AVESHead (same as training NB)
class AVESHead(nn.Module):
    def __init__(self, d_input=EMB_DIM_AVES, d_hidden=256, n_classes=N_CLASSES,
                 dropout=0.1, n_sites=N_SITES, meta_dim=META_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, emb, site_ids=None, hours=None,
                prior_logit=None, lambda_prior=0.0):
        x = self.input_proj(emb)
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            x = x + meta.unsqueeze(1)
        h = self.mlp(x) * self.temperature + self.bias
        if prior_logit is not None and lambda_prior > 0:
            h = h + lambda_prior * prior_logit.unsqueeze(1)
        return torch.sigmoid(h)


aves_heads = []
for seed in [42, 123, 777, 2024, 9999]:
    p = AVES_HEAD_DIR / f"aves_head_seed{seed}.pt"
    assert p.exists(), f"missing {p}"
    h = AVESHead()
    h.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    h.eval()
    aves_heads.append(h)
print(f"AVES heads: {len(aves_heads)}")

t0 = time.time()
WINDOW_SAMPLES_AVES = AVES_SR * WINDOW_SEC
FILES_PER_BATCH = 16   # ONNX is fast, larger batch OK

# Pre-resample all files in audio_cache
print("  resampling 32k → 16k...")
all_windows_aves = np.zeros((len(audio_cache), N_WINDOWS, WINDOW_SAMPLES_AVES), dtype=np.float32)
for fi, raw_60s in enumerate(audio_cache):
    raw_aves = librosa.resample(raw_60s, orig_sr=SR, target_sr=AVES_SR).astype(np.float32)
    target = AVES_SR * 60
    if len(raw_aves) < target:
        raw_aves = np.pad(raw_aves, (0, target - len(raw_aves)))
    elif len(raw_aves) > target:
        raw_aves = raw_aves[:target]
    all_windows_aves[fi] = raw_aves.reshape(N_WINDOWS, WINDOW_SAMPLES_AVES)
print(f"    resample done in {time.time()-t0:.0f}s")

# Batched wav2vec2 ONNX forward
all_emb_aves = np.zeros((len(audio_cache), N_WINDOWS, EMB_DIM_AVES), dtype=np.float32)
for bi in range(0, len(audio_cache), FILES_PER_BATCH):
    bj = min(bi + FILES_PER_BATCH, len(audio_cache))
    batch_files = all_windows_aves[bi:bj]            # (B, 12, 80000)
    flat = batch_files.reshape(-1, WINDOW_SAMPLES_AVES)  # (B*12, 80000)
    inputs = aves_fe([flat[i] for i in range(flat.shape[0])],
                      sampling_rate=AVES_SR, return_tensors="np", padding=True)
    # ONNX expects float32 input_values
    out = aves_sess.run(None, {AVES_INPUT_NAME: inputs["input_values"].astype(np.float32)})[0]
    # out: (B*12, T, 768) → mean over T
    emb_flat = out.mean(axis=1)  # (B*12, 768)
    all_emb_aves[bi:bj] = emb_flat.reshape(bj - bi, N_WINDOWS, EMB_DIM_AVES)
    if (bi // FILES_PER_BATCH) % 3 == 0 or bj == len(audio_cache):
        print(f"  AVES forward [{bj}/{len(audio_cache)}] {time.time()-t0:.0f}s")
print(f"  wav2vec2 ONNX forward done in {time.time()-t0:.0f}s")

# Apply 5 heads to all per-file embeddings
probs_aves = np.zeros((len(audio_cache), N_WINDOWS, N_CLASSES), dtype=np.float32)
emb_all_t = torch.from_numpy(all_emb_aves)  # (N_files, 12, 768) on CPU
with torch.no_grad():
    for fi in range(len(audio_cache)):
        fpath = test_files[fi]
        _m = META_PAT.search(fpath)
        site_id = int(_m.group(1)) if _m else 0
        hour = int(_m.group(3)) if _m else 0
        site_t = torch.tensor([site_id], dtype=torch.long)
        hour_t = torch.tensor([hour], dtype=torch.long)
        prior_vec = compute_prior_logit(site_id, hour)
        prior_t = torch.tensor(prior_vec, dtype=torch.float32).unsqueeze(0)
        emb_t = emb_all_t[fi:fi + 1]
        p_sum = torch.zeros(1, N_WINDOWS, N_CLASSES)
        for h in aves_heads:
            p = h(emb_t, site_ids=site_t, hours=hour_t,
                  prior_logit=prior_t, lambda_prior=LAMBDA_PRIOR)
            p_sum = p_sum + p
        probs_aves[fi] = (p_sum / len(aves_heads)).squeeze(0).numpy()

print(f"AVES done: {probs_aves.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("aves-infer", AVES_INFER))

# Stage D: 3-way blend + submission
BLEND = r"""# === Stage D: 3-way logit blend + submission ===
BLEND_W_EXP010 = 0.45    # NB4 (Perch + ProtoSSM/MLP)
BLEND_W_SED    = 0.45    # Tucker SED 5-fold
BLEND_W_AVES   = 0.10    # AVES head (conservative first try)


def _to_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


l010 = _to_logit(probs_exp010)
lsed = _to_logit(probs_sed)
laves = _to_logit(probs_aves)
l_blend = BLEND_W_EXP010 * l010 + BLEND_W_SED * lsed + BLEND_W_AVES * laves
probs_blend = 1.0 / (1.0 + np.exp(-np.clip(l_blend, -50, 50)))
print(f"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
print(f"  ratios: NB4={BLEND_W_EXP010} / Tucker={BLEND_W_SED} / AVES={BLEND_W_AVES}")

# file_confidence_scale (NB4 post-processing)
preds_array = probs_blend.reshape(-1, N_CLASSES)
_n, _c = preds_array.shape
_view = preds_array.reshape(-1, N_WINDOWS, _c)
_sorted = np.sort(_view, axis=1)
_topk_mean = _sorted[:, -FCS_TOP_K:, :].mean(axis=1, keepdims=True)
_scale = np.power(_topk_mean, FCS_POWER)
preds_array = (_view * _scale).reshape(_n, _c).astype(np.float32)

submission = pd.DataFrame(preds_array, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)
extra = our_ids - expected_ids
if extra:
    submission = submission[submission["row_id"].isin(expected_ids)]
submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

total = time.time() - START
print(f"\nSubmission: {submission.shape}, total {total:.0f}s ({total/60:.1f} min)")
print(f"Mean pred: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"Max pred:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head())"""
cells.append(code_cell("blend", BLEND))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = HERE / "nb_blend_aves.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
