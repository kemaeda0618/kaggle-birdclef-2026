"""Generate exp010 NB4 v7 + Tucker public 5-fold SED blend.

Replaces our self-trained exp012 fold0 (LB 0.890) with the public 5-fold ONNX
ensemble from `tuckerarrants/bc2026-distilled-sed-public` (LB ~0.917-0.93).

Mirrors mattiaangeli's 0.943 NB SED branch (Cell 11).
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
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\n"
    "\n"
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX** (`tuckerarrants/bc2026-distilled-sed-public`) を使用。\n"
    "Public NB の 0.941-0.943 帯と同じ SED ブランチ。"))

cells.append(code_cell("install", INSTALL))
cells.append(code_cell("imports", IMPORTS))
cells.append(code_cell("config", CONFIG))
cells.append(code_cell("taxonomy", TAXONOMY))
cells.append(code_cell("load", LOAD))
cells.append(code_cell("model", MODEL))
cells.append(code_cell("train", TRAIN))
cells.append(code_cell("onnx-test", ONNX_TEST))

# Stage A: NB4 v7 inference (probs_exp010, audio_cache for SED reuse)
# v4: + konbu17 LinearHead (Perch emb @ W.T + b) computed inline per file
NB4_INFER = r"""# === Stage A: NB4 v7 inference + konbu17 LinearHead ===
test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))
if len(test_files) == 0:
    print("No test files, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))[:8]
print(f"Test files: {len(test_files)}")

# Load konbu17 train_audio LinearHead weights
KONBU_W = None
KONBU_B = None
KONBU_MASK = None
for p in Path("/kaggle/input").rglob("head_weights_train_audio.npz"):
    _hw = np.load(p, allow_pickle=True)
    KONBU_W = _hw["W"].astype(np.float32)        # (234, 1536)
    KONBU_B = _hw["b"].astype(np.float32)        # (234,)
    KONBU_MASK = _hw["trained_mask"].astype(np.float32)  # (234,)
    print(f"Konbu17 head loaded: {p}")
    print(f"  W={KONBU_W.shape}, b={KONBU_B.shape}, trained={int(KONBU_MASK.sum())}/234")
    break
if KONBU_W is None:
    print("WARN: konbu17 head not attached, will skip 3rd axis")

all_row_ids = []
probs_exp010 = []
probs_konbu  = []
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

        # E19 (NB4 v11 で +0.002 確認): file-level species consistency boost
        if USE_E19 and E19_BETA > 0:
            if E19_AGG == "max":
                file_sig = l_blend.max(dim=1, keepdim=True).values
            elif E19_AGG == "mean":
                file_sig = l_blend.mean(dim=1, keepdim=True)
            elif E19_AGG == "median":
                file_sig = l_blend.median(dim=1, keepdim=True).values
            else:
                file_sig = None
            if file_sig is not None:
                l_blend = (1.0 - E19_BETA) * l_blend + E19_BETA * file_sig

        agg = torch.sigmoid(l_blend)
        probs_e10 = agg.squeeze(0).numpy()
    probs_exp010.append(probs_e10)

    # konbu17 LinearHead on Perch emb
    if KONBU_W is not None:
        head_logit = emb.astype(np.float32) @ KONBU_W.T + KONBU_B   # (12, 234)
        head_logit = head_logit * KONBU_MASK.reshape(1, -1)          # zero untrained
        head_prob = 1.0 / (1.0 + np.exp(-np.clip(head_logit, -50, 50))).astype(np.float32)
        probs_konbu.append(head_prob)
    else:
        probs_konbu.append(np.full((N_WINDOWS, N_CLASSES), 0.5, dtype=np.float32))

    for wi in range(N_WINDOWS):
        end_sec = (wi + 1) * WINDOW_SEC
        all_row_ids.append(f"{stem}_{end_sec}")

    if (fi + 1) % 10 == 0 or fi == len(test_files) - 1:
        elapsed = time.time() - t0
        print(f"  exp010 [{fi+1}/{len(test_files)}] {elapsed:.0f}s")

executor.shutdown()
probs_exp010 = np.stack(probs_exp010)
probs_konbu  = np.stack(probs_konbu)
print(f"exp010 done: {probs_exp010.shape}, konbu={probs_konbu.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("nb4-infer", NB4_INFER))

# Stage B: Tucker public 5-fold SED inference (mirroring mattiaangeli)
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
probs_sed = []   # (N_files, 12, 234)
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

# Stage C: blend + submission
BLEND = r"""# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===
# v8: revert ratio to 50:50 (best confirmed by v1/v5/v7 ablation)
# + Sonotype mirror (公開 0.946 NB Cell 40 由来、混同しやすい sonotype 内 max 伝播)
BLEND_W_EXP010 = 0.5
BLEND_W_SED    = 0.5

# Convert each model's probs to per-class rank (across all rows)
flat_010 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
rank_010 = pd.DataFrame(flat_010).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)

# Rank-space blend
blend_flat = BLEND_W_EXP010 * rank_010 + BLEND_W_SED * rank_sed   # (N_total, N_CLASSES)

# === Sonotype mirror (公開 0.946 NB Cell 40) ===
# 混同しやすい sonotype グループ内で max 伝播 → 各 species の AUC を一括底上げ
MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)
col_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
mirror_count = 0
for group in MIRROR_PAIRS:
    valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
    if len(valid_idx) >= 2:
        group_max = blend_flat[:, valid_idx].max(axis=1, keepdims=True)
        blend_flat[:, valid_idx] = group_max
        mirror_count += len(valid_idx)
print(f"  Sonotype mirror applied to {mirror_count} columns")

probs_blend = blend_flat.reshape(probs_exp010.shape)
print(f"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
print(f"  ratios (rank-space): NB4={BLEND_W_EXP010} / Tucker={BLEND_W_SED}")

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
out_path = HERE / "nb_blend_tucker.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
