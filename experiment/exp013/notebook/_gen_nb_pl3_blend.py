"""Generate exp013 NB3: 3-way blend (NB4 + Tucker SED + R1 student SED) submission.

v8 (NB4 + Tucker rank 50:50 + Sonotype mirror, LB 0.942) に R1 student SED を追加。

Blend 構成:
- NB4 (Perch + ProtoSSM + MLP)            rank weight 0.50
- Tucker public 5-fold SED                rank weight 0.25  ← v8 では 0.50、半分に
- R1 student SED (NB2 出力 ep30)          rank weight 0.25  ← 新規

R1 student は Tucker と同アーキ + Tucker pseudo で学習 → high correlated。
両者を SED アームの ensemble として扱い、weight を共有。

Hardware: Kaggle CPU 90 min 上限内 (NB4 ~85min + Tucker ~5min + R1 ~1min)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXP010 = HERE.parent.parent / "exp010" / "notebook"
sys.path.insert(0, str(EXP010))

# v8 (exp010) の共通パーツを import
from _gen_nb4_blend import (
    code_cell, md_cell,
    INSTALL, IMPORTS, CONFIG, TAXONOMY, LOAD, MODEL, TRAIN, ONNX_TEST,
)


cells = []

cells.append(md_cell("hdr",
    "# exp013 NB3: 3-way blend (NB4 + Tucker SED + R1 student SED)\n"
    "\n"
    "**目的**: F25 R1 検証。v8 (LB 0.942) に NB2 学習の R1 student SED を追加して 3-way blend。\n"
    "\n"
    "**Blend ratio (rank space)**\n"
    "- NB4 (Perch+ProtoSSM+MLP): **0.50**\n"
    "- Tucker public 5-fold SED: **0.25** (v8 の 0.50 から半減)\n"
    "- **R1 student SED (NB2 出力)**: **0.25**\n"
    "\n"
    "Tucker と R1 は同アーキ + R1 は Tucker pseudo 経由で学習 → high correlated。\n"
    "両者を SED アームの ensemble として weight を共有。\n"
    "\n"
    "**期待 LB**\n"
    "- ケース B (中度の効果): 0.943-0.946 (+0.001 ~ +0.004)\n"
    "- ケース A (Tucker と被る): ±0\n"
    "- ケース C (悪化): -0.001 ~ -0.003"))

# v8 と同じ準備セル群 (再学習はしない、推論のみ)
cells.append(code_cell("install", INSTALL))
cells.append(code_cell("imports", IMPORTS))
cells.append(code_cell("config", CONFIG))
cells.append(code_cell("taxonomy", TAXONOMY))
cells.append(code_cell("load", LOAD))
cells.append(code_cell("model", MODEL))
cells.append(code_cell("train", TRAIN))
cells.append(code_cell("onnx-test", ONNX_TEST))

# Stage A: NB4 v7 inference + konbu17 LinearHead (v8 と同一)
NB4_INFER = r"""# === Stage A: NB4 v7 inference + konbu17 LinearHead ===
test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))
if len(test_files) == 0:
    print("No test files, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))[:8]
print(f"Test files: {len(test_files)}")

KONBU_W = None
KONBU_B = None
KONBU_MASK = None
for p in Path("/kaggle/input").rglob("head_weights_train_audio.npz"):
    _hw = np.load(p, allow_pickle=True)
    KONBU_W = _hw["W"].astype(np.float32)
    KONBU_B = _hw["b"].astype(np.float32)
    KONBU_MASK = _hw["trained_mask"].astype(np.float32)
    print(f"Konbu17 head loaded: {p}")
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

    if KONBU_W is not None:
        head_logit = emb.astype(np.float32) @ KONBU_W.T + KONBU_B
        head_logit = head_logit * KONBU_MASK.reshape(1, -1)
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

# Stage B: Tucker public 5-fold SED inference + audio_cache 流用 (v8 と同一)
SED_INFER = r"""# === Stage B: Tucker 5-fold SED ONNX ensemble (audio_cache 流用) ===
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

# 全 mel を audio_cache から事前計算 (Tucker と R1 で共用)
mel_cache = []
for raw_60s in audio_cache:
    chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
    mel_cache.append(audio_to_mel(chunks))

t0 = time.time()
probs_sed = []
for fi, mel in enumerate(mel_cache):
    p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    p_mean = p_sum / len(sed_sessions)
    p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_sed.append(p_mean)
    if (fi + 1) % 10 == 0 or fi == len(mel_cache) - 1:
        elapsed = time.time() - t0
        print(f"  Tucker SED [{fi+1}/{len(mel_cache)}] {elapsed:.0f}s")

probs_sed = np.stack(probs_sed)
print(f"Tucker SED done: {probs_sed.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("sed-infer", SED_INFER))

# Stage B': R1 student SED inference — torchaudio mel で NB2 学習と完全一致
R1_INFER = r"""# === Stage B': R1 student SED (NB2 出力) ===
# ★ 重要: NB2 (学習) は torchaudio.transforms.MelSpectrogram + AmplitudeToDB を使った。
# librosa の htk=True でも数値が微妙に違う (windowing/precision の違い)。
# torchaudio をそのまま使って NB2 と完全一致させる。
import torch
import torchaudio

# torchaudio mel transform (NB2 学習時と同設定)
_mel_t = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
    n_mels=N_MELS_SED, f_min=FMIN_SED, f_max=FMAX_SED, power=2.0,
)
_db_t = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB_SED)


def audio_to_mel_torchaudio(chunks):
    # NB2 学習と完全一致 (torchaudio HTK + AmplitudeToDB + per-spectrogram standardize)
    wav = torch.from_numpy(chunks.astype(np.float32))   # (N_WINDOWS, WINDOW_SAMP)
    s = _db_t(_mel_t(wav))                               # (N_WINDOWS, n_mels, T)
    m = s.mean(dim=(1, 2), keepdim=True)
    v = s.std(dim=(1, 2), keepdim=True)
    s = (s - m) / (v + 1e-6)
    return s.unsqueeze(1).numpy().astype(np.float32)    # (N_WINDOWS, 1, n_mels, T)


def find_r1_onnx():
    hits = sorted(Path("/kaggle/input").rglob("student_sed_fold0.onnx"))
    assert hits, "student_sed_fold0.onnx not found — attach maekeso/birdclef2026-exp013-r1-student-sed"
    return hits[0]


r1_path = find_r1_onnx()
print(f"R1 student ONNX: {r1_path}")
r1_data = r1_path.with_suffix(r1_path.suffix + ".data")
if r1_data.exists():
    print(f"  + external data: {r1_data.name} ({r1_data.stat().st_size/1024/1024:.1f} MB)")

r1_session = make_sed_session(r1_path)
r1_input_name = r1_session.get_inputs()[0].name
r1_output_names = [o.name for o in r1_session.get_outputs()]
print(f"R1 input: {r1_input_name}, outputs: {r1_output_names}")

# Sanity check on first file
mel_ta_test = audio_to_mel_torchaudio(audio_cache[0].reshape(N_WINDOWS, WINDOW_SAMPLES))
print(f"  mel_torchaudio shape: {mel_ta_test.shape}, mean={mel_ta_test.mean():.4f}, "
      f"std={mel_ta_test.std():.4f}, has_nan={np.isnan(mel_ta_test).any()}")
_test_outs = r1_session.run(None, {r1_input_name: mel_ta_test})
print(f"  R1 test out clip: shape={_test_outs[0].shape}, "
      f"mean={_test_outs[0].mean():.4f}, std={_test_outs[0].std():.4f}, "
      f"has_nan={np.isnan(_test_outs[0]).any()}, has_inf={np.isinf(_test_outs[0]).any()}")
print(f"  R1 test out frame: shape={_test_outs[1].shape}, "
      f"mean={_test_outs[1].mean():.4f}, std={_test_outs[1].std():.4f}")

t0 = time.time()
probs_r1 = []
for fi, raw_60s in enumerate(audio_cache):
    chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
    mel_ta = audio_to_mel_torchaudio(chunks)        # ★ torchaudio mel (NB2 と完全一致)
    outs = r1_session.run(None, {r1_input_name: mel_ta})
    clip_logits = outs[0]                  # (12, 234)
    framewise   = outs[1]                  # (12, T_out, 234)
    frame_max   = framewise.max(axis=1)    # (12, 234)
    p = 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    p = gaussian_filter1d(p, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_r1.append(p)
    if (fi + 1) % 20 == 0 or fi == len(audio_cache) - 1:
        elapsed = time.time() - t0
        print(f"  R1 student [{fi+1}/{len(audio_cache)}] {elapsed:.0f}s")

probs_r1 = np.stack(probs_r1)
print(f"R1 student done: {probs_r1.shape} in {time.time()-t0:.0f}s")
print(f"  Stats: mean={probs_r1.mean():.4f}, max={probs_r1.max():.4f}, "
      f"std={probs_r1.std():.6f}, has_nan={np.isnan(probs_r1).any()}")"""
cells.append(code_cell("r1-infer", R1_INFER))

# Stage C: 3-way rank blend + Sonotype mirror + submission
BLEND = r"""# === Stage C: 3-way rank blend + Sonotype mirror + submission ===
# v8 baseline (NB4 0.5 + Tucker 0.5) に R1 student を追加
# Tucker と R1 は SED arm として weight 共有 (high correlated 想定)
W_NB4 = 0.50
W_TUCKER = 0.25
W_R1     = 0.25

# Convert each branch to per-class rank
flat_010    = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_tucker = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_r1     = probs_r1.reshape(-1, probs_r1.shape[-1])
rank_010    = pd.DataFrame(flat_010).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_tucker = pd.DataFrame(flat_tucker).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_r1     = pd.DataFrame(flat_r1).rank(axis=0, pct=True).to_numpy().astype(np.float32)

# 3-way rank blend
blend_flat = W_NB4 * rank_010 + W_TUCKER * rank_tucker + W_R1 * rank_r1

# Diagnostics
import scipy.stats as scs
corr_tucker_r1, _ = scs.spearmanr(flat_tucker.flatten(), flat_r1.flatten())
corr_nb4_tucker,  _ = scs.spearmanr(flat_010.flatten(),    flat_tucker.flatten())
corr_nb4_r1,      _ = scs.spearmanr(flat_010.flatten(),    flat_r1.flatten())
print(f"  Spearman NB4-Tucker:    {corr_nb4_tucker:.4f}")
print(f"  Spearman NB4-R1:        {corr_nb4_r1:.4f}")
print(f"  Spearman Tucker-R1:     {corr_tucker_r1:.4f}  ← R1 と Tucker の重複度")

# === Sonotype mirror (v8 と同じ) ===
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
print(f"  ratios (rank-space): NB4={W_NB4} / Tucker={W_TUCKER} / R1={W_R1}")

# file_confidence_scale (v8 と同じ)
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
out_path = HERE / "nb_pl3_blend.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
