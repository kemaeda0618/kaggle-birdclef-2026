"""Generate exp010+exp012 blend submission notebook.

NB4 v7 (Perch + ProtoSSM + MLP) と exp012 (distilled SED) の logit-space blend。
両方を同じ test_soundscapes に対して推論 → 0.5*0.5 平均 → submission。

CPU 90 分制限に注意。NB4 と exp012 が両方走るので時間タイト。
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
    "# exp010 NB4 v7 + exp012 distilled SED blend\n"
    "\n"
    "両方を順番に推論 → logit-space で 50/50 ブレンド → submission。\n"
    "\n"
    "**所要時間**: NB4 (~40 min) + exp012 (~30 min) ≈ 70 min, CPU 90 min ぎりぎり"))

# Stage 1: NB4 v7 cells (install / imports / config / taxonomy / load / model / train / onnx-test)
cells.append(code_cell("install", INSTALL))
cells.append(code_cell("imports", IMPORTS))
cells.append(code_cell("config", CONFIG))
cells.append(code_cell("taxonomy", TAXONOMY))
cells.append(code_cell("load", LOAD))
cells.append(code_cell("model", MODEL))
cells.append(code_cell("train", TRAIN))
cells.append(code_cell("onnx-test", ONNX_TEST))

# Stage 2: NB4 inference (probs_exp010)
NB4_INFER = r"""# === Stage A: NB4 v7 inference ===
test_files = sorted(glob.glob(str(TEST_DIR / "*.ogg")))
if len(test_files) == 0:
    print("No test files, using train_soundscapes as fallback")
    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))[:8]
print(f"Test files: {len(test_files)}")

all_row_ids = []
probs_exp010 = []   # list of (12, 234)
audio_cache = []    # save audio for exp012 reuse

t0 = time.time()
for _m in proto_models + mlp_models:
    _m.eval()

from concurrent.futures import ThreadPoolExecutor

def _load_windows(fp):
    y = read_audio(fp, target_samples=SR * 60)
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES), y  # also return raw 60s for exp012

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
    stacked = torch.stack(ens_probs, dim=0)
    return stacked.mean(dim=0)


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
probs_exp010 = np.stack(probs_exp010)   # (N_files, 12, 234)
print(f"exp010 done: {probs_exp010.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("nb4-infer", NB4_INFER))

# Stage 3: exp012 model + inference
EXP012_LOAD = r"""# === Stage B: exp012 model load ===
import timm

EXP012_KERNEL_CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp012-train-fold0"),
]
EXP012_ROOT = next((p for p in EXP012_KERNEL_CANDIDATES if p.exists()), None)
if EXP012_ROOT is None:
    for p in Path("/kaggle/input").rglob("fold0_best_macro.pt"):
        EXP012_ROOT = p.parent; break
assert EXP012_ROOT is not None, "exp012 weights not attached"
EXP012_CKPT = next((EXP012_ROOT / n for n in
                    ["fold0_best_macro.pt", "fold0_best_ns22.pt"] if (EXP012_ROOT / n).exists()), None)
assert EXP012_CKPT is not None, f"checkpoint not in {EXP012_ROOT}"
print(f"exp012 ckpt: {EXP012_CKPT}")

# Constants must match exp012 training
E12_N_FFT, E12_HOP, E12_NMELS = 2048, 512, 256
E12_FMIN, E12_FMAX, E12_TOPDB = 20, 16000, 80
E12_CHUNK_N = SR * 5    # 160000
E12_NFRAMES = E12_CHUNK_N // E12_HOP + 1
E12_BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class BirdSEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            E12_BACKBONE, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=0.1)
        with torch.no_grad():
            d = torch.randn(1, 1, E12_NMELS, E12_NFRAMES)
            self.backbone_dim = self.backbone(d).shape[1]
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(self.backbone_dim, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.att = nn.Conv1d(512, N_CLASSES, 1, bias=True)
        self.cla = nn.Conv1d(512, N_CLASSES, 1, bias=True)

    def forward(self, x):
        h = self.backbone(x)
        h_cls = h.detach()
        h_cls = self.gem_freq(h_cls).permute(0, 2, 1)
        h_cls = self.dense(h_cls).permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise = self.cla(h_cls)
        clip = torch.sum(norm_att * framewise, dim=2)
        return clip, framewise.permute(0, 2, 1)


sed_model = BirdSEDModel().eval()
ckpt = torch.load(str(EXP012_CKPT), map_location="cpu", weights_only=False)
missing, unexpected = sed_model.load_state_dict(ckpt, strict=False)
print(f"exp012 loaded; missing={len(missing)}, unexpected={len(unexpected)}")
print(f"exp012 params={sum(p.numel() for p in sed_model.parameters())/1e6:.2f}M")"""
cells.append(code_cell("exp012-load", EXP012_LOAD))

# Stage 4: exp012 inference (reuse audio_cache)
EXP012_INFER = r"""# === Stage C: exp012 inference (reuses audio_cache from Stage A) ===
import librosa
from scipy.ndimage import convolve1d

GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

def _audio_to_mel(chunks):
    mels = []
    for i in range(chunks.shape[0]):
        S = librosa.feature.melspectrogram(
            y=chunks[i], sr=SR, n_fft=E12_N_FFT, hop_length=E12_HOP,
            n_mels=E12_NMELS, fmin=E12_FMIN, fmax=E12_FMAX, power=2.0)
        S_dB = librosa.power_to_db(S, top_db=E12_TOPDB)
        S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
        mels.append(S_dB)
    return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)


t0 = time.time()
probs_exp012 = []   # list of (12, 234)
for fi, raw_60s in enumerate(audio_cache):
    chunks = raw_60s.reshape(N_WINDOWS, E12_CHUNK_N)
    mel = _audio_to_mel(chunks)
    with torch.no_grad():
        mel_t = torch.from_numpy(mel)
        clip, framewise = sed_model(mel_t)
        frame_max = framewise.max(dim=1).values
    logits = 0.5 * clip.numpy() + 0.5 * frame_max.numpy()  # (12, 234)
    # 5-tap Gaussian smoothing in logit space
    logits_smooth = convolve1d(logits, GAUSSIAN_KERNEL, axis=0, mode="nearest")
    probs_e12 = 1.0 / (1.0 + np.exp(-np.clip(logits_smooth, -50, 50))).astype(np.float32)
    probs_exp012.append(probs_e12)

    if (fi + 1) % 10 == 0 or fi == len(audio_cache) - 1:
        elapsed = time.time() - t0
        print(f"  exp012 [{fi+1}/{len(audio_cache)}] {elapsed:.0f}s")

probs_exp012 = np.stack(probs_exp012)
print(f"exp012 done: {probs_exp012.shape} in {time.time()-t0:.0f}s")"""
cells.append(code_cell("exp012-infer", EXP012_INFER))

# Stage 5: blend + submission
BLEND = r"""# === Stage D: logit-space 50/50 blend + submission ===
BLEND_W_EXP010 = 0.7   # NB4 heavy: exp010=0.924 vs exp012=0.890, gap 0.034 → weight stronger
BLEND_W_EXP012 = 0.3

def _to_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


l010 = _to_logit(probs_exp010)
l012 = _to_logit(probs_exp012)
l_blend = BLEND_W_EXP010 * l010 + BLEND_W_EXP012 * l012
probs_blend = 1.0 / (1.0 + np.exp(-np.clip(l_blend, -50, 50)))
print(f"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")

# Apply file_confidence_scale (NB4 post-processing)
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
out_path = HERE / "nb_blend_exp012.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written: {out_path} ({len(cells)} cells)")
