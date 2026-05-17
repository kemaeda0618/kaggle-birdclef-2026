"""exp018 v1: 3-way blend = NB4 v7 + Tucker public SED + exp017 R2 (eca_nfnet_l0 SED).

v5 (NB4 + Tucker + exp014 R2 ep10, 0.40/0.45/0.15) = LB 0.945 を base に、
3rd 軸を exp014 R2 → exp017 R2 (eca_nfnet_l0、LB 0.921、gap -0.004) に swap。
- mel param は Tucker / exp014 と完全一致 (32kHz/5s/n_mels=256/hop=512/fmin=20/fmax=16000) → audio_cache 共有可
- exp017 model は PyTorch (timm eca_nfnet_l0、r2_ckpt_best_ns22.pth)
- 推論: 0.5*sigmoid(clip_logits) + 0.5*sigmoid(frame_max) → gaussian smooth (sigma=0.65)
- 3-way blend: rank-space 加重平均、比率 0.40/0.45/0.15

ratio config (各 push で書き換え、新 version になる):
  v1: (NB4, Tucker, e17) = (0.40, 0.45, 0.15)  ← v5 と同比率で e14→e17 swap、効果 isolate
  v2: (0.35, 0.40, 0.25)  ← LB +0.001+ なら e17 寄せ
  v3: (0.30, 0.40, 0.30)  ← NFNet を 2nd 軸格上げ

期待 LB: v5 (0.945) → 0.946-0.948 (e17 R2 の test gen +0.014 を blend に持ち込み)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp018/notebook
# Tucker base generator lives under exp010
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

# ============================================================================
# Cell: e17-infer (exp017 R2 eca_nfnet_l0 推論。audio_cache を再利用)
# ============================================================================
E17_INFER = r'''# === exp017 R2 (eca_nfnet_l0 + Perch distill) inference ===
# Use audio_cache built by NB4 stage (60s raw per file).
# Output probs_e17 of shape (N_files, 12, N_CLASSES) in sigmoid space, gaussian smoothed.
import timm
import torchaudio

# ---- Locate exp017 R2 ckpt (Dataset 経由) ----
E17_STATE_DIR = None
for _p in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights"),
    Path("/kaggle/input/birdclef2026-exp017-weights"),
]:
    if _p.exists() and (any(_p.rglob("r2_ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_latest*.pth"))):
        E17_STATE_DIR = _p; break
if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r2_ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
if E17_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        E17_STATE_DIR = _hit.parent; break
assert E17_STATE_DIR is not None, "exp017 ckpt not found"
print(f"exp017 state dir: {E17_STATE_DIR}")

E17_CKPT = None
for _name in [
    "r2_ckpt_best_ns22.pth",   # R2 best (val 0.9249) — preferred
    "r2_ckpt_best_macro.pth",
    "r2_ckpt_latest.pth",
    "ckpt_best_ns22.pth",      # R1 v2 fallback (val 0.9166)
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]:
    _hits = list(E17_STATE_DIR.rglob(_name))
    if _hits:
        E17_CKPT = _hits[0]; break
assert E17_CKPT is not None, f"No ckpt under {E17_STATE_DIR}"
print(f"  ckpt: {E17_CKPT.name} ({E17_CKPT.stat().st_size/1e6:.1f}MB)")

# ---- exp017 config (must match training) ----
E17_BACKBONE = "eca_nfnet_l0"
E17_N_MELS   = 256
E17_N_FFT    = 2048
E17_HOP      = 512
E17_FMIN     = 20
E17_FMAX     = 16000
E17_TRAIN_SAMPLES = SR * 5
E17_USE_DISTILL = True
E17_PERCH_DIM = 1536

# ---- Architecture (copy of exp017 BirdSEDModel) ----
class _E17MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=E17_N_FFT, hop_length=E17_HOP,
            n_mels=E17_N_MELS, f_min=E17_FMIN, f_max=E17_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _E17GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E17DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _E17SED(nn.Module):
    def __init__(self, backbone_name=E17_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E17_TRAIN_SAMPLES // E17_HOP + 1
            dummy = torch.randn(1, 1, E17_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E17GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E17_USE_DISTILL:
            self.distill_head = _E17DistillHead(self.backbone_dim, E17_PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if E17_USE_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


# ---- Load weights ----
try:
    _state = torch.load(str(E17_CKPT), map_location="cpu", weights_only=False)
except TypeError:
    _state = torch.load(str(E17_CKPT), map_location="cpu")
print(f"  ckpt epoch={_state.get('epoch')}, best_ns22={_state.get('best_ns22', float('nan')):.4f}")

e17_model = _E17SED().to(torch.device("cpu"))
e17_model.load_state_dict(_state["model_state"], strict=False)
e17_model.eval()
e17_mel_tf = _E17MelTF().to(torch.device("cpu"))
print(f"  e17 loaded: {sum(p.numel() for p in e17_model.parameters())/1e6:.1f}M params")

# ---- Inference using audio_cache (60s raw per file) ----
t0 = time.time()
probs_e17 = []  # (N_files, 12, N_CLASSES)
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)        # (12, 1, 160000)
        _mel = e17_mel_tf(_wav_t)
        # per-instance standardize (same as exp017 training/infer)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _clip, _frame = e17_model(_mel, return_framewise=True)
        _frame_max = _frame.max(dim=1).values
        # sigmoid space blend (mirrors Tucker recipe: 0.5*sigmoid(clip) + 0.5*sigmoid(frame_max))
        _p_clip   = torch.sigmoid(_clip).numpy().astype(np.float32)
        _p_frame  = torch.sigmoid(_frame_max).numpy().astype(np.float32)
        _p_mean   = 0.5 * _p_clip + 0.5 * _p_frame              # (12, N_CLASSES)
        # gaussian smooth across windows (Tucker sigma=0.65)
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e17.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e17 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e17 = np.stack(probs_e17).astype(np.float32)
print(f"e17 done: {probs_e17.shape} in {time.time()-t0:.0f}s")
'''

# ============================================================================
# New BLEND cell (3-way rank blend + Sonotype mirror + FCS).
# We replace the entire v8 BLEND cell.
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 + Tucker + exp017) + Sonotype mirror + submission ===
# exp018 v1 ratio config (per push, this is the only block to edit between versions):
BLEND_W_E10  = 0.40   # NB4 v7 weight
BLEND_W_SED  = 0.45   # Tucker public SED weight
BLEND_W_E17  = 0.15   # exp017 R2 (eca_nfnet_l0) weight
USE_SED_PRE_AVG = False   # True: Tucker と e17 を logit 空間で先に平均 → NB4 と 2-way rank blend

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6, \
    "blend weights must sum to 1.0"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])

if USE_SED_PRE_AVG:
    _eps = 1e-7
    _l_sed = np.log(np.clip(flat_sed, _eps, 1 - _eps)) - np.log1p(-np.clip(flat_sed, _eps, 1 - _eps))
    _l_e17 = np.log(np.clip(flat_e17, _eps, 1 - _eps)) - np.log1p(-np.clip(flat_e17, _eps, 1 - _eps))
    _w_sed = BLEND_W_SED / (BLEND_W_SED + BLEND_W_E17)
    _w_e17 = BLEND_W_E17 / (BLEND_W_SED + BLEND_W_E17)
    _l_sed_avg = _w_sed * _l_sed + _w_e17 * _l_e17
    flat_sed_combined = (1.0 / (1.0 + np.exp(-np.clip(_l_sed_avg, -50, 50)))).astype(np.float32)
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed_combined).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    _w_nb4 = BLEND_W_E10
    _w_sed_total = BLEND_W_SED + BLEND_W_E17
    blend_flat = _w_nb4 * rank_e10 + _w_sed_total * rank_sed
    print(f"  SED-pre-avg mode: NB4={_w_nb4:.2f} / (Tucker+e17)={_w_sed_total:.2f} "
          f"(Tucker:e17 internal = {_w_sed:.2f}:{_w_e17:.2f})")
else:
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    blend_flat = (BLEND_W_E10 * rank_e10
                  + BLEND_W_SED * rank_sed
                  + BLEND_W_E17 * rank_e17)
    print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e17={BLEND_W_E17}")

# === Sonotype mirror (公開 0.946 NB Cell 40) ===
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

# file_confidence_scale (same as v8)
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
print(submission.head())
'''

# ============================================================================
# Patch source: insert e17-infer cell append + replace BLEND constant.
# ============================================================================
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e17-infer", E17_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in tucker source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

# Inject E17_INFER constant + replace BLEND constant body.
BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E17 inference cell (exp018 3-way blend extension) ===\n'
    'E17_INFER = ' + repr(E17_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, E17_CONST_BLOCK)

# Replace the entire BLEND constant value with our 3-way (e17) version.
OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

# Patch output filename + header markdown.
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp018 v1 = NB4 v7 + Tucker SED + exp017 R2 eca_nfnet_l0 (3-way rank blend, Sonotype mirror)\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "v5 (NB4+Tucker+exp014 R2 LB 0.945) の 3rd 軸を **exp017 R2 (eca_nfnet_l0, LB 0.921, gap -0.004) に swap**。Tucker 公開 5-fold ONNX",
)

# exec under exp018/notebook so HERE / out_path resolve here.
# tucker.py imports `_gen_nb4_blend` from its own dir; expose exp010 to sys.path first.
import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
