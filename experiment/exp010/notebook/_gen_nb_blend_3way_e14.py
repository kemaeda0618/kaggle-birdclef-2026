"""v12: 3-way blend = NB4 v7 (Perch retrieval/MLP) + Tucker public SED + exp014 R1 (HGNet SED).

diversity 増を狙う 3 軸目に exp014 R1 fold0 (LB 0.897、HGNetV2-B0 + Perch distill) を投入。
- mel param は Tucker と完全一致 (32kHz/5s/n_mels=256/hop=512/fmin=20/fmax=16000) なので audio_cache 共有可
- exp014 model は PyTorch (timm HGNetV2-B0、ckpt_best_ns22.pth)
- 推論: 0.5*sigmoid(clip_logits) + 0.5*sigmoid(frame_max) → gaussian smooth (Tucker と同じ sigma=0.65)
- 3-way blend: rank-space 加重平均

ratio config (各 push で書き換え、新 version になる):
  v1: (NB4, Tucker, e14) = (0.45, 0.45, 0.10)  ← デフォルト、e14 控えめスタート
  v2: (0.50, 0.35, 0.15)
  v3: SED-pre-avg variant (Tucker と e14 を logit-space で平均 → NB4 と rank blend)
  v4: (0.40, 0.40, 0.20)
  v5: 中間結果を見て fine-tune

期待 LB: v8 (0.942) → +0.001-0.004 (diversity 追加効果)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# ============================================================================
# Cell: e14-infer (exp014 R1 fold0 推論。audio_cache を再利用)
# ============================================================================
E14_INFER = r'''# === exp014 R2 ep10 (HGNetV2-B0 + Perch distill) inference ===
# Use audio_cache built by NB4 stage (60s raw per file).
# Output probs_e14 of shape (N_files, 12, N_CLASSES) in sigmoid space, gaussian smoothed.
import timm
import torchaudio

# ---- Locate exp014 R2 ckpt (R1 as fallback) ----
E14_STATE_DIR = None
for _p in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp014-train-r2"),
    Path("/kaggle/input/birdclef2026-exp014-train-r2"),
    Path("/kaggle/input/datasets/maekeso/exp014-state-r2"),
    Path("/kaggle/input/exp014-state-r2"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp014-train"),
    Path("/kaggle/input/birdclef2026-exp014-train"),
    Path("/kaggle/input/datasets/maekeso/exp014-state"),
    Path("/kaggle/input/exp014-state"),
]:
    if _p.exists() and (any(_p.rglob("ckpt_best_ns22.pth")) or any(_p.rglob("ckpt_latest*.pth"))):
        E14_STATE_DIR = _p; break
if E14_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        E14_STATE_DIR = _hit.parent; break
assert E14_STATE_DIR is not None, "exp014 ckpt not found"
print(f"exp014 state dir: {E14_STATE_DIR}")

E14_CKPT = None
for _name in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth"]:
    _hits = list(E14_STATE_DIR.rglob(_name))
    if _hits:
        E14_CKPT = _hits[0]; break
assert E14_CKPT is not None, f"No ckpt under {E14_STATE_DIR}"
print(f"  ckpt: {E14_CKPT.name} ({E14_CKPT.stat().st_size/1e6:.1f}MB)")

# ---- exp014 config (must match training) ----
E14_BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
E14_N_MELS   = 256
E14_N_FFT    = 2048
E14_HOP      = 512
E14_FMIN     = 20
E14_FMAX     = 16000
E14_TRAIN_SAMPLES = SR * 5
E14_USE_DISTILL = True
E14_PERCH_DIM = 1536

# ---- Architecture (copy of exp014 BirdSEDModel) ----
class _E14MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=E14_N_FFT, hop_length=E14_HOP,
            n_mels=E14_N_MELS, f_min=E14_FMIN, f_max=E14_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _E14GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E14DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _E14SED(nn.Module):
    def __init__(self, backbone_name=E14_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E14_TRAIN_SAMPLES // E14_HOP + 1
            dummy = torch.randn(1, 1, E14_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E14GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E14_USE_DISTILL:
            self.distill_head = _E14DistillHead(self.backbone_dim, E14_PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if E14_USE_DISTILL else h
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
    _state = torch.load(str(E14_CKPT), map_location="cpu", weights_only=False)
except TypeError:
    _state = torch.load(str(E14_CKPT), map_location="cpu")
print(f"  ckpt epoch={_state.get('epoch')}, best_ns22={_state.get('best_ns22', float('nan')):.4f}")

e14_model = _E14SED().to(torch.device("cpu"))
e14_model.load_state_dict(_state["model_state"], strict=False)
e14_model.eval()
e14_mel_tf = _E14MelTF().to(torch.device("cpu"))
print(f"  e14 loaded: {sum(p.numel() for p in e14_model.parameters())/1e6:.1f}M params")

# ---- Inference using audio_cache (60s raw per file) ----
t0 = time.time()
probs_e14 = []  # (N_files, 12, N_CLASSES)
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)        # (12, 1, 160000)
        _mel = e14_mel_tf(_wav_t)
        # per-instance standardize (same as exp014 training/infer)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _clip, _frame = e14_model(_mel, return_framewise=True)
        _frame_max = _frame.max(dim=1).values
        # sigmoid space blend (mirrors Tucker recipe: 0.5*sigmoid(clip) + 0.5*sigmoid(frame_max))
        _p_clip   = torch.sigmoid(_clip).numpy().astype(np.float32)
        _p_frame  = torch.sigmoid(_frame_max).numpy().astype(np.float32)
        _p_mean   = 0.5 * _p_clip + 0.5 * _p_frame              # (12, N_CLASSES)
        # gaussian smooth across windows (Tucker sigma=0.65)
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e14.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e14 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e14 = np.stack(probs_e14).astype(np.float32)
print(f"e14 done: {probs_e14.shape} in {time.time()-t0:.0f}s")
'''

# ============================================================================
# New BLEND cell (3-way rank blend + Sonotype mirror + FCS).
# We replace the entire v8 BLEND cell.
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 + Tucker + exp014) + Sonotype mirror + submission ===
# v12 ratio config (per push, this is the only block to edit between versions):
BLEND_W_E10  = 0.40   # NB4 v7 weight
BLEND_W_SED  = 0.45   # Tucker public SED weight
BLEND_W_E14  = 0.15   # exp014 R2 ep10 weight
USE_SED_PRE_AVG = False   # True: Tucker と e14 を logit 空間で先に平均 → NB4 と 2-way rank blend

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e14 = probs_e14.reshape(-1, probs_e14.shape[-1])

if USE_SED_PRE_AVG:
    # SED branch: weighted avg in logit space, then rank
    _eps = 1e-7
    _l_sed = np.log(np.clip(flat_sed, _eps, 1 - _eps)) - np.log1p(-np.clip(flat_sed, _eps, 1 - _eps))
    _l_e14 = np.log(np.clip(flat_e14, _eps, 1 - _eps)) - np.log1p(-np.clip(flat_e14, _eps, 1 - _eps))
    _w_sed = BLEND_W_SED / (BLEND_W_SED + BLEND_W_E14)
    _w_e14 = BLEND_W_E14 / (BLEND_W_SED + BLEND_W_E14)
    _l_sed_avg = _w_sed * _l_sed + _w_e14 * _l_e14
    flat_sed_combined = (1.0 / (1.0 + np.exp(-np.clip(_l_sed_avg, -50, 50)))).astype(np.float32)
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed_combined).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    _w_nb4 = BLEND_W_E10
    _w_sed_total = BLEND_W_SED + BLEND_W_E14
    blend_flat = _w_nb4 * rank_e10 + _w_sed_total * rank_sed
    print(f"  SED-pre-avg mode: NB4={_w_nb4:.2f} / (Tucker+e14)={_w_sed_total:.2f} "
          f"(Tucker:e14 internal = {_w_sed:.2f}:{_w_e14:.2f})")
else:
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_e14 = pd.DataFrame(flat_e14).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    blend_flat = (BLEND_W_E10 * rank_e10
                  + BLEND_W_SED * rank_sed
                  + BLEND_W_E14 * rank_e14)
    print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e14={BLEND_W_E14}")

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
# Patch source: insert e14-infer cell append + replace BLEND constant.
# ============================================================================
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e14-infer", E14_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in v8 source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

# Inject E14_INFER constant + replace BLEND constant body.
# Find the existing BLEND constant block boundary and replace.
# We'll inject E14_INFER right before the BLEND = r"""...""" definition.
BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E14_CONST_BLOCK = (
    '# === E14 inference cell (3-way blend extension) ===\n'
    'E14_INFER = ' + repr(E14_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, E14_CONST_BLOCK)

# Replace the entire BLEND constant value with our 3-way version.
# The original BLEND assignment spans from `BLEND = r"""...` to the matching `"""`.
# We use a regex to capture and replace.
OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
# Replace the entire BLEND = r"""...""" block including its closing triple-quote.
# Match from `BLEND = r"""` to the next `"""` (non-greedy across newlines).
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
# Use lambda for replacement so backslash sequences in NEW_BLEND aren't interpreted
# as backreferences/escape sequences by re.sub (this would turn \n into a real newline,
# breaking f-strings that contain literal "\n" tokens).
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

# Patch output filename + header markdown.
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_3way_e14.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# v12 = NB4 v7 + Tucker SED + exp014 R1 (3-way rank blend with Sonotype mirror)\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "v8 (NB4+Tucker LB 0.942) に exp014 R1 fold0 (LB 0.897) を 3 軸目として追加。Tucker 公開 5-fold ONNX",
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_3way_e14_inner.py")})
