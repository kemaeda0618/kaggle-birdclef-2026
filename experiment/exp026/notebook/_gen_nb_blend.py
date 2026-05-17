"""exp026: NB4 + Tucker + (e17 R2 **5-fold weight-averaged + ONNX**) blend.

- exp019 ratio (w35-40-25) 維持で **5-fold ensemble 効果 isolate**
- e17 5-fold ckpts (r2_fold{0-4}_ckpt_best_ns22.pth from `birdclef2026-exp020-weights-5fold`)
- ONNX 変換 in NB (推論時に export): PyTorch ckpt → 重み平均 → 1 ONNX → onnxruntime 推論
- NFNet (BN-free + WS Conv) は weight averaging 数学的に clean
- 期待 LB: 0.948-0.951 (5-fold variance 削減で +0.001-0.003)

設計上の判断:
- 5-fold 個別 ONNX 推論 (5x time) は 90 min 超なので **weight averaging で 1 ONNX**
- ONNX は PyTorch 比 ~1.5-2x 高速、e17 部分 15-20 min で済む
- export 工数 inline ~30 秒
"""
import re
from pathlib import Path

HERE = Path(__file__).parent  # experiment/exp026/notebook
TUCKER_SRC_PATH = HERE.parent.parent / "exp010" / "notebook" / "_gen_nb_blend_tucker.py"
SRC = TUCKER_SRC_PATH.read_text(encoding="utf-8")

# ============================================================================
# E17 5-fold weight-averaged + ONNX inference cell
# (exp018 e17 single inference cell を完全置換、share audio_cache from Tucker)
# ============================================================================
E17_5FOLD_ONNX_INFER = r'''# === exp020 e17 R2 5-fold weight-averaged + ONNX inference ===
# --- Offline install onnxruntime from wheel dataset (Kernels Only / internet=False) ---
import subprocess, sys as _sys
_WHEEL_CANDIDATES = [
    Path("/kaggle/input/datasets/romantamrazov/onnxruntime-1-24-4"),
    Path("/kaggle/input/onnxruntime-1-24-4"),
]
_wheel_dir = None
for _wd in _WHEEL_CANDIDATES:
    if _wd.exists():
        _wheel_dir = _wd; break
if _wheel_dir is None:
    for _hit in Path("/kaggle/input").rglob("onnxruntime-1.24.4-*.whl"):
        _wheel_dir = _hit.parent; break
assert _wheel_dir is not None, "onnxruntime wheel dataset not found"
print(f"[onnx] wheel dir: {_wheel_dir}")

_pip_result = subprocess.run(
    [_sys.executable, "-m", "pip", "install",
     "--no-index", "--no-deps",
     "--find-links", str(_wheel_dir), "onnxruntime"],
    capture_output=True, text=True,
)
if _pip_result.returncode != 0:
    print(f"[onnx] STDOUT: {_pip_result.stdout}\n[onnx] STDERR: {_pip_result.stderr}")
    raise RuntimeError("offline install of onnxruntime failed")
print("[onnx] onnxruntime installed offline")

import timm
import torchaudio
import onnxruntime as ort

# ---- Locate exp020 5-fold ckpts ----
E17_DATASET_DIR = None
for _p in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp020-weights-5fold"),
    Path("/kaggle/input/birdclef2026-exp020-weights-5fold"),
]:
    if _p.exists() and (any(_p.rglob("r2_fold0_ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_best_ns22.pth"))):
        E17_DATASET_DIR = _p; break
if E17_DATASET_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r2_fold0_ckpt_best_ns22.pth"):
        E17_DATASET_DIR = _hit.parent; break
assert E17_DATASET_DIR is not None, "exp020 5-fold dataset not found"
print(f"exp020 5-fold dataset: {E17_DATASET_DIR}")

# Find all 5 fold ckpts
E17_FOLD_CKPTS = []
for _fk in range(5):
    _candidates = [
        E17_DATASET_DIR / f"r2_fold{_fk}_ckpt_best_ns22.pth",
        E17_DATASET_DIR / f"fold{_fk}" / "r2_ckpt_best_ns22.pth",
        E17_DATASET_DIR / f"fold{_fk}_ckpt_best_ns22.pth",
    ]
    _p = None
    for _c in _candidates:
        if _c.exists():
            _p = _c; break
    if _p is None:
        for _hit in E17_DATASET_DIR.rglob(f"*fold{_fk}*ckpt_best_ns22*.pth"):
            _p = _hit; break
    assert _p is not None, f"Fold {_fk} ckpt not found in {E17_DATASET_DIR}"
    E17_FOLD_CKPTS.append(_p)
    print(f"  fold {_fk}: {_p.name} ({_p.stat().st_size/1e6:.1f}MB)")

# ---- exp017/020 SED config (must match training) ----
E17_BACKBONE = "eca_nfnet_l0"
E17_N_MELS   = 256
E17_N_FFT    = 2048
E17_HOP      = 512
E17_FMIN     = 20
E17_FMAX     = 16000
E17_TRAIN_SAMPLES = SR * 5
E17_USE_DISTILL = True
E17_PERCH_DIM = 1536

# ---- Architecture (same as exp017/018) ----
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
    def forward(self, x):
        h = self.backbone(x)
        h_cls = h.detach() if E17_USE_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        # ONNX-friendly: always return both (no Bool kwarg)
        return clip_logits, framewise_logits.permute(0, 2, 1)


# ---- Load 5 ckpts + weight average ----
print(f"\n[E17 5-fold] Loading + averaging 5 ckpts...")
_states = []
for _fk, _p in enumerate(E17_FOLD_CKPTS):
    try:
        _state = torch.load(str(_p), map_location="cpu", weights_only=False)
    except TypeError:
        _state = torch.load(str(_p), map_location="cpu")
    print(f"  fold {_fk}: epoch={_state.get('epoch')}, "
          f"best_ns22={_state.get('best_ns22', float('nan')):.4f}")
    _states.append(_state["model_state"])

# Weight average (NFNet BN-free 安全)
_avg_state = {}
for _k in _states[0].keys():
    _tensors = [s[_k].float() for s in _states]
    _avg_state[_k] = sum(_tensors) / len(_tensors)
print(f"[E17 5-fold] averaged {len(_states)} ckpts into 1 model")

# Build model + load avg state
e17_model = _E17SED().to(torch.device("cpu"))
e17_model.load_state_dict(_avg_state, strict=False)
e17_model.eval()
e17_mel_tf = _E17MelTF().to(torch.device("cpu"))
print(f"[E17 5-fold] model: {sum(p.numel() for p in e17_model.parameters())/1e6:.1f}M params")

# ---- Export to ONNX (in-NB, at inference time) ----
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
_onnx_path = "e17_5fold_avg.onnx"
_n_tf = E17_TRAIN_SAMPLES // E17_HOP + 1   # time frames
_dummy_mel = torch.randn(1, 1, E17_N_MELS, _n_tf)
torch.onnx.export(
    e17_model,
    _dummy_mel,
    _onnx_path,
    input_names=["mel"],
    output_names=["clip_logits", "frame_logits"],
    dynamic_axes={
        "mel": {0: "batch"},
        "clip_logits": {0: "batch"},
        "frame_logits": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,   # 公開 wheel offline 環境向け: legacy TorchScript exporter (onnxscript 不要)
)
_onnx_size_mb = Path(_onnx_path).stat().st_size / 1e6
print(f"[E17 5-fold] exported ONNX: {_onnx_path} ({_onnx_size_mb:.1f}MB)")

# Free PyTorch model after ONNX export
del e17_model
import gc; gc.collect()

# ---- Load ONNX session ----
_sess_opts = ort.SessionOptions()
_sess_opts.intra_op_num_threads = 4
_sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
e17_session = ort.InferenceSession(_onnx_path, sess_options=_sess_opts, providers=["CPUExecutionProvider"])
print(f"[E17 5-fold] ONNX session ready (4 threads, CPU)")

# ---- Inference using audio_cache (60s raw per file) ----
t0 = time.time()
probs_e17 = []
for _fi, _raw_60s in enumerate(audio_cache):
    _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
    _wav_t = torch.from_numpy(_chunks).unsqueeze(1)        # (12, 1, 160000)
    _mel = e17_mel_tf(_wav_t)
    # per-instance standardize
    for _i in range(_mel.size(0)):
        _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
    # ONNX inference (numpy input)
    _clip_np, _frame_np = e17_session.run(None, {"mel": _mel.numpy().astype(np.float32)})
    _frame_max_np = _frame_np.max(axis=1)
    # sigmoid blend
    _p_clip   = 1.0 / (1.0 + np.exp(-_clip_np))
    _p_frame  = 1.0 / (1.0 + np.exp(-_frame_max_np))
    _p_mean   = 0.5 * _p_clip + 0.5 * _p_frame
    # gaussian smooth (Tucker sigma=0.65)
    _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_e17.append(_p_smooth)
    if (_fi + 1) % 20 == 0 or _fi == len(audio_cache) - 1:
        print(f"  e17 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e17 = np.stack(probs_e17).astype(np.float32)
print(f"e17 5-fold avg ONNX done: {probs_e17.shape} in {time.time()-t0:.0f}s")
'''

# ============================================================================
# BLEND cell: exp019 と同一 ratio (w35-40-25)、e17 5-fold ONNX 効果 isolate
# ============================================================================
NEW_BLEND = r'''# === Stage C: 3-way rank blend (NB4 + Tucker + e17 5-fold ONNX) + Sonotype mirror + submission ===
# exp026 ratio (w35-40-25, exp019 と同): e17 5-fold avg ONNX 効果 isolate
BLEND_W_E10  = 0.35   # NB4 v7
BLEND_W_SED  = 0.40   # Tucker public SED
BLEND_W_E17  = 0.25   # e17 R2 5-fold weight-averaged ONNX
USE_SED_PRE_AVG = False

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6, \
    "blend weights must sum to 1.0"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])

rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
blend_flat = (BLEND_W_E10 * rank_e10
              + BLEND_W_SED * rank_sed
              + BLEND_W_E17 * rank_e17)
print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e17 5-fold avg ONNX={BLEND_W_E17}")

# Sonotype mirror groups
MIRROR_PAIRS = (
    # ─── 既存 (公開 0.948 NB から継承、spectrogram + acoustic 根拠) ───
    ("47158son15", "47158son16"),                                   # J=1.000  A=1.000
    ("47158son09", "47158son12"),                                   # J=0.222  A=0.995 (acoustic-only)
    ("47158son02", "47158son14"),                                   # J=0.583  A=0.998
    ("47158son13", "47158son21", "47158son22", "47158son23"),       # J=0.611-1.000  A=0.929-1.000
    # ─── 🆕 Tier 1 追加 (J>=0.5 AND A>=0.85、experiment/sonotype_viz 解析) ───
    ("47158son06", "47158son14"),                                   # J=0.667  A=0.947
    ("47158son02", "47158son06"),                                   # J=0.389  A=0.946  (s02-s14 と合体して (s02,s06,s14) 形成)
    ("47158son08", "47158son20"),                                   # J=0.706  A=0.943
    ("47158son11", "47158son24"),                                   # J=0.667  A=0.943
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
# Patch source: insert e17 5-fold ONNX cell + replace BLEND constant
# ============================================================================
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("e17-5fold-onnx", E17_5FOLD_ONNX_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in tucker source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
E17_CONST_BLOCK = (
    '# === E17 5-fold weight-averaged + ONNX inference cell (exp026) ===\n'
    'E17_5FOLD_ONNX_INFER = ' + repr(E17_5FOLD_ONNX_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, E17_CONST_BLOCK)

OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

# Inject onnxruntime + onnx install (if not already in INSTALL cell)
# The tucker base INSTALL pip-installs onnxruntime via existing line. Confirm.
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_5fold_onnx.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp026 = NB4 + Tucker + (e17 R2 5-fold weight-averaged + inline ONNX export) blend\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "exp019 (w35-40-25 LB 0.947) の e17 単体を **e17 R2 5-fold weight-averaged + 推論時 ONNX 変換** に置換、ratio 同一で 5-fold ensemble 効果 isolate。Tucker 公開 5-fold ONNX",
)

import sys
_EXP010_NB_DIR = str(TUCKER_SRC_PATH.parent)
if _EXP010_NB_DIR not in sys.path:
    sys.path.insert(0, _EXP010_NB_DIR)
exec(mod, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
