"""v13: 5-way blend = NB4 v7 + Tucker public SED + exp014 R2 ep10 + exp015 R1 + exp016 R2.

Adds 2 more SED axes on top of v12 (3-way):
- exp015 R1: convnext_pico.d1_in1k, kernel_sources `birdclef2026-exp015-train`, ckpt_best_ns22.pth
- exp016 R2: regnety_008, dataset_sources `birdclef2026-exp016-weights`, r2_ckpt_best_ns22.pth

3 SED models share the same architecture (BirdSEDModel: backbone + GeMFreqPool + Dense + Att/Cla),
only the backbone name and ckpt path differ. We share the class definition and call inference 3 times.

ratio config (initial v6 push):
  (NB4, Tucker, e14, e15, e16) = (0.30, 0.40, 0.10, 0.05, 0.15)  # 案 A、sum=1.0

期待 LB: v5 (3-way, 0.945) → 0.945-0.947 (e16 R2 +0.0015、e15 R1 +0.0-0.001、e14 weight 減 -0.001 で net +)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# ============================================================================
# Cell: sed-common (shared BirdSEDModel architecture for e14/e15/e16)
# ============================================================================
SED_COMMON = r'''# === Shared SED architecture (used by e14, e15, e16) ===
import timm
import torchaudio

SED_N_MELS   = 256
SED_N_FFT    = 2048
SED_HOP      = 512
SED_FMIN     = 20
SED_FMAX     = 16000
SED_SAMPLES  = SR * 5
SED_PERCH_DIM = 1536


class _SEDMelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP,
            n_mels=SED_N_MELS, f_min=SED_FMIN, f_max=SED_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _SEDGeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _SEDDistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _SEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512, use_distill=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = SED_SAMPLES // SED_HOP + 1
            dummy = torch.randn(1, 1, SED_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _SEDGeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.use_distill = use_distill
        if use_distill:
            self.distill_head = _SEDDistillHead(self.backbone_dim, SED_PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if self.use_distill else h
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


_sed_mel_tf = _SEDMelTF().to(torch.device("cpu"))


def _run_sed_inference(label, backbone_name, ckpt_path):
    """Run SED inference using shared audio_cache. Returns probs of shape (N_files, 12, N_CLASSES)."""
    try:
        _state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        _state = torch.load(str(ckpt_path), map_location="cpu")
    print(f"  [{label}] ckpt epoch={_state.get('epoch')}, "
          f"best_ns22={_state.get('best_ns22', float('nan')):.4f}")
    _model = _SEDModel(backbone_name=backbone_name).to(torch.device("cpu"))
    _model.load_state_dict(_state["model_state"], strict=False)
    _model.eval()
    print(f"  [{label}] loaded {sum(p.numel() for p in _model.parameters())/1e6:.1f}M params")

    t0 = time.time()
    probs_out = []
    with torch.no_grad():
        for _fi, _raw_60s in enumerate(audio_cache):
            _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
            _wav_t = torch.from_numpy(_chunks).unsqueeze(1)
            _mel = _sed_mel_tf(_wav_t)
            for _i in range(_mel.size(0)):
                _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
            _clip, _frame = _model(_mel, return_framewise=True)
            _frame_max = _frame.max(dim=1).values
            _p_clip   = torch.sigmoid(_clip).numpy().astype(np.float32)
            _p_frame  = torch.sigmoid(_frame_max).numpy().astype(np.float32)
            _p_mean   = 0.5 * _p_clip + 0.5 * _p_frame
            _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
            probs_out.append(_p_smooth)
            if (_fi + 1) % 20 == 0 or _fi == len(audio_cache) - 1:
                print(f"  [{label}] [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")
    return np.stack(probs_out).astype(np.float32)


def _resolve_ckpt(label, dir_candidates, ckpt_names):
    """Return first matching ckpt path under any candidate dir, or None."""
    _state_dir = None
    for _p in dir_candidates:
        if _p.exists():
            for _name in ckpt_names:
                if any(_p.rglob(_name)):
                    _state_dir = _p; break
            if _state_dir is not None: break
    if _state_dir is None:
        # last-resort sweep
        for _name in ckpt_names:
            for _hit in Path("/kaggle/input").rglob(_name):
                _state_dir = _hit.parent; break
            if _state_dir is not None: break
    if _state_dir is None:
        print(f"  [{label}] WARNING: ckpt dir not found")
        return None
    for _name in ckpt_names:
        _hits = list(_state_dir.rglob(_name))
        if _hits:
            print(f"  [{label}] ckpt: {_hits[0]} ({_hits[0].stat().st_size/1e6:.1f}MB)")
            return _hits[0]
    print(f"  [{label}] WARNING: no ckpt under {_state_dir}")
    return None
'''

# ============================================================================
# Cells: per-SED inference (e14 / e15 / e16)
# ============================================================================
E14_INFER = r'''# === exp014 R2 ep10 (HGNetV2-B0) inference ===
_e14_ckpt = _resolve_ckpt(
    "e14",
    [Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp014-train-r2"),
     Path("/kaggle/input/birdclef2026-exp014-train-r2"),
     Path("/kaggle/input/datasets/maekeso/exp014-state-r2"),
     Path("/kaggle/input/exp014-state-r2"),
     Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp014-train"),
     Path("/kaggle/input/birdclef2026-exp014-train")],
    ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth"],
)
assert _e14_ckpt is not None, "exp014 R2 ckpt not found"
probs_e14 = _run_sed_inference("e14", "hgnetv2_b0.ssld_stage2_ft_in1k", _e14_ckpt)
print(f"e14 done: {probs_e14.shape}")
'''

E15_INFER = r'''# === exp015 R2 (convnext_pico) inference ===
_e15_ckpt = _resolve_ckpt(
    "e15",
    [Path("/kaggle/input/datasets/maekeso/birdclef2026-exp015-weights"),
     Path("/kaggle/input/birdclef2026-exp015-weights"),
     Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp015-train")],
    ["r2_ckpt_best_ns22.pth", "r2_ckpt_best_macro.pth", "r2_ckpt_latest.pth",
     "ckpt_best_ns22.pth", "ckpt_best_macro.pth"],
)
assert _e15_ckpt is not None, "exp015 R2 ckpt not found"
probs_e15 = _run_sed_inference("e15", "convnext_pico.d1_in1k", _e15_ckpt)
print(f"e15 done: {probs_e15.shape}")
'''

E16_INFER = r'''# === exp016 R2 (regnety_008) inference ===
_e16_ckpt = _resolve_ckpt(
    "e16",
    [Path("/kaggle/input/datasets/maekeso/birdclef2026-exp016-weights"),
     Path("/kaggle/input/birdclef2026-exp016-weights"),
     Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp016-train")],
    ["r2_ckpt_best_ns22.pth", "r2_ckpt_best_macro.pth", "r2_ckpt_latest.pth",
     "ckpt_best_ns22.pth", "ckpt_best_macro.pth"],
)
assert _e16_ckpt is not None, "exp016 R2 ckpt not found"
probs_e16 = _run_sed_inference("e16", "regnety_008", _e16_ckpt)
print(f"e16 done: {probs_e16.shape}")
'''

# ============================================================================
# 5-way BLEND + Sonotype mirror + FCS + submission
# ============================================================================
NEW_BLEND = r'''# === Stage C: 5-way rank blend (NB4 + Tucker + e14 + e15 + e16) + Sonotype mirror + submission ===
# v13 ratio config (per push, this is the only block to edit between versions):
# v5 (2026-05-15): D 案 = 完全均等 5-way (BC2025 1 位流)、全 e1X を R2 ckpt に統一
BLEND_W_E10  = 0.20   # NB4 v7 weight
BLEND_W_SED  = 0.20   # Tucker public 5-fold SED weight
BLEND_W_E14  = 0.20   # exp014 R2 ep10 weight
BLEND_W_E15  = 0.20   # exp015 R2 weight (val 0.9311、新規)
BLEND_W_E16  = 0.20   # exp016 R2 weight

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E14 + BLEND_W_E15 + BLEND_W_E16 - 1.0) < 1e-6, \
    "blend weights must sum to 1.0"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e14 = probs_e14.reshape(-1, probs_e14.shape[-1])
flat_e15 = probs_e15.reshape(-1, probs_e15.shape[-1])
flat_e16 = probs_e16.reshape(-1, probs_e16.shape[-1])

rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e14 = pd.DataFrame(flat_e14).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e15 = pd.DataFrame(flat_e15).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e16 = pd.DataFrame(flat_e16).rank(axis=0, pct=True).to_numpy().astype(np.float32)

blend_flat = (BLEND_W_E10 * rank_e10
              + BLEND_W_SED * rank_sed
              + BLEND_W_E14 * rank_e14
              + BLEND_W_E15 * rank_e15
              + BLEND_W_E16 * rank_e16)
print(f"  5-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / "
      f"e14={BLEND_W_E14} / e15={BLEND_W_E15} / e16={BLEND_W_E16}")

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

# file_confidence_scale (same as v5)
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
# Patch source: insert sed-common + 3 SED inference cells + replace BLEND.
# ============================================================================
CELL_APPEND_MARKER = 'cells.append(code_cell("blend", BLEND))'
CELL_APPEND_REPLACEMENT = (
    'cells.append(code_cell("sed-common", SED_COMMON))\n'
    'cells.append(code_cell("e14-infer", E14_INFER))\n'
    'cells.append(code_cell("e15-infer", E15_INFER))\n'
    'cells.append(code_cell("e16-infer", E16_INFER))\n'
    'cells.append(code_cell("blend", BLEND))'
)
assert CELL_APPEND_MARKER in SRC, "Could not find blend cells.append() in tucker source"
mod = SRC.replace(CELL_APPEND_MARKER, CELL_APPEND_REPLACEMENT)

# Inject 4 string constants right before BLEND constant definition.
BLEND_CONST_MARKER = '# Stage C: blend + submission\nBLEND = r"""'
INJECT_CONSTS = (
    '# === SED shared + per-model inference cells ===\n'
    'SED_COMMON = ' + repr(SED_COMMON) + '\n\n'
    'E14_INFER = ' + repr(E14_INFER) + '\n\n'
    'E15_INFER = ' + repr(E15_INFER) + '\n\n'
    'E16_INFER = ' + repr(E16_INFER) + '\n\n'
    '# Stage C: blend + submission\nBLEND = r"""'
)
assert BLEND_CONST_MARKER in mod, "Could not find BLEND constant boundary"
mod = mod.replace(BLEND_CONST_MARKER, INJECT_CONSTS)

# Replace the entire BLEND constant value with our 5-way version.
OLD_BLEND_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ==='
assert re.search(OLD_BLEND_PATTERN, mod), "Could not find old BLEND start"
BLEND_FULL_PATTERN = r'BLEND = r"""# === Stage C: rank blend 50:50 \+ Sonotype mirror \(v8\) \+ submission ===.*?"""'
_blend_replacement = 'BLEND = r"""' + NEW_BLEND + '"""'
mod = re.sub(BLEND_FULL_PATTERN, lambda _m: _blend_replacement, mod, count=1, flags=re.DOTALL)

# Patch output filename + header markdown.
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_5way.ipynb"',
    mod,
)
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# v13 = NB4 + Tucker + exp014 R2 + exp015 R1 + exp016 R2 (5-way rank blend)\\n",
)
mod = mod.replace(
    "exp012 (自前 fold0, LB 0.890) を捨てて **Tucker 公開 5-fold ONNX**",
    "v5 (NB4+Tucker+e14 R2 LB 0.945) に exp015 R1 + exp016 R2 を加えた 5-way blend",
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_5way_inner.py")})
