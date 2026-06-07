"""Build exp085 4-model ensemble NB by:
1. Adding openvino install to Cell 0 (v313_00)
2. Inserting new cell e015-infer after e29-infer
3. Updating blend cell to 4-way (NB4/Tucker/exp029/exp015 → exp037/Tucker/exp029/exp015)
4. Updating header
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp085_4model.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Updated Cell 0: add openvino wheel install
# ============================================================
new_cell0_src = '''# ── Cell 0: Install ONNX Runtime + TF 2.20 + OpenVINO (offline wheels) ─────
import subprocess, sys, os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    raise FileNotFoundError(pattern)

# Try ONNX first (150x faster than TF SavedModel)
ONNX_WHL = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
if ONNX_WHL.exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(ONNX_WHL)], check=True)
    print("ONNX Runtime installed")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                str(find_wheel("tensorboard-2.20.0-*.whl"))], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                str(find_wheel("tensorflow-2.20.0-*.whl"))], check=True)
print("TF 2.20 installed")

# ★ exp085: install OpenVINO from cooolz/openvino-package wheel dataset (offline)
import glob as _glob_ov
OV_WHEEL_DIR = None
for _p in [Path("/kaggle/input/datasets/cooolz/openvino-package"),
           Path("/kaggle/input/openvino-package")]:
    if _p.exists():
        OV_WHEEL_DIR = _p; break
assert OV_WHEEL_DIR is not None, "openvino wheel dataset not attached (cooolz/openvino-package)"
_ov_whl  = _glob_ov.glob(str(OV_WHEEL_DIR / "openvino-*.whl"))
_tel_whl = _glob_ov.glob(str(OV_WHEEL_DIR / "openvino_telemetry-*.whl"))
assert _ov_whl and _tel_whl, f"openvino wheels not found in {OV_WHEEL_DIR}"
# --no-deps to avoid numpy upgrade (which breaks scipy)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                *_ov_whl, *_tel_whl], check=True)
print("OpenVINO installed")

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    print("ONNX Runtime available ✅")
except ImportError:
    _ONNX_AVAILABLE = False
    print("ONNX not available, falling back to TF")

import openvino as ov
print(f"OpenVINO available ✅ ({ov.__version__})")
'''

# ============================================================
# New cell e015-infer (insert after e29-infer)
# ============================================================
e015_infer_src = '''# === exp015 R2 OV (convnext_pico + Tucker mel) inference ===
# 4th stream for exp085. Reuses audio_cache (60s raw per file) and _E17MelTF (Tucker mel).
# Output probs_e015 of shape (N_files, 12, N_CLASSES), sigmoid space + gaussian smoothed.
import openvino as ov

# ---- Locate exp015 R2 OV IR (Dataset 経由) ----
E015_OV_PATH = None
for _root in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp015-weights"),
    Path("/kaggle/input/birdclef2026-exp015-weights"),
]:
    if _root.exists():
        _cand = _root / "r2_ov" / "model.xml"
        if _cand.exists():
            E015_OV_PATH = _cand; break
        # rglob fallback
        for _hit in _root.rglob("r2_ov/model.xml"):
            E015_OV_PATH = _hit; break
        if E015_OV_PATH: break

assert E015_OV_PATH is not None, "exp015 R2 OV IR (r2_ov/model.xml) not found"
print(f"exp015 OV: {E015_OV_PATH}")

# ---- Load OV model ----
_e015_core = ov.Core()
_e015_ov_model = _e015_core.read_model(str(E015_OV_PATH))
_e015_compiled = _e015_core.compile_model(_e015_ov_model, "CPU")
_e015_input_port = _e015_compiled.input(0)
print(f"  exp015 OV compiled, input shape={_e015_compiled.input(0).partial_shape}")

# Reuse _E17MelTF (Tucker mel: n_mels=256, hop=512, n_fft=2048) — same as exp015 training
# Already defined in e29-infer cell
e015_mel_tf = _E17MelTF().to(torch.device("cpu"))

def _identify_ov_outputs(result_dict):
    """Identify clip_logits (2D) and framewise (3D) by ndim — robust to missing names."""
    arrays = list(result_dict.values())
    clip_logits, framewise = None, None
    for arr in arrays:
        if arr.ndim == 2:
            clip_logits = arr
        elif arr.ndim == 3:
            framewise = arr
    assert clip_logits is not None and framewise is not None, \\
        f"OV output shape mismatch: {[a.shape for a in arrays]}"
    return clip_logits, framewise

# ---- Inference using audio_cache (60s raw per file) ----
t0 = time.time()
probs_e015 = []   # (N_files, 12, N_CLASSES)
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)        # (12, 1, 160000)
        _mel = e015_mel_tf(_wav_t)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _mel_np = _mel.numpy().astype(np.float32)              # (12, 1, 256, 313)
        _result = _e015_compiled.create_infer_request().infer({_e015_input_port: _mel_np})
        _clip_logits, _framewise = _identify_ov_outputs(_result)
        # framewise shape: (B, time, num_class) — max over time axis 1
        _frame_max = _framewise.max(axis=1)
        # sigmoid space blend (mirrors Tucker/e29 recipe: 0.5*sig(clip) + 0.5*sig(frame_max))
        _p_clip   = sigmoid_sed(_clip_logits).astype(np.float32)
        _p_frame  = sigmoid_sed(_frame_max).astype(np.float32)
        _p_mean   = 0.5 * _p_clip + 0.5 * _p_frame             # (12, N_CLASSES)
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e015.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e015 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e015 = np.stack(probs_e015).astype(np.float32)
print(f"e015 done: {probs_e015.shape} in {time.time()-t0:.0f}s")
'''

# ============================================================
# Updated blend cell (4-way) — replace 3-way with 4-way
# ============================================================
new_blend_src = '''# === Stage D: 4-way rank blend (exp037 + Tucker + exp029 + exp015) + Sonotype mirror + submission ===
# ★ exp085: 4-way blend (was 3-way in exp078).
# Weight allocation (sum = 1.00):
#   exp037 v313 (LightProtoSSM):  0.30  (was 0.35 in exp078)
#   Tucker SED 5-fold (effv2_b0):  0.35  (was 0.40)
#   exp029 R3 (eca_nfnet_l1):       0.20  (was 0.25)
#   exp015 R2 (convnext_pico):     0.15  ★ NEW
BLEND_W_E10  = 0.30   # exp037 weight
BLEND_W_SED  = 0.35   # Tucker public SED weight
BLEND_W_E17  = 0.20   # exp029 R3 (eca_nfnet_l1) weight
BLEND_W_E015 = 0.15   # ★ exp015 R2 (convnext_pico) weight
USE_SED_PRE_AVG = False  # 4-way pure rank blend (no pre-avg)

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E015 - 1.0) < 1e-6, \\
    "blend weights must sum to 1.0"

flat_e10  = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed  = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17  = probs_e17.reshape(-1, probs_e17.shape[-1])    # ← actually exp029
flat_e015 = probs_e015.reshape(-1, probs_e015.shape[-1])  # ← ★ exp015 R2

rank_e10  = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed  = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17  = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e015 = pd.DataFrame(flat_e015).rank(axis=0, pct=True).to_numpy().astype(np.float32)

blend_flat = (BLEND_W_E10  * rank_e10
              + BLEND_W_SED  * rank_sed
              + BLEND_W_E17  * rank_e17
              + BLEND_W_E015 * rank_e015)
print(f"  4-way rank blend: exp037={BLEND_W_E10} / Tucker={BLEND_W_SED} / e29={BLEND_W_E17} / e015={BLEND_W_E015}")

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
print(f"\\nSubmission: {submission.shape}, total {total:.0f}s ({total/60:.1f} min)")
print(f"Mean pred: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"Max pred:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head())
'''

# ============================================================
# Updated header
# ============================================================
new_header_src = '''# exp085 — 4-way blend (exp078 base + exp015 R2 OV)

**exp078 (LB 0.950) を base に exp015 R2 (convnext_pico OV) を 4th stream として追加**

## Stack

| # | Model | 単 LB | Weight |
|---|---|---|---|
| 1 | **exp037 v313** (LightProtoSSM + MLP probes + ResSSM) | 0.930 | **0.30** |
| 2 | **Tucker SED 5-fold** ONNX (effv2_b0, public) | 0.937 | **0.35** |
| 3 | **exp029 R3** (eca_nfnet_l1) | 0.923 | **0.20** |
| 4 | **exp015 R2 OV** (convnext_pico) ★ NEW | 0.910 | **0.15** |

**Weight sum = 1.00**

## Expected
- LB **0.951-0.955** (exp078 baseline 0.950 から +0.001-0.005)
- Run time **~70-80 min** on Kaggle CPU
  - exp037 ~30 min + Tucker SED ~15-20 min + exp029 R3 ~8-12 min + **exp015 OV ~3-5 min** + audio reload ~5-8 min

## Backbone diversity (5 families! ※ Tucker SED の effv2_b0 含む)
- LightProtoSSM (Perch probe, exp037)
- EfficientNet v2 b0 (Tucker SED)
- NFNet (eca_nfnet_l1, exp029 R3)
- ConvNeXt (convnext_pico, exp015) ★ NEW

## Pipeline
1. Cell 0-25: Train exp037 v313 in-NB (Perch v2 emb → LightProtoSSM + MLP probes + ResidualSSM)
2. Cell `bridge`: Build audio_cache from test files (60s raw per file)
3. Cell `sed-infer`: Tucker SED 5-fold inference (sigmoid + smooth)
4. Cell `e29-infer`: exp029 R3 PyTorch inference (sigmoid + smooth)
5. Cell `e015-infer`: ★ exp015 R2 OV inference (sigmoid + smooth)
6. Cell `blend`: 4-way **rank blend** + Sonotype mirror + submission

## Inputs (Dataset attachments)
- `birdclef-2026` (competition)
- `tuckerarrants/perch-v2-no-dft-onnx`
- `tuckerarrants/bc2026-distilled-sed-public` (Tucker SED 5-fold ONNX)
- `cooolz/openvino-package` ★ NEW (OV wheel offline install)
- `maekeso/birdclef2026-exp029-l1-single` (exp029 R3 ckpt)
- `maekeso/birdclef2026-exp015-weights` ★ NEW (exp015 R2 OV IR via r2_ov/)
- `rishikeshjani/perch-onnx-for-birdclef-2026` (ONNX runtime wheel)
- `rishikeshjani/tensorflow-2-20-0-wheel-cp312`
'''

# ============================================================
# Apply edits
# ============================================================

# Find cells by id
cells = nb["cells"]
def cell_idx_by_id(cells, cid):
    for i, c in enumerate(cells):
        if c.get("id") == cid:
            return i
    return -1

i_hdr = cell_idx_by_id(cells, "hdr")
i_cell0 = cell_idx_by_id(cells, "v313_00")
i_e29 = cell_idx_by_id(cells, "e29-infer")
i_blend = cell_idx_by_id(cells, "blend")

assert i_hdr >= 0 and i_cell0 >= 0 and i_e29 >= 0 and i_blend >= 0, \
    f"cell not found: hdr={i_hdr}, cell0={i_cell0}, e29={i_e29}, blend={i_blend}"

# Update header
cells[i_hdr]["source"] = new_header_src.splitlines(keepends=True)

# Update Cell 0 (v313_00) to add openvino wheel install
cells[i_cell0]["source"] = new_cell0_src.splitlines(keepends=True)

# Update blend cell to 4-way
cells[i_blend]["source"] = new_blend_src.splitlines(keepends=True)

# Insert new e015-infer cell between e29-infer and blend
new_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": e015_infer_src.splitlines(keepends=True),
    "id": "e015-infer",
}
cells.insert(i_e29 + 1, new_cell)

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")
print(f"Cell count: {len(cells)}")

# Verify
print("\nCell order:")
for i, c in enumerate(cells):
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    label = ""
    for ln in src.split("\n")[:5]:
        s = ln.strip()
        if s and not s.startswith("# ===") and not s.startswith("```"):
            label = s[:80]; break
    print(f"  {i:2d} id={cid:25s} | {label}")
