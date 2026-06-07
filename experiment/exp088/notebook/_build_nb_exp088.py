"""Build exp088 NB from exp085 by:
1. Replace e015-infer cell with e082-infer cell (20s sliding + Babych mel)
2. Update blend cell: exp015 → exp082 (var names + weight comment)
3. Update header
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp088_4model.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# e082-infer cell content (REPLACES e015-infer)
# ============================================================
e082_infer_src = '''# === exp082 R2 OV (eca_nfnet_l0 + 20s sliding window + Babych mel) inference ===
# 4th stream for exp088. Window-axis + Mel-axis diversity (vs exp078 / exp085).
# - Window: 20s sliding (12 windows step 5s, padded 75s with 7.5s lead/trail)
# - Mel: Babych spec (n_mels=224, n_fft=4096, hop=1252, fmin=0, fmax=16000)
# Output probs_e082 of shape (N_files, 12, N_CLASSES), sigmoid space + gaussian smoothed.
import openvino as ov

# ---- Locate exp082 R2 OV IR (Dataset 経由) ----
E082_OV_PATH = None
for _root in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp082-weights"),
    Path("/kaggle/input/birdclef2026-exp082-weights"),
]:
    if _root.exists():
        _cand = _root / "r2_ov" / "model.xml"
        if _cand.exists():
            E082_OV_PATH = _cand; break
        for _hit in _root.rglob("r2_ov/model.xml"):
            E082_OV_PATH = _hit; break
        if E082_OV_PATH: break

assert E082_OV_PATH is not None, "exp082 R2 OV IR (r2_ov/model.xml) not found"
print(f"exp082 OV: {E082_OV_PATH}")

# ---- Load OV model ----
_e082_core = ov.Core()
_e082_ov_model = _e082_core.read_model(str(E082_OV_PATH))
_e082_compiled = _e082_core.compile_model(_e082_ov_model, "CPU")
_e082_input_port = _e082_compiled.input(0)
print(f"  exp082 OV compiled, input shape={_e082_compiled.input(0).partial_shape}")

# ---- Babych mel transform (different from Tucker mel used by other models) ----
import torchaudio
class _E082MelTF(nn.Module):
    """Babych mel: n_fft=4096, hop=1252, n_mels=224, fmin=0, fmax=16000 — matches exp082 training."""
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=4096, hop_length=1252,
            n_mels=224, f_min=0, f_max=16000, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))

e082_mel_tf = _E082MelTF().to(torch.device("cpu"))

# ---- 20s sliding window chunking (working v6 pattern from ensemble6) ----
def _prepare_input_sliding_20s(audio_60s):
    """20s sliding window: padded 75s with 7.5s lead/trail, 12 windows step 5s.

    - padded_sec = (12-1)*5 + 20 = 75
    - pad_lead_samples = (75-60)*SR // 2 = 240000 (= 7.5s)
    - Output: (12, 640000) = (12, 20s*32000Hz)
    """
    DURATION_SEC = 20
    padded_sec = (N_WINDOWS - 1) * 5 + DURATION_SEC   # 75
    padded_samples = padded_sec * SR
    pad_lead_samples = ((padded_sec - 60) * SR) // 2  # 240000 (7.5s)
    audio_padded = np.zeros(padded_samples, dtype=np.float32)
    audio_padded[pad_lead_samples:pad_lead_samples + len(audio_60s)] = audio_60s
    chunks = np.zeros((N_WINDOWS, SR * DURATION_SEC), dtype=np.float32)
    step_samples = 5 * SR
    win_samples = DURATION_SEC * SR
    for k in range(N_WINDOWS):
        start = k * step_samples
        chunks[k] = audio_padded[start:start + win_samples]
    return chunks


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

# ---- Inference using audio_cache (60s raw per file) + 20s sliding window ----
t0 = time.time()
probs_e082 = []   # (N_files, 12, N_CLASSES)
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _prepare_input_sliding_20s(_raw_60s)  # (12, 640000)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)  # (12, 1, 640000)
        _mel = e082_mel_tf(_wav_t)                       # (12, 1, 224, 512)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _mel_np = _mel.numpy().astype(np.float32)
        _result = _e082_compiled.create_infer_request().infer({_e082_input_port: _mel_np})
        _clip_logits, _framewise = _identify_ov_outputs(_result)
        _frame_max = _framewise.max(axis=1)
        _p_clip   = sigmoid_sed(_clip_logits).astype(np.float32)
        _p_frame  = sigmoid_sed(_frame_max).astype(np.float32)
        _p_mean   = 0.5 * _p_clip + 0.5 * _p_frame
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e082.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e082 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e082 = np.stack(probs_e082).astype(np.float32)
print(f"e082 done: {probs_e082.shape} in {time.time()-t0:.0f}s")
'''

# ============================================================
# blend cell content (4-way with exp082 instead of exp015)
# ============================================================
new_blend_src = '''# === Stage D: 4-way rank blend (exp037 + Tucker + exp029 + exp082) + Sonotype mirror + submission ===
# ★ exp088: 4-way blend (exp078 base + exp082 R2 20s sliding + Babych mel).
# Weight allocation (sum = 1.00):
#   exp037 v313 (LightProtoSSM):   0.30  (was 0.35 in exp078)
#   Tucker SED 5-fold (effv2_b0):  0.35  (was 0.40)
#   exp029 R3 (eca_nfnet_l1):       0.20  (was 0.25)
#   exp082 R2 (eca_nfnet_l0, 20s + Babych mel): 0.15  ★ NEW (window+mel axis)
BLEND_W_E10  = 0.30   # exp037 weight
BLEND_W_SED  = 0.35   # Tucker public SED weight
BLEND_W_E17  = 0.20   # exp029 R3 (eca_nfnet_l1) weight
BLEND_W_E082 = 0.15   # ★ exp082 R2 (20s + Babych) weight
USE_SED_PRE_AVG = False  # 4-way pure rank blend (no pre-avg)

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E082 - 1.0) < 1e-6, \\
    "blend weights must sum to 1.0"

flat_e10  = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed  = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17  = probs_e17.reshape(-1, probs_e17.shape[-1])    # ← actually exp029
flat_e082 = probs_e082.reshape(-1, probs_e082.shape[-1])  # ← ★ exp082 R2

rank_e10  = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed  = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17  = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e082 = pd.DataFrame(flat_e082).rank(axis=0, pct=True).to_numpy().astype(np.float32)

blend_flat = (BLEND_W_E10  * rank_e10
              + BLEND_W_SED  * rank_sed
              + BLEND_W_E17  * rank_e17
              + BLEND_W_E082 * rank_e082)
print(f"  4-way rank blend: exp037={BLEND_W_E10} / Tucker={BLEND_W_SED} / e29={BLEND_W_E17} / e082={BLEND_W_E082}")

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
new_header_src = '''# exp088 — 4-way blend (exp078 base + exp082 R2 OV: 20s sliding + Babych mel)

**exp078 (LB 0.950) を base に exp082 R2 を 4th stream として追加**

★ exp085 (= exp078 + exp015 ConvNeXt) = LB 0.949 (-0.001 drag, ConvNeXt redundant) の反省を踏まえ、
exp088 は **window 軸 (20s) + mel 軸 (Babych mel)** という exp078 で未開拓の 2 軸を同時投入。

## Stack

| # | Model | 単 LB | Window | Mel | Weight |
|---|---|---|---|---|---|
| 1 | **exp037 v313** (LightProtoSSM + MLP probes + ResSSM) | 0.930 | — | Perch embed | **0.30** |
| 2 | **Tucker SED 5-fold** ONNX (effv2_b0) | 0.937 | 5s | Tucker | **0.35** |
| 3 | **exp029 R3** (eca_nfnet_l1) | 0.923 | 5s | Tucker | **0.20** |
| 4 | **exp082 R2 OV** (eca_nfnet_l0, 20s, Babych mel) ★ NEW | ~0.91 | **20s sliding** | **Babych** | **0.15** |

**Weight sum = 1.00**

## Expected
- LB **0.951-0.955** (exp078 baseline 0.950 から +0.001-0.005)
- exp085 (-0.001 drag) より高確率で improvement: window + mel 軸の二重新規性
- Run time **~75-85 min** on Kaggle CPU
  - exp037 ~30 min + Tucker SED ~15-20 min + exp029 R3 ~8-12 min + **exp082 OV ~12-15 min** + audio reload ~5-8 min

## ★ exp085 (ConvNeXt) と exp088 (20s+Babych) の差

| 項目 | exp085 | exp088 |
|---|---|---|
| 4th model | exp015 (ConvNeXt × 5s × Tucker mel) | exp082 (NFNet × **20s** × **Babych** mel) |
| Backbone diversity vs exp078 | + ConvNeXt | 重複 (l0 vs l1 だが族同じ) |
| Window diversity vs exp078 | 同 5s | **+20s 新軸** ★ |
| Mel diversity vs exp078 | 同 Tucker | **+Babych 新軸** ★ |
| 期待 ensemble bonus | +0.000-0.002 | +0.002-0.005 |

## Pipeline
1. Cell 0-25: Train exp037 v313 in-NB
2. Cell `bridge`: Build audio_cache from test files (60s raw per file)
3. Cell `sed-infer`: Tucker SED 5-fold inference
4. Cell `e29-infer`: exp029 R3 PyTorch inference
5. Cell `e082-infer`: ★ exp082 R2 OV inference (**20s sliding + Babych mel**)
6. Cell `blend`: 4-way rank blend + Sonotype mirror + submission

## Inputs (Dataset attachments)
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`
- `maekeso/birdclef2026-exp029-l1-single` (exp029 R3 ckpt)
- `cooolz/openvino-package` (OV wheel)
- `maekeso/birdclef2026-exp082-weights` ★ NEW (exp082 R2 OV IR via r2_ov/)
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
'''

# ============================================================
# Apply edits
# ============================================================
cells = nb["cells"]
def cell_idx_by_id(cells, cid):
    for i, c in enumerate(cells):
        if c.get("id") == cid:
            return i
    return -1

i_hdr = cell_idx_by_id(cells, "hdr")
i_e015 = cell_idx_by_id(cells, "e015-infer")
i_blend = cell_idx_by_id(cells, "blend")

assert i_hdr >= 0 and i_e015 >= 0 and i_blend >= 0, \
    f"cell not found: hdr={i_hdr}, e015={i_e015}, blend={i_blend}"

# Update header
cells[i_hdr]["source"] = new_header_src.splitlines(keepends=True)

# Replace e015-infer with e082-infer
cells[i_e015]["source"] = e082_infer_src.splitlines(keepends=True)
cells[i_e015]["id"] = "e082-infer"

# Update blend cell
cells[i_blend]["source"] = new_blend_src.splitlines(keepends=True)

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
