"""v8 (0.942) + Gates 1/2/3 (mtoshidesu 0.947 NB Cell 34 由来、BirdNET なし).

v8 baseline: NB4 + Tucker rank blend 50:50 + Sonotype mirror = LB 0.942.
このスクリプトは v8 source を読み、BLEND cell に Gate 1 (Noise suppress) /
Gate 2 (Cauchy temporal continuity) / Gate 3 (SED spike preservation) を
rank blend と Sonotype mirror の間に挿入する。

Gate 3b (BirdNET spike) は BirdNET 軸がないため skip。

期待 LB: v8 (0.942) → +0.003-0.005 = 0.945-0.947
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# ============================================================================
# Gates code to insert after rank blend, before Sonotype mirror.
# ============================================================================
GATES_CODE = '''
# === Gates 1/2/3 (mtoshidesu 0.947 NB Cell 34 由来、BirdNET なし) ===
EPS_GATE = 1e-5
p_proto = np.clip(flat_010, EPS_GATE, 1.0 - EPS_GATE)
p_sed   = np.clip(flat_sed, EPS_GATE, 1.0 - EPS_GATE)

# File ID per row (Gate 2 で per-file Cauchy kernel に使う)
_file_ids = np.array(["_".join(str(_r).split("_")[:-1]) for _r in all_row_ids])

# Gate 1: Noise suppression (Proto confident なのに SED 強い disagreement → Proto 戻し)
fake_only = (p_proto > 0.50) & (p_sed < 0.08)
blend_flat = np.where(fake_only, (1.0 - 0.08) * blend_flat + 0.08 * rank_010, blend_flat).astype(np.float32)
print(f"  Gate 1 (Noise suppress): {int(fake_only.sum())} cells modified")

# Gate 2: Temporal continuity (fat-tailed Cauchy/Student-t kernel, 35s window = 7 chunks)
_offs = np.arange(-3, 4, dtype=np.float32)
_proto_kernel = (1.0 + (_offs / 1.20) ** 2 / 2.0) ** (-1.5)
_proto_kernel = (_proto_kernel / _proto_kernel.sum()).astype(np.float32)

_pa_ctx = p_proto.copy()
for _fid in pd.unique(_file_ids):
    _m = _file_ids == _fid
    _x = p_proto[_m]
    if len(_x) > 1:
        _xp = np.pad(_x, ((3, 3), (0, 0)), mode="edge")
        _pa_ctx[_m] = sum(_proto_kernel[_i] * _xp[_i:_i + len(_x)] for _i in range(7))

_xctx = pd.DataFrame(_pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
proto_cont = (_xctx > 0.88) & (rank_010 > 0.75) & (p_sed < 0.12) & (~fake_only)
blend_flat = np.where(
    proto_cont,
    (1.0 - 0.15) * blend_flat + 0.15 * np.maximum(rank_010, _xctx),
    blend_flat,
).astype(np.float32)
print(f"  Gate 2 (Temporal continuity): {int(proto_cont.sum())} cells modified")

# Gate 3: SED spike preservation (SED 単独で高 rank かつ Proto 低 → SED 注入)
sed_only = (rank_sed > 0.95) & (rank_010 < 0.80) & (~fake_only) & (~proto_cont)
blend_flat = np.where(sed_only, (1.0 - 0.12) * blend_flat + 0.12 * rank_sed, blend_flat).astype(np.float32)
print(f"  Gate 3 (SED spike): {int(sed_only.sum())} cells modified")
# === End Gates 1/2/3 ===

'''

# ============================================================================
# Patch: insert GATES_CODE after rank blend, before Sonotype mirror.
# ============================================================================
BLEND_MARKER = "blend_flat = BLEND_W_EXP010 * rank_010 + BLEND_W_SED * rank_sed   # (N_total, N_CLASSES)\n"
assert BLEND_MARKER in SRC, "Could not find rank-blend marker in v8 source"
mod = SRC.replace(BLEND_MARKER, BLEND_MARKER + GATES_CODE)

# Patch comment header (v8 → v12 gates).
mod = mod.replace(
    "# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===",
    "# === Stage C: rank blend 50:50 + Gates 1/2/3 + Sonotype mirror (v12) + submission ===",
)
mod = mod.replace(
    "# v8: revert ratio to 50:50 (best confirmed by v1/v5/v7 ablation)",
    "# v12 = v8 (rank 50:50 + Sonotype mirror, LB 0.942) + Gates 1/2/3 (mtoshidesu 0.947 NB 由来)",
)

# Output to a different ipynb filename so v8 baseline ipynb is preserved.
mod = mod.replace(
    'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_tucker_gates.ipynb"',
)

# Update header markdown.
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# exp010 NB4 + Tucker SED + Gates 1/2/3 (v12)\\n",
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_tucker_gates_inner.py")})
