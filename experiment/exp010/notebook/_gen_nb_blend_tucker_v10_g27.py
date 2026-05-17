"""v10: v8 + G27 (Cross-branch agreement gate, RANK-BASED).

v6 で公開 0.946 NB の絶対閾値 (proto>0.50) を試して失敗 (我々の出力分布で発火せず)。
本 v10 では **rank-based** に書き換えて再挑戦。

Logic:
  fake_only:  rank_proto > 0.95 & rank_sed < 0.10  → blend を 1.08x
  proto_cont: rank_proto > 0.88 & rank_sed < 0.15  → blend を 1.15x
  sed_only:   rank_sed   > 0.95 & rank_proto < 0.80 → blend を 1.12x

期待 LB: +0.002-0.005 (v8 0.942 → 0.944-0.947)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# v8 の BLEND cell の rank-blend と Sonotype mirror の間に G27 gate logic を挿入
OLD = """# Rank-space blend
blend_flat = BLEND_W_EXP010 * rank_010 + BLEND_W_SED * rank_sed   # (N_total, N_CLASSES)

# === Sonotype mirror (公開 0.946 NB Cell 40) ==="""

NEW = """# Rank-space blend
blend_flat = BLEND_W_EXP010 * rank_010 + BLEND_W_SED * rank_sed   # (N_total, N_CLASSES)

# === G27: Cross-branch agreement gate (RANK-BASED) ===
# v6 で絶対閾値版は失敗 (出力分布不整合)、rank-based に置換
# fake_only:  NB4 高 conf × SED 低 conf  → NB4 を信じて boost
# proto_cont: NB4 中高 × SED 低         → NB4 を強めに boost
# sed_only:   SED 高 conf × NB4 低 conf  → SED を信じて boost
fake_only  = (rank_010 > 0.95) & (rank_sed < 0.10)
proto_cont = (rank_010 > 0.88) & (rank_sed < 0.15) & (~fake_only)
sed_only   = (rank_sed > 0.95) & (rank_010 < 0.80) & (~fake_only) & (~proto_cont)

blend_flat = blend_flat.copy()
blend_flat[fake_only]  *= 1.08
blend_flat[proto_cont] *= 1.15
blend_flat[sed_only]   *= 1.12
blend_flat = np.clip(blend_flat, 0.0, 1.0)

print(f"  G27 gate fired:")
print(f"    fake_only:  {int(fake_only.sum()):>6d} cells ({fake_only.mean()*100:.2f}%)")
print(f"    proto_cont: {int(proto_cont.sum()):>6d} cells ({proto_cont.mean()*100:.2f}%)")
print(f"    sed_only:   {int(sed_only.sum()):>6d} cells ({sed_only.mean()*100:.2f}%)")

# === Sonotype mirror (公開 0.946 NB Cell 40) ==="""

assert OLD in SRC, "Could not locate rank-blend / Sonotype boundary in v8 source"
mod = SRC.replace(OLD, NEW)

# Header comment
mod = mod.replace(
    "# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===",
    "# === Stage C: rank blend + G27 gate (v10) + Sonotype mirror + submission ==="
)

# Output filename
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_tucker_v10_g27.ipynb"',
    mod,
)

# Header markdown
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# v10 = v8 + G27 (rank-based cross-branch gate)\\n",
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_tucker_v10_g27_inner.py")})
