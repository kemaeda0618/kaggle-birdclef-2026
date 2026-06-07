"""exp038: exp031 base から weight tweak (w30-40-30、exp029 boost).

Base: exp031 (NB4 0.35 / Tucker 0.40 / exp029 0.25) = LB 0.949
Change: BLEND_W_E10 0.35→0.30、BLEND_W_E17 0.25→0.30 (Tucker 0.40 維持)
        sum = 0.30 + 0.40 + 0.30 = 1.00 ✅

Motivation: exp029 (我々の自前 nfnet_l1 R3 distill、LB 0.923) を boost、
            NB4 (Karnakbayev public 系譜) を相対的に dampening → 公開 NB 系譜離脱寄り

Per memory `feedback_blend_naming_convention`: slug `birdclef2026-exp038-blend-w30-40-30`
Per memory `project_exp019_status`: 0.35/0.40/0.25 は weight 弾性飽和判定済、
   ±0.05 swing で +0.001 内外を期待
"""
import re as _re
from pathlib import Path

HERE = Path(__file__).parent
EXP031_GEN_PATH = HERE.parent.parent / "exp031" / "notebook" / "_gen_nb_blend.py"
SRC = EXP031_GEN_PATH.read_text(encoding="utf-8")

print(f"[exp038] reading exp031 base: {EXP031_GEN_PATH.name}")
print(f"  base size: {len(SRC)} chars")

# ============================================================================
# Patch 1: blend weight values
# ============================================================================
SRC = SRC.replace(
    'BLEND_W_E10  = 0.35   # NB4 v7 weight',
    'BLEND_W_E10  = 0.30   # exp038: NB4 0.35→0.30 (-0.05)',
)
SRC = SRC.replace(
    'BLEND_W_E17  = 0.25   # exp029 R3 (eca_nfnet_l1) weight — note var name kept "E17" for code reuse',
    'BLEND_W_E17  = 0.30   # exp038: exp029 0.25→0.30 (+0.05) — note var name kept "E17" for code reuse',
)

# ============================================================================
# Patch 2: header / ratio config comment
# ============================================================================
SRC = SRC.replace(
    "exp031 ratio config (w35-40-25, slot3 = exp029 instead of e17):",
    "exp038 ratio config (w30-40-30, exp029 boost from w35-40-25):",
)

# ============================================================================
# Patch 3: output ipynb name
# ============================================================================
# (exp031's output is nb_blend.ipynb, we keep same filename since dir is different)

# ============================================================================
# Patch 4: exp number references
# ============================================================================
SRC = SRC.replace("exp031", "exp038")

print(f"[exp038] patches applied, exec'ing modified source ({len(SRC)} chars)")

exec(SRC, {"__file__": str(HERE / "_gen_nb_blend_inner.py")})
