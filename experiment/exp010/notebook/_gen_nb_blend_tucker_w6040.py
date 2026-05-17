"""Generate v8 blend NB with weights 60:40 (NB4=0.60, Tucker=0.40).

Wrapper around _gen_nb_blend_tucker.py that overrides the blend weights and
output filename. Same slug uses different version, or new slug.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# Override 1: blend weights
mod = re.sub(r"BLEND_W_EXP010 = 0\.5", "BLEND_W_EXP010 = 0.6", SRC)
mod = re.sub(r"BLEND_W_SED    = 0\.5", "BLEND_W_SED    = 0.4", mod)

# Override 2: output ipynb filename
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_tucker_w6040.ipynb"',
    mod,
)

# Override 3: header comment for clarity
mod = mod.replace(
    "# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===",
    "# === Stage C: rank blend 60:40 (NB4 heavier) + Sonotype mirror + submission ==="
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_tucker_w6040_inner.py")})
