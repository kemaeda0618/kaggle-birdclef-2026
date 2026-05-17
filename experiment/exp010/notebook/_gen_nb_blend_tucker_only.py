"""Generate v8 blend NB with weights 0:100 (NB4=0.00, Tucker=1.00) — Tucker only.

Same Sonotype mirror + file_conf_scale post-processing, just zero out NB4.
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

mod = re.sub(r"BLEND_W_EXP010 = 0\.5", "BLEND_W_EXP010 = 0.0", SRC)
mod = re.sub(r"BLEND_W_SED    = 0\.5", "BLEND_W_SED    = 1.0", mod)

mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_tucker_only.ipynb"',
    mod,
)

mod = mod.replace(
    "# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===",
    "# === Stage C: Tucker only (0:100) + Sonotype mirror + submission ==="
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_tucker_only_inner.py")})
