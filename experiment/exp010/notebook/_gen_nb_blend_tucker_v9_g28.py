"""v9: v8 + G28 (Student-t kernel smoothing for Tucker SED).

Diff vs v8: 1 line replace in SED_INFER
  旧: p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest")
  新: p_mean = convolve1d(p_mean, student_t_kernel, axis=0, mode="nearest")

Student-t kernel:
  df=1.5, scale=1.20, offsets [-3..3], normalized
  fat-tail → 短い call の高 conf 値を保持 (Gaussian は強い平滑で薄める)

公開 0.946 NB (yaroslavkholmirzayev) で実証された手法。
期待 LB: +0.001-0.003 (v8 0.942 → 0.943-0.945)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_gen_nb_blend_tucker.py").read_text(encoding="utf-8")

# Override 1: gaussian_filter1d を Student-t kernel に置換
OLD_LINE = 'p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)'
NEW_BLOCK = """# G28: Student-t kernel smoothing (公開 0.946 NB 由来、fat-tail で短 call 保持)
    from scipy.ndimage import convolve1d
    _offs = np.arange(-3, 4)
    _kernel = (1 + (_offs / 1.20)**2 / 1.5) ** (-(1.5 + 1) / 2)
    _kernel = _kernel / _kernel.sum()
    p_mean = convolve1d(p_mean, _kernel, axis=0, mode="nearest").astype(np.float32)"""

assert OLD_LINE in SRC, "Could not find gaussian_filter1d line in v8 source"
mod = SRC.replace(OLD_LINE, NEW_BLOCK)

# Override 2: header comment in BLEND cell (more informative)
mod = mod.replace(
    "# === Stage C: rank blend 50:50 + Sonotype mirror (v8) + submission ===",
    "# === Stage C: rank blend 50:50 + Sonotype mirror + G28 (v9) + submission ==="
)

# Override 3: output ipynb filename
mod = re.sub(
    r'out_path = HERE / "nb_blend_tucker.ipynb"',
    'out_path = HERE / "nb_blend_tucker_v9_g28.ipynb"',
    mod,
)

# Override 4: header markdown
mod = mod.replace(
    "# exp010 NB4 v7 + Tucker public 5-fold SED blend\\n",
    "# v9 = v8 + G28 (Student-t kernel temporal smoothing)\\n",
)

exec(mod, {"__file__": str(HERE / "_gen_nb_blend_tucker_v9_g28_inner.py")})
