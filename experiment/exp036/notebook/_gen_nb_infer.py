"""exp036 inference NB (Kaggle CPU 90min, internet=off).

Base: exp017 nb_infer.ipynb (eca_nfnet_l0 SED architecture identical to exp017 R2).
Changes:
- Dataset: maekeso/birdclef2026-exp017-weights → maekeso/birdclef2026-exp036-softauc-weights
- ckpt priority: prefer exp036 SoftAUC fine-tune ckpt (val 0.9269)

期待 LB: exp017 R2 base 0.921 + Soft AUC fine-tune val Δ+0.002 → standalone 0.922-0.924
"""
import re as _re
from pathlib import Path

HERE = Path(__file__).parent
EXP017_GEN_PATH = HERE.parent.parent / "exp017" / "notebook" / "_gen_nb_infer.py"
SRC = EXP017_GEN_PATH.read_text(encoding="utf-8")

print(f"[exp036 infer] reading exp017 base: {EXP017_GEN_PATH}")
print(f"  base size: {len(SRC)} chars")

# ============================================================================
# Patch 1: dataset slug → exp036-softauc-weights
# ============================================================================
SRC = SRC.replace(
    "birdclef2026-exp017-weights",
    "birdclef2026-exp036-softauc-weights",
)

# Also patch the notebooks fallback path (was exp017-train)
SRC = SRC.replace(
    "birdclef2026-exp017-train",
    "birdclef2026-exp036-train",  # 念のため (実 NB は実際存在しない可能性ありだが harmless)
)

# ============================================================================
# Patch 2: output ipynb name and message text
# ============================================================================
SRC = SRC.replace(
    'out_path = HERE / "nb_infer.ipynb"',
    'out_path = HERE / "nb_infer_softauc.ipynb"',
)

# ============================================================================
# Patch 3: Header markdown - update title/desc
# ============================================================================
SRC = SRC.replace(
    "# exp017 — Inference",
    "# exp036 — Inference (Soft AUC fine-tune from exp017 R2)",
)
SRC = SRC.replace(
    "Load best eca_nfnet_l0 SED ckpt from Kaggle Dataset",
    "Load exp036 Soft AUC fine-tune ckpt (val ns22 0.9269、exp017 R2 base 0.9249 +0.0020) from Kaggle Dataset",
)

# ============================================================================
# Patch 4: Comment annotation tweaks (informative only)
# ============================================================================
SRC = SRC.replace(
    "R1 v2 / R2 ckpts",
    "exp036 Soft AUC fine-tune ckpt (val 0.9269)",
)
SRC = SRC.replace(
    "R1 v2: val_ns22 0.9166 (after LR fix from v1 0.9065)",
    "exp036: val_ns22 0.9269 (Soft AUC fine-tune from exp017 R2 0.9249)",
)

# ============================================================================
# Patch 5: 🐛 fix rglob patterns - exp036 ckpts have r2_ prefix
# Original: rglob("ckpt_best_*.pth") doesn't match "r2_ckpt_best_ns22.pth"
# Fix: prepend * wildcard so r2_ prefix is matched
# ============================================================================
SRC = SRC.replace(
    'if any(p.rglob("ckpt_best_*.pth")) or any(p.rglob("ckpt_latest*.pth")):',
    'if any(p.rglob("*ckpt_best_*.pth")) or any(p.rglob("*ckpt_latest*.pth")):',
)
SRC = SRC.replace(
    'for hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):',
    'for hit in Path("/kaggle/input").rglob("*ckpt_best_ns22.pth"):',
)

print(f"[exp036 infer] patches applied, exec'ing modified source ({len(SRC)} chars)")

# Execute the modified source
exec(SRC, {"__file__": str(HERE / "_gen_nb_infer_inner.py")})
