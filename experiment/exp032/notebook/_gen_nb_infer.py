"""Generate exp032 inference NB (Kaggle CPU 90min, internet=off).

Patches exp017 infer NB to use tf_efficientnet_b3 + Babych init (no Perch).

Changes:
- BACKBONE: eca_nfnet_l0 -> tf_efficientnet_b3.ns_jft_in1k
- USE_PERCH_DISTILL: True -> False
- Ckpt source: birdclef2026-exp017-weights -> birdclef2026-exp032-weights
- Ckpt names: drop r2_* prefix (R1 only for now)
- Slug-friendly header
"""
import re as _re
from pathlib import Path

HERE = Path(__file__).parent
EXP017_GEN_PATH = HERE.parent.parent / "exp017" / "notebook" / "_gen_nb_infer.py"
SRC = EXP017_GEN_PATH.read_text(encoding="utf-8")

# ============================================================================
# Patch 1: BACKBONE
# ============================================================================
SRC = SRC.replace(
    'BACKBONE = "eca_nfnet_l0"',
    'BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"',
)

# ============================================================================
# Patch 2: Disable Perch distill
# ============================================================================
SRC = SRC.replace(
    'USE_PERCH_DISTILL = True',
    'USE_PERCH_DISTILL = False  # exp032: Perch なし (Antoine 流)',
)

# ============================================================================
# Patch 3: Ckpt source dataset
# ============================================================================
SRC = SRC.replace(
    'birdclef2026-exp017-weights',
    'birdclef2026-exp032-weights',
)
SRC = SRC.replace(
    'birdclef2026-exp017-train',
    'birdclef2026-exp032-train',
)

# ============================================================================
# Patch 4: Ckpt priority — drop r2_* (R1 only), prefer ckpt_best_ns22.pth
# ============================================================================
SRC = SRC.replace(
    '''CKPT_PRIORITY = [
    "r2_ckpt_best_ns22.pth",   # R2 best (TBD val)
    "r2_ckpt_best_macro.pth",
    "r2_ckpt_latest.pth",
    "ckpt_best_ns22.pth",      # R1 v2 fallback (val 0.9166)
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]''',
    '''CKPT_PRIORITY = [
    "ckpt_best_ns22.pth",      # exp032 R1 (val 0.9156)
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]''',
)

# ============================================================================
# Patch 5: Header text
# ============================================================================
SRC = SRC.replace(
    '# exp017 — Inference (Kaggle CPU 90min, internet=off)',
    '# exp032 R1 — Inference (Kaggle CPU 90min, internet=off)',
)
SRC = SRC.replace(
    'Load best eca_nfnet_l0 SED ckpt from Kaggle Dataset, run inference on test_soundscapes,',
    'Load best tf_efficientnet_b3 (Babych init, no Perch) SED ckpt, run inference on test_soundscapes,',
)
SRC = SRC.replace(
    "(R1 v2 / R2 ckpts; uploaded from train NB Cell 17)",
    "(R1 ckpt; val 0.9156 / val_macro 0.9156)",
)

# ============================================================================
# Patch 6: Strip leftover R1 v2 mentions
# ============================================================================
SRC = SRC.replace(
    "R1 v2: val_ns22 0.9166 (after LR fix from v1 0.9065)",
    "exp032 R1: val_ns22 0.9156 (b3 + Babych init, no Perch, 25 ep)",
)

# ============================================================================
# Patch 7: Output ipynb name
# ============================================================================
SRC = _re.sub(
    r'out_path = HERE / "nb_infer\.ipynb"',
    'out_path = HERE / "nb_infer.ipynb"',
    SRC,
)

# Patch mel params for inference (must match training)
_mel_patches = [
    ('N_FFT      = 2048',     'N_FFT      = 4096   # exp032: Babych b3 mel param 整合'),
    ('HOP_LENGTH = 512',      'HOP_LENGTH = 312    # exp032: image (224, 512)'),
    ('N_MELS     = 256',      'N_MELS     = 224    # exp032: Babych mel'),
    ('FMIN       = 20',       'FMIN       = 0      # exp032: Babych mel'),
]
for old, new in _mel_patches:
    if old in SRC:
        SRC = SRC.replace(old, new)
        print(f"  mel-patch infer: {old[:50]}")
    else:
        print(f"  WARNING: mel-patch infer not found: {old[:50]}")

# Exec the patched gen
exec(SRC, {"__file__": str(HERE / "_gen_nb_infer_inner.py")})
