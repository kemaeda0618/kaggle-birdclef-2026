"""exp036: exp029 R3 + Soft AUC Loss fine-tune (Colab Pro A100).

discussion 700763 BC25 4位 Dylan Liu の **Soft AUC Loss** (rank-aware loss) を
exp029 R3 ckpt (val 0.9409 / LB 0.923) に warm-start fine-tune で適用。

期待 ΔLB: +0.001-0.005 (0.949 帯 scale 換算、上振れ +0.005-0.010)。

Changes from exp029 R3 train NB:
- Loss: BCE → 0.5 * BCE + 0.5 * SoftAUC
- N_TOTAL_EPOCHS: 20 → 8 (fine-tune only)
- WARMUP_EPOCHS: 4 → 1
- LR: 3e-4 → 1e-4 (warm-start なので保守)
- Warm-start: exp029 R3 ckpt (`r3_fold0_ckpt_best_ns22.pth`) を init として load
- Output: exp036/r3-softauc/ckpt_best_ns22.pth
- Dataset upload: maekeso/birdclef2026-exp036-softauc-weights
"""
import re as _re
from pathlib import Path

HERE = Path(__file__).parent
# ★ exp036: exp017 R2 (LB 0.921 既測定) を base に Soft AUC fine-tune
# exp029 R3 と比べて R2 → ablation 比較が clean
EXP017_GEN_PATH = HERE.parent.parent / "exp017" / "notebook" / "_gen_nb_train_r2.py"
SRC = EXP017_GEN_PATH.read_text(encoding="utf-8")

# ============================================================================
# Patch 1: epochs / LR / warmup (fine-tune mode)
# ============================================================================
# Find existing N_TOTAL_EPOCHS line and replace (exp017 R2 used ep=20)
import re as _re2
# Replace ALL occurrences (markdown bullets + code) for consistency
SRC = _re2.sub(
    r'N_TOTAL_EPOCHS\s*=\s*\d+[^\n]*',
    'N_TOTAL_EPOCHS = 8          # ★ exp036: warm-start fine-tune (8 ep で十分)',
    SRC,
)
SRC = _re2.sub(
    r'WARMUP_EPOCHS\s*=\s*\d+[^\n]*',
    'WARMUP_EPOCHS = 1   # ★ exp036: fine-tune なので短く',
    SRC,
)
SRC = _re2.sub(
    r'^LR\s*=\s*[\de\-\.]+[^\n]*',
    'LR = 1e-4           # ★ exp036: warm-start なので保守、catastrophic forgetting 回避',
    SRC, flags=_re2.MULTILINE,
)

# ============================================================================
# Patch 2: Drive output path & dataset (exp017 → exp036)
# 注意: R1 pseudo は exp017 の input をそのまま使う (再生成不要)
#       output (r2 ckpt, r2-pseudo) のみ exp036 にする
# ============================================================================
# Output dirs: exp017/r2 → exp036/r2、exp017/r2-pseudo → exp036/r2-pseudo
SRC = SRC.replace(
    '"output" / "exp017" / "r2"',
    '"output" / "exp036" / "r2"',
)
SRC = SRC.replace(
    '"output" / "exp017" / "r2-pseudo"',
    '"output" / "exp036" / "r2-pseudo"',
)
# Dataset upload name
SRC = SRC.replace(
    "birdclef2026-exp017-weights",
    "birdclef2026-exp036-softauc-weights",
)
# NOTE: DRIVE_R1_PSEUDO = "output" / "exp017" / "r1-pseudo" / ... stays as exp017

# ============================================================================
# Patch 3: Inject Soft AUC loss function before training loop
# Find the model creation cell, add soft_auc_loss helper
# ============================================================================
SOFT_AUC_FN = r'''
# ★ exp036: Soft AUC Loss (BC25 4位 Dylan Liu 流、rank-aware loss)
# logits: (B, C), labels: (B, C), mask: (B, C) - 1 if active species, 0 else
def soft_auc_loss_multilabel(logits, labels, mask=None, sharpness=10.0):
    B, C = logits.shape
    if mask is None:
        mask = torch.ones_like(labels)
    loss = 0.0
    count = 0
    for c in range(C):
        valid = mask[:, c] > 0
        if valid.sum() < 2:
            continue
        cls_logits = logits[valid, c]
        cls_labels = labels[valid, c]
        pos_idx = (cls_labels > 0.5)
        neg_idx = ~pos_idx
        if pos_idx.sum() == 0 or neg_idx.sum() == 0:
            continue
        pos_scores = cls_logits[pos_idx]
        neg_scores = cls_logits[neg_idx]
        # Pairwise difference (n_pos, n_neg)
        diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        # Soft penalty: sigmoid(-diff * k) - large when neg > pos
        pair_loss = torch.sigmoid(-diffs * sharpness).mean()
        loss = loss + pair_loss
        count += 1
    if count == 0:
        return logits.sum() * 0.0  # zero tensor with grad
    return loss / count

SOFT_AUC_ALPHA = 0.5     # BCE vs Soft AUC mixing: 0.5 = 50:50 ([[user's choice]])
SOFT_AUC_SHARPNESS = 10.0
print(f"\\n★ exp036: SOFT_AUC_ALPHA={SOFT_AUC_ALPHA}, SHARPNESS={SOFT_AUC_SHARPNESS}")
'''

# Insert SOFT_AUC_FN before make_model() (common to exp017/exp029)
_target = 'def make_model():'
if _target in SRC:
    SRC = SRC.replace(_target, SOFT_AUC_FN + '\n\n' + _target, 1)
    print("  Inserted Soft AUC function before make_model()")
else:
    print("  WARNING: make_model() marker not found")

# ============================================================================
# Patch 3.5: Warm-start from exp029 R3 ckpt (load model_state at init)
# Inject ckpt load logic just before "train_r2" begins
# ============================================================================
WARMSTART_CODE = r'''
# ★ exp036: Warm-start helper - load exp017 R2 ckpt as init
def warmstart_from_exp017(model):
    candidates = [
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights/r2_ckpt_best_ns22.pth"),
        DRIVE_INPUT_DIR / "output" / "exp017" / "r2" / "r2_ckpt_best_ns22.pth",
        DRIVE_INPUT_DIR / "output" / "exp017" / "r2" / "ckpt_best_ns22.pth",
    ]
    for c in candidates:
        if c.exists():
            print(f"[exp036 warm-start] Loading exp017 R2 ckpt: {c}")
            try:
                _state = torch.load(str(c), map_location="cpu", weights_only=False)
            except TypeError:
                _state = torch.load(str(c), map_location="cpu")
            if "model_state" in _state:
                msg = model.load_state_dict(_state["model_state"], strict=False)
            else:
                msg = model.load_state_dict(_state, strict=False)
            print(f"  loaded (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
            print(f"  source val_ns22={_state.get('best_ns22', float('nan')):.4f}")
            return True
    print(f"[exp036 warm-start] exp017 R2 ckpt NOT found in: {candidates}")
    return False

'''
# Insert WARMSTART_CODE right before SOFT_AUC_FN (which is inserted before train_r2)
SRC = SRC.replace(SOFT_AUC_FN, WARMSTART_CODE + SOFT_AUC_FN, 1)

# Insert warmstart call in model creation:
# Find where BirdSEDModel is instantiated in train_r2 and add warmstart
SRC = SRC.replace(
    'model = BirdSEDModel(',
    '_tmp_model = BirdSEDModel(',
    1,
)
# Adjust to: create model, then warmstart, then continue
SRC = SRC.replace(
    '_tmp_model = BirdSEDModel(',
    'model = BirdSEDModel(',
)
# Add warmstart call after model creation
# Find the line after which model is created (somewhere in train_r2)
# Use simpler insertion: find pattern "model.to(device)" and inject before
SRC = SRC.replace(
    "model = BirdSEDModel(\n",
    "model = BirdSEDModel(\n",  # keep as-is for now, will inject elsewhere
)

# Cleaner: inject after .to(device) for fold_k > 0 should not warmstart again
# Just inject warmstart_from_exp029(model) call after BirdSEDModel(...).to(device)
# Find the train_r2 function model load
_warm_inject = '_warmstarted = warmstart_from_exp029(model)\n        '
# Find first `model = BirdSEDModel(...)\n...\nmodel.to(device)`  pattern
_pattern_make_model = 'def make_model():\n    m = BirdSEDModel().to(device)\n    m = m.to(memory_format=torch.channels_last)\n    return m'
_replacement_make_model = 'def make_model():\n    m = BirdSEDModel().to(device)\n    warmstart_from_exp017(m)  # ★ exp036: exp017 R2 init load\n    m = m.to(memory_format=torch.channels_last)\n    return m'
if _pattern_make_model in SRC:
    SRC = SRC.replace(_pattern_make_model, _replacement_make_model, 1)
    print("  Inserted warmstart call in make_model()")
else:
    # exp017 R2 NB may have different model creation pattern - try alternatives
    for alt_pattern in [
        'model = BirdSEDModel().to(device)\n    model = model.to(memory_format=torch.channels_last)',
        'model = BirdSEDModel().to(device)',
    ]:
        if alt_pattern in SRC:
            SRC = SRC.replace(
                alt_pattern,
                alt_pattern.replace(
                    'model = BirdSEDModel().to(device)',
                    'model = BirdSEDModel().to(device)\n    warmstart_from_exp017(model)  # ★ exp036',
                ),
                1,
            )
            print(f"  Inserted warmstart (alt pattern: {alt_pattern[:50]}...)")
            break
    else:
        print("  WARNING: model creation pattern not found, warm-start may not activate")

# ============================================================================
# Patch 4: Modify loss computation in training loop
# ============================================================================
OLD_LOSS = '''            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
            bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
            bce = 0.5 * bce_clip + 0.5 * bce_frame
            ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
            cls_loss = (ps * sw).mean()'''

NEW_LOSS = '''            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
            bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
            bce = 0.5 * bce_clip + 0.5 * bce_frame
            ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
            bce_cls_loss = (ps * sw).mean()
            # ★ exp036: Soft AUC loss on clip_logits
            soft_auc = soft_auc_loss_multilabel(clip_logits, lb, mask=mk, sharpness=SOFT_AUC_SHARPNESS)
            cls_loss = (1 - SOFT_AUC_ALPHA) * bce_cls_loss + SOFT_AUC_ALPHA * soft_auc'''

if OLD_LOSS in SRC:
    SRC = SRC.replace(OLD_LOSS, NEW_LOSS)
    print("  Patched loss: BCE + Soft AUC")
else:
    print("  WARNING: OLD_LOSS pattern not found, please verify")

# ============================================================================
# Patch 5: Header
# ============================================================================
SRC = SRC.replace(
    "exp017 R2",
    "exp036 Soft AUC fine-tune (from exp017 R2)",
)
SRC = SRC.replace(
    "exp017 R1 pseudo",
    "exp017 R1 pseudo (warm-start ckpt = exp017 R2)",
)

# ============================================================================
# Patch 6: Output ipynb name
# ============================================================================
SRC = _re.sub(
    r'out_path = HERE / "nb_train[^"]*\.ipynb"',
    'out_path = HERE / "nb_train_softauc.ipynb"',
    SRC,
)

exec(SRC, {"__file__": str(HERE / "_gen_nb_train_softauc_inner.py")})
