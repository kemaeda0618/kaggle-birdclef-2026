"""Patch exp100 gen script -> exp102 (3-fold, 30 epoch, WRS warm-up, EMA).

Changes from exp100 R3:
  1. FOLDS [0] -> [0, 1, 2]                       (3-fold ensemble)
  2. N_TOTAL_EPOCHS 20 -> 30                       (Babych late jump 完走)
  3. WARMUP_EPOCHS 4 -> 6                          (epoch 比例)
  4. BATCH 192 -> 640                              (Blackwell 適正)
  5. NUM_WORKERS 16 -> 24                          (48 vCPU 適正)
  6. WRS warm-up: uniform->inv-freq gradient ep 1-5
  7. EMA decay 0.999 weight averaging + val 評価
  8. Slug: birdclef2026-exp102-l1-3fold-r3p

Run: python _patch_to_exp102.py
Then: python _gen_nb_train_r3.py  ->  produces nb_train_r3.ipynb
"""
import io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(__file__).with_name("_gen_nb_train_r3.py")
content = SRC.read_text(encoding="utf-8")
print(f"Original size: {len(content)} chars")

# ============================================================
# Patch 1-5: Config 5 行
# ============================================================
simple_patches = [
    # FOLDS
    ("FOLDS = [0]               # single fold 学習 (homogenize 回避)、ただし split は 5-fold で行う",
     "FOLDS = [0, 1, 2]         # ★ exp102: 3-fold ensemble"),

    # N_TOTAL_EPOCHS
    ("N_TOTAL_EPOCHS = 20         # ★ exp100 Option C: 15 → 20、warmup 4/20 = 20% で適正",
     "N_TOTAL_EPOCHS = 30         # ★ exp102: 20 -> 30 (Babych late jump 完走)"),

    # WARMUP_EPOCHS
    ("WARMUP_EPOCHS = 4         # ★ exp100: l1 で +1 ep (exp022 同設定)",
     "WARMUP_EPOCHS = 6         # ★ exp102: 4 -> 6 (epoch 比例)"),

    # BATCH
    ("BATCH = 192",
     "BATCH = 640         # ★ exp102: 192 -> 640 (Blackwell 95.6 GB)"),

    # NUM_WORKERS
    ("NUM_WORKERS = 16        # Plan I: 12 → 16、data loading 並列度 +33%",
     "NUM_WORKERS = 24        # ★ exp102: 16 -> 24 (48 vCPU 適正)"),
]
for old, new in simple_patches:
    assert content.count(old) == 1, f"Pattern not unique: {old[:60]!r} (count={content.count(old)})"
    content = content.replace(old, new)
    print(f"[OK] {old[:60]}")

# ============================================================
# Patch 6: Add WRS_WARMUP_EPOCHS config
# ============================================================
WRS_CONFIG_NEW = """NUM_WORKERS = 24        # ★ exp102: 16 -> 24 (48 vCPU 適正)
PERSISTENT_WORKERS = True

# ★ exp102: WRS warm-up + EMA config
WRS_WARMUP_EPOCHS = 5    # ep 1-5: uniform -> inv-freq gradient transition
USE_EMA = True
EMA_DECAY = 0.999"""

# Replace 'NUM_WORKERS = ... \nPERSISTENT_WORKERS = True' to also add WRS/EMA
WRS_CONFIG_OLD = """NUM_WORKERS = 24        # ★ exp102: 16 -> 24 (48 vCPU 適正)
PERSISTENT_WORKERS = True"""
assert WRS_CONFIG_OLD in content, "WRS config anchor not found"
content = content.replace(WRS_CONFIG_OLD, WRS_CONFIG_NEW)
print("[OK] WRS_WARMUP_EPOCHS + EMA config added")

# ============================================================
# Patch 7: WRS warm-up — compute effective_w per epoch in train_r2
# ============================================================
# Original (in train_r2 epoch loop):
OLD_WRS = """smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch, focal_weights=_focal_weights)"""

NEW_WRS = """# ★ exp102: WRS warm-up — uniform -> inv-freq gradient (ep 1: pure uniform, ep WRS_WARMUP_EPOCHS+: pure inv-freq)
        _wrs_alpha = min(1.0, (epoch + 1) / max(WRS_WARMUP_EPOCHS, 1))
        _focal_uniform = np.full(len(_focal_weights), 1.0 / len(_focal_weights), dtype=np.float64)
        _focal_effective = (1.0 - _wrs_alpha) * _focal_uniform + _wrs_alpha * _focal_weights
        _focal_effective = _focal_effective / _focal_effective.sum()
        if epoch < WRS_WARMUP_EPOCHS:
            print(f"  [WRS warm-up] ep {epoch+1}: alpha={_wrs_alpha:.2f}, max/min ratio={_focal_effective.max()/max(_focal_effective.min(),1e-12):.1f}")
        smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch, focal_weights=_focal_effective)"""

assert OLD_WRS in content, "WRS smp construction anchor not found"
content = content.replace(OLD_WRS, NEW_WRS)
print("[OK] WRS warm-up logic added")

# ============================================================
# Patch 8: EMA — class + integration in train_r2
# ============================================================
# Insert EMA class before train_r2 function, integrate into train loop.
# EMA class injection point: right after eval helpers, before train_r2 cell starts.

EMA_CLASS = """
# ★ exp102: EMA (Exponential Moving Average) weight averaging
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        # Also track buffers (BN stats etc.)
        self.shadow_buf = {n: b.detach().clone() for n, b in model.named_buffers()}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # Buffers: replace (BN stats benefit from EMA too)
        for n, b in model.named_buffers():
            if n in self.shadow_buf:
                self.shadow_buf[n].mul_(self.decay).add_(b.detach().float(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        # Swap model weights with EMA, return original to restore later
        backup_p = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        backup_b = {n: b.detach().clone() for n, b in model.named_buffers()}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])
        for n, b in model.named_buffers():
            if n in self.shadow_buf:
                b.data.copy_(self.shadow_buf[n].to(b.dtype))
        return backup_p, backup_b
    @torch.no_grad()
    def restore(self, model, backup_p, backup_b):
        for n, p in model.named_parameters():
            if n in backup_p:
                p.data.copy_(backup_p[n])
        for n, b in model.named_buffers():
            if n in backup_b:
                b.data.copy_(backup_b[n])

print("OK ModelEMA class defined")
"""

# Inject EMA class right before "def train_r2(" — search the def
OLD_TRAIN_R2_DEF = "def train_r2(fold_k, pseudo_meta_df, Y_pseudo, out_dir):"
assert OLD_TRAIN_R2_DEF in content, "train_r2 def anchor not found"
content = content.replace(OLD_TRAIN_R2_DEF, EMA_CLASS.lstrip() + "\n\n" + OLD_TRAIN_R2_DEF)
print("[OK] EMA class injected")

# Patch: EMA init right after `model = make_model()`
OLD_MAKE = """model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)"""
NEW_MAKE = """model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # ★ exp102: EMA initialization
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None"""
assert OLD_MAKE in content, "make_model + optimizer anchor not found"
content = content.replace(OLD_MAKE, NEW_MAKE)
print("[OK] EMA init added")

# Patch: EMA update after optimizer.step() in the training inner loop
# Find unique pattern of optimizer.step() in train loop (not other places)
OLD_OPT_STEP = """scaler.step(optimizer)
            scaler.update()
            scheduler.step()"""
NEW_OPT_STEP = """scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # ★ exp102: EMA update after each optimizer step
            if ema is not None:
                ema.update(model)"""
assert OLD_OPT_STEP in content, "optimizer step anchor not found"
content = content.replace(OLD_OPT_STEP, NEW_OPT_STEP)
print("[OK] EMA update inserted after optimizer.step()")

# Patch: EMA evaluation at epoch end — find val evaluation section and add EMA branch
# Locate "_predict_from_waveforms(model, ..." call in epoch summary
OLD_VAL_PRED = "preds = _predict_from_waveforms(model, mel_transform, val_wavs)"
NEW_VAL_PRED = """preds = _predict_from_waveforms(model, mel_transform, val_wavs)
        # ★ exp102: EMA evaluation
        ema_metrics = None
        if ema is not None and len(val_wavs) > 0:
            _bp, _bb = ema.apply_to(model)
            preds_ema = _predict_from_waveforms(model, mel_transform, val_wavs)
            ema.restore(model, _bp, _bb); del _bp, _bb
            ema_metrics = compute_metrics(Y_val, preds_ema, ns22_val, TAXON_MASKS)
            print(f"    EMA val_ns22={ema_metrics['non_s22_macro']:.4f} val_macro={ema_metrics['macro_auc_all']:.4f}")"""

assert OLD_VAL_PRED in content, "val predict anchor not found"
content = content.replace(OLD_VAL_PRED, NEW_VAL_PRED)
print("[OK] EMA val evaluation added")

# Patch: best_ns22 selection — also consider EMA score
OLD_BEST_NS22 = """if (not math.isnan(val_ns22)) and val_ns22 > best_ns22:
            best_ns22 = val_ns22
            state_to_save["best_ns22"] = best_ns22
            torch.save(state_to_save, out_dir / "ckpt_best_ns22.pth")"""
NEW_BEST_NS22 = """if (not math.isnan(val_ns22)) and val_ns22 > best_ns22:
            best_ns22 = val_ns22
            state_to_save["best_ns22"] = best_ns22
            torch.save(state_to_save, out_dir / "ckpt_best_ns22.pth")
        # ★ exp102: EMA best ckpt — overwrite ckpt_best_ns22.pth if EMA score better
        if ema_metrics is not None and (not math.isnan(ema_metrics['non_s22_macro'])) and ema_metrics['non_s22_macro'] > best_ns22:
            best_ns22 = ema_metrics['non_s22_macro']
            _bp, _bb = ema.apply_to(model)
            ema_state = {"epoch": epoch + 1, "best_ns22": best_ns22,
                         "best_macro": ema_metrics['macro_auc_all'],
                         "model_state": model.state_dict(),
                         "ema": True}
            torch.save(ema_state, out_dir / "ckpt_best_ns22.pth")
            ema.restore(model, _bp, _bb); del _bp, _bb
            print(f"    [BEST EMA] saved EMA ckpt as best_ns22 ({best_ns22:.4f})")"""

assert OLD_BEST_NS22 in content, "best_ns22 save anchor not found"
content = content.replace(OLD_BEST_NS22, NEW_BEST_NS22)
print("[OK] EMA best ckpt selection added")

# ============================================================
# Patch 9: Slug rename (exp100 -> exp102)
# ============================================================
slug_patches = [
    ("birdclef2026-exp102-l1-babych", "birdclef2026-exp102-l1-3fold-r3p"),
    ("birdclef2026 exp102 l1 babych (R3 from e17, mixup 1.0 + dp 0.15 + WRS)",
     "birdclef2026 exp102 l1 3fold r3p (R3 from e17, mixup 1.0 + dp 0.15 + WRS warm-up + EMA, 30 ep)"),
    ("exp100", "exp102"),
]
for old, new in slug_patches:
    if old in content:
        cnt = content.count(old)
        content = content.replace(old, new)
        print(f"[{cnt}x OK] {old[:60]!r}")

SRC.write_text(content, encoding="utf-8")
print(f"\n[OK] Saved {SRC} ({len(content)} chars)")
