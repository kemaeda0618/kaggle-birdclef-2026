"""Patch exp102 v2: Fix EMA val eval bug + M7-style logging + Per-fold upload.

Issues fixed:
  1. EMA val eval indentation broken (val_ns22 not set when EMA None)
  2. `compute_metrics` → `full_eval` (function name bug)
  3. Add M7-style logging: per-species AUC list, class_stats, taxon str, BEST tag
  4. Add per-fold incremental upload (instead of end-of-all-folds)

Re-clones from exp100 + applies all patches fresh.

Run: python _patch_v2_logging_upload.py
Then: python _gen_nb_train_r3.py  ->  produces nb_train_r3.ipynb
"""
import io, sys, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
SRC = HERE / "_gen_nb_train_r3.py"

# Re-clone from exp100 for clean slate
EXP100_GEN = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp100\notebook\_gen_nb_train_r3.py")
shutil.copy(EXP100_GEN, SRC)
print(f"Re-cloned from exp100 ({EXP100_GEN.stat().st_size} bytes)")

content = SRC.read_text(encoding="utf-8")
print(f"Original size: {len(content)} chars\n")

# ============================================================
# Patch 1-5: Simple config patches
# ============================================================
print("=== Phase 1: Config patches ===")
config_patches = [
    ("FOLDS = [0]               # single fold 学習 (homogenize 回避)、ただし split は 5-fold で行う",
     "FOLDS = [0, 1, 2]         # ★ exp102: 3-fold ensemble"),
    ("N_TOTAL_EPOCHS = 20         # ★ exp100 Option C: 15 → 20、warmup 4/20 = 20% で適正",
     "N_TOTAL_EPOCHS = 30         # ★ exp102: 20 -> 30 (Babych late jump 完走)"),
    ("WARMUP_EPOCHS = 4         # ★ exp100: l1 で +1 ep (exp022 同設定)",
     "WARMUP_EPOCHS = 6         # ★ exp102: 4 -> 6 (epoch 比例)"),
    ("BATCH = 192", "BATCH = 640         # ★ exp102: 192 -> 640 (Blackwell 95.6 GB)"),
    ("NUM_WORKERS = 16        # Plan I: 12 → 16、data loading 並列度 +33%",
     "NUM_WORKERS = 24        # ★ exp102: 16 -> 24 (48 vCPU 適正)"),
]
for old, new in config_patches:
    assert old in content, f"MISS: {old[:60]!r}"
    content = content.replace(old, new)
    print(f"  [OK] {old[:60]}")

# Config addition: WRS warmup + EMA + STEP_LOG_INTERVAL + NS22_K
OLD_CFG = """NUM_WORKERS = 24        # ★ exp102: 16 -> 24 (48 vCPU 適正)
PERSISTENT_WORKERS = True"""
NEW_CFG = """NUM_WORKERS = 24        # ★ exp102: 16 -> 24 (48 vCPU 適正)
PERSISTENT_WORKERS = True

# ★ exp102: WRS warm-up + EMA + M7 logging config
WRS_WARMUP_EPOCHS = 5    # ep 1-5: uniform -> inv-freq gradient transition
USE_EMA = True
EMA_DECAY = 0.999
STEP_LOG_INTERVAL = 50   # batch step log frequency
NS22_K = 22              # lowest 22 species AUC for non_s22_macro"""
assert OLD_CFG in content
content = content.replace(OLD_CFG, NEW_CFG)
print("  [OK] WRS warmup + EMA + log config added")

# ============================================================
# Patch 6: WRS warm-up — pre-compute uniform/effective per epoch
# ============================================================
print("\n=== Phase 2: WRS warm-up ===")
OLD_WRS = "smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch, focal_weights=_focal_weights)"
NEW_WRS = """# ★ exp102: WRS warm-up — uniform -> inv-freq gradient (ep 1 uniform, ep WRS_WARMUP_EPOCHS+ pure inv-freq)
        _wrs_alpha = min(1.0, (epoch + 1) / max(WRS_WARMUP_EPOCHS, 1))
        _focal_uniform = np.full(len(_focal_weights), 1.0 / len(_focal_weights), dtype=np.float64)
        _focal_effective = (1.0 - _wrs_alpha) * _focal_uniform + _wrs_alpha * _focal_weights
        _focal_effective = _focal_effective / _focal_effective.sum()
        if epoch < WRS_WARMUP_EPOCHS:
            print(f"  [WRS warm-up] ep {epoch+1}: alpha={_wrs_alpha:.2f}, max/min ratio={_focal_effective.max()/max(_focal_effective.min(),1e-12):.1f}")
        smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch, focal_weights=_focal_effective)"""
assert OLD_WRS in content
content = content.replace(OLD_WRS, NEW_WRS)
print("  [OK] WRS warm-up logic")

# ============================================================
# Patch 7: M7 logging helpers — add to eval cell
# ============================================================
print("\n=== Phase 3: M7 logging helpers ===")
OLD_EVAL_END = 'print("OK eval helpers")'
NEW_EVAL_END = '''# ★ exp102: M7-style per-species + class stats + taxon line
def compute_per_species_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col): continue
        try:
            aucs.append((int(c), float(roc_auc_score(col, y_pred[:, c]))))
        except ValueError: continue
    return aucs


def class_stats_str(aucs):
    if len(aucs) == 0:
        return "n=0 median=nan p25=nan p75=nan #>0.5=0 #>0.7=0 #>0.9=0 #perfect=0"
    vals = np.array([a for _, a in aucs])
    return (f"n={len(vals)} median={np.median(vals):.3f} p25={np.percentile(vals,25):.3f} "
            f"p75={np.percentile(vals,75):.3f} "
            f"#>0.5={int((vals>0.5).sum())} #>0.7={int((vals>0.7).sum())} "
            f"#>0.9={int((vals>0.9).sum())} #perfect={int((vals>=1.0).sum())}")


def taxon_str(y_true, y_pred, ns22_mask, taxon_masks):
    parts = []
    for t in ["Insecta", "Reptilia", "Amphibia", "Mammalia", "Aves"]:
        mask = taxon_masks.get(t, np.array([], dtype=int))
        if len(mask) == 0:
            parts.append(f"{t}=nan"); continue
        aucs = compute_per_species_auc(y_true, y_pred, mask=ns22_mask, class_mask=mask)
        m = float(np.mean([a for _, a in aucs])) if aucs else float("nan")
        parts.append(f"{t}={m:.3f}" if not np.isnan(m) else f"{t}=nan")
    return "taxon: " + " ".join(parts)


print("OK eval helpers (with M7 logging)")'''
assert OLD_EVAL_END in content
content = content.replace(OLD_EVAL_END, NEW_EVAL_END)
print("  [OK] M7 helpers added")

# ============================================================
# Patch 8: EMA class — inject before train_r2
# ============================================================
print("\n=== Phase 4: EMA class ===")
EMA_CLASS = """
# ★ exp102: EMA (Exponential Moving Average) weight averaging
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.shadow_buf = {n: b.detach().clone() for n, b in model.named_buffers()}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for n, b in model.named_buffers():
            if n in self.shadow_buf and b.dtype.is_floating_point:
                self.shadow_buf[n].mul_(self.decay).add_(b.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
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
OLD_DEF = "def train_r2(fold_k, pseudo_meta_df, Y_pseudo, out_dir):"
assert OLD_DEF in content
content = content.replace(OLD_DEF, EMA_CLASS.lstrip() + "\n\n" + OLD_DEF)
print("  [OK] EMA class injected")

OLD_MAKE = """model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)"""
NEW_MAKE = """model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None  # ★ exp102: EMA init"""
assert OLD_MAKE in content
content = content.replace(OLD_MAKE, NEW_MAKE)
print("  [OK] EMA init in train_r2")

OLD_STEP = """scaler.step(optimizer)
            scaler.update()
            scheduler.step()"""
NEW_STEP = """scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)  # ★ exp102: EMA update"""
assert OLD_STEP in content
content = content.replace(OLD_STEP, NEW_STEP)
print("  [OK] EMA update after opt step")

# ============================================================
# Patch 9: REWRITE val eval block (fix bug + add M7 logging + EMA val + BEST tag)
# ============================================================
print("\n=== Phase 5: Val eval + M7 logging + best ckpt ===")
# Find original val eval + print + ckpt save block
OLD_VAL_BLOCK = '''        if len(val_wavs) > 0:
            val_preds = _predict_from_waveforms(model, mel_transform, val_wavs)
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            val_ns22 = r["non_s22_macro"]
            val_macro = r["macro_auc_all"]
        else:
            val_ns22 = val_macro = float("nan")

        ep_elapsed = time.time() - ep_start
        epoch_times.append(ep_elapsed)
        ep = epoch + 1

        state_to_save = {
            "epoch": ep, "fold": fold_k, "round": "r2",
            "n_total_epochs": N_TOTAL_EPOCHS,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_ns22": best_ns22, "best_macro": best_macro,
        }
        torch.save(state_to_save, out_dir / "ckpt_latest.pth")
        if (not math.isnan(val_ns22)) and val_ns22 > best_ns22:
            best_ns22 = val_ns22
            state_to_save["best_ns22"] = best_ns22
            torch.save(state_to_save, out_dir / "ckpt_best_ns22.pth")
        if (not math.isnan(val_macro)) and val_macro > best_macro:
            best_macro = val_macro
            state_to_save["best_macro"] = best_macro
            torch.save(state_to_save, out_dir / "ckpt_best_macro.pth")

        history.append({"epoch": ep, "train_loss": round(train_loss_avg, 5),
                        "cls_loss": round(cls_loss_avg, 5),
                        "dist_loss": round(dist_loss_avg, 5),
                        "val_ns22": val_ns22, "val_macro": val_macro,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_sec": round(ep_elapsed, 1)})
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        total_elapsed = time.time() - SESSION_START
        print(f"\\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} "
              f"cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} "
              f"({ep_elapsed:.0f}s, session {total_elapsed/60:.1f}min) ===\\n")
        gc.collect()
        torch.cuda.empty_cache()'''

NEW_VAL_BLOCK = '''        # ★ exp102: val eval (regular + EMA) with M7-style logging
        val_ns22 = val_macro = float("nan")
        val_ns22_ema = val_macro_ema = float("nan")
        per_species = []
        tax_line = "taxon: (n/a)"
        cls_line = "n=0"
        best_source = "reg"  # "reg" | "ema"

        if len(val_wavs) > 0:
            val_preds = _predict_from_waveforms(model, mel_transform, val_wavs)
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            val_ns22 = r["non_s22_macro"]; val_macro = r["macro_auc_all"]
            per_species = compute_per_species_auc(Y_val, val_preds, mask=ns22_val)
            tax_line = taxon_str(Y_val, val_preds, ns22_val, TAXON_MASKS)
            cls_line = class_stats_str(per_species)

        # EMA eval
        if ema is not None and len(val_wavs) > 0:
            _bp, _bb = ema.apply_to(model)
            try:
                val_preds_ema = _predict_from_waveforms(model, mel_transform, val_wavs)
                r_ema = full_eval(Y_val, val_preds_ema, ns22_val, TAXON_MASKS)
                val_ns22_ema = r_ema["non_s22_macro"]; val_macro_ema = r_ema["macro_auc_all"]
                per_species_ema = compute_per_species_auc(Y_val, val_preds_ema, mask=ns22_val)
                tax_line_ema = taxon_str(Y_val, val_preds_ema, ns22_val, TAXON_MASKS)
                cls_line_ema = class_stats_str(per_species_ema)
            finally:
                ema.restore(model, _bp, _bb); del _bp, _bb

        ep_elapsed = time.time() - ep_start
        epoch_times.append(ep_elapsed)
        ep = epoch + 1

        # Decide best ns22 source between regular and EMA
        cand_ns22 = val_ns22
        cand_source = "reg"
        if not math.isnan(val_ns22_ema) and (math.isnan(cand_ns22) or val_ns22_ema > cand_ns22):
            cand_ns22 = val_ns22_ema; cand_source = "ema"

        is_best_ns22 = (not math.isnan(cand_ns22)) and (cand_ns22 > best_ns22)
        is_best_macro = (not math.isnan(val_macro)) and (val_macro > best_macro)

        state_to_save = {
            "epoch": ep, "fold": fold_k, "round": "r2",
            "n_total_epochs": N_TOTAL_EPOCHS,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_ns22": best_ns22, "best_macro": best_macro,
        }
        torch.save(state_to_save, out_dir / "ckpt_latest.pth")

        # Save best ckpt — use EMA weights if EMA score better
        if is_best_ns22:
            best_ns22 = cand_ns22
            best_source = cand_source
            if best_source == "ema":
                _bp, _bb = ema.apply_to(model)
                ema_state = {"epoch": ep, "fold": fold_k, "round": "r2",
                             "best_ns22": best_ns22, "best_macro": val_macro_ema,
                             "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                             "ema": True}
                torch.save(ema_state, out_dir / "ckpt_best_ns22.pth")
                ema.restore(model, _bp, _bb); del _bp, _bb
            else:
                state_to_save["best_ns22"] = best_ns22
                torch.save(state_to_save, out_dir / "ckpt_best_ns22.pth")

        if is_best_macro:
            best_macro = val_macro
            state_to_save["best_macro"] = best_macro
            torch.save(state_to_save, out_dir / "ckpt_best_macro.pth")

        history.append({"epoch": ep, "train_loss": round(train_loss_avg, 5),
                        "cls_loss": round(cls_loss_avg, 5),
                        "dist_loss": round(dist_loss_avg, 5),
                        "val_ns22": val_ns22, "val_macro": val_macro,
                        "val_ns22_ema": val_ns22_ema, "val_macro_ema": val_macro_ema,
                        "best_ns22": best_ns22, "best_source": best_source,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_sec": round(ep_elapsed, 1)})
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        total_elapsed = time.time() - SESSION_START
        best_tag = "BEST " if is_best_ns22 else ""
        cur_lr = optimizer.param_groups[0]["lr"]

        # === M7-style epoch summary ===
        print(f"\\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} "
              f"(cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f}) "
              f"val_ns22={val_ns22:.4f} val_macro={val_macro:.4f} "
              f"{best_tag}lr={cur_lr:.2e} ({ep_elapsed:.0f}s, session {total_elapsed/60:.1f}min) ===")
        if not math.isnan(val_ns22_ema):
            ema_best_tag = "BEST " if (is_best_ns22 and best_source == "ema") else ""
            print(f"    EMA  val_ns22={val_ns22_ema:.4f} val_macro={val_macro_ema:.4f} {ema_best_tag}")
        print(f"    {tax_line}")
        print(f"    class: {cls_line}")
        if not math.isnan(val_ns22_ema):
            print(f"    EMA  {tax_line_ema}")
            print(f"    EMA class: {cls_line_ema}")
        if is_best_ns22:
            print(f"    [BEST] ns22 updated to {best_ns22:.4f} (source: {best_source})")

        gc.collect()
        torch.cuda.empty_cache()'''

assert OLD_VAL_BLOCK in content, "val eval block anchor not found"
content = content.replace(OLD_VAL_BLOCK, NEW_VAL_BLOCK)
print("  [OK] Val eval block REWRITTEN (bug fix + M7 logging + EMA + BEST tag)")

# ============================================================
# Phase 6: SKIPPED — per-fold upload disabled per user request
# (use existing final batch upload at end of fold loop)
# ============================================================
print("\n=== Phase 6: per-fold upload SKIPPED (final batch upload only) ===")

# ============================================================
# Patch 11: Slug rename + exp100 -> exp102 + cosmetic
# ============================================================
print("\n=== Phase 7: Slug + naming ===")
# First all exp100 -> exp102
n_blanket = content.count("exp100")
content = content.replace("exp100", "exp102")
print(f"  [{n_blanket}x OK] exp100 -> exp102 blanket")

# Then fix the slug (which is now wrong)
slug_patches = [
    ("birdclef2026-exp102-l1-babych", "birdclef2026-exp102-l1-3fold-r3p"),
    ("birdclef2026 exp102 l1 babych (R3 from e17, mixup 1.0 + dp 0.15 + WRS)",
     "birdclef2026 exp102 l1 3fold r3p (mixup 1.0 + dp 0.15 + WRS warm-up + EMA, 30 ep)"),
]
for old, new in slug_patches:
    if old in content:
        cnt = content.count(old)
        content = content.replace(old, new)
        print(f"  [{cnt}x OK] slug: {old[:60]}")

# Save
SRC.write_text(content, encoding="utf-8")
print(f"\n[OK] Saved {SRC} ({len(content)} chars)")
