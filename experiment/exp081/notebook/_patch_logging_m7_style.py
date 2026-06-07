"""Patch exp081 R1 and R2 NB to use M7-style train loop logging:
- per-step log (loss, bce, distill, lr)
- per-epoch: val_ns22 + val_macro + BEST tag
- taxon line (Insecta=X Reptilia=X Amphibia=X Mammalia=X Aves=X)
- class stats line (n/median/p25/p75/#>0.5/#>0.7/#>0.9/#perfect)
"""
import json
from pathlib import Path

NB_PATHS = [
    Path(__file__).with_name("nb_train_r1.ipynb"),
    Path(__file__).with_name("nb_train_r2.ipynb"),
]


def find_cell(cells, cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    return None


def patch_eval_cell(eval_cell):
    """Add M7-style helpers to eval cell."""
    src = "".join(eval_cell["source"])
    if "compute_per_species_auc" in src:
        print("  [eval] M7 helpers already present, skipping")
        return False

    helpers = '''

# ============================================================
# ★ M7-style helpers (per-species AUC, class stats, taxon str)
# ============================================================
def compute_per_species_auc(y_true, y_pred, class_mask=None):
    """Return list of (class_idx, AUC) for non-saturated species."""
    indices = range(y_true.shape[1]) if class_mask is None else class_mask
    aucs = []
    for c in indices:
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            auc = roc_auc_score(col, y_pred[:, c])
            aucs.append((int(c), float(auc)))
        except ValueError:
            continue
    return aucs


def macro_auc_from_list(aucs):
    import numpy as _np_m7
    return float(_np_m7.mean([a for _, a in aucs])) if len(aucs) > 0 else float("nan")


def lowest_k_mean(aucs, k=22):
    import numpy as _np_m7
    if len(aucs) == 0:
        return float("nan")
    sorted_aucs = sorted([a for _, a in aucs])
    k_eff = min(k, len(sorted_aucs))
    return float(_np_m7.mean(sorted_aucs[:k_eff]))


def class_stats_str(aucs):
    import numpy as _np_m7
    if len(aucs) == 0:
        return "n=0 median=nan p25=nan p75=nan #>0.5=0 #>0.7=0 #>0.9=0 #perfect=0"
    vals = _np_m7.array([a for _, a in aucs])
    return (f"n={len(vals)} median={_np_m7.median(vals):.3f} "
            f"p25={_np_m7.percentile(vals,25):.3f} p75={_np_m7.percentile(vals,75):.3f} "
            f"#>0.5={int((vals>0.5).sum())} #>0.7={int((vals>0.7).sum())} "
            f"#>0.9={int((vals>0.9).sum())} #perfect={int((vals>=1.0).sum())}")


def taxon_str_m7(y_true, y_pred, taxon_masks):
    import math as _math_m7
    parts = []
    for t in ["Insecta", "Reptilia", "Amphibia", "Mammalia", "Aves"]:
        mask = taxon_masks.get(t, [])
        if len(mask) == 0:
            parts.append(f"{t}=nan")
            continue
        aucs = compute_per_species_auc(y_true, y_pred, class_mask=list(mask))
        m = macro_auc_from_list(aucs)
        parts.append(f"{t}={m:.3f}" if not _math_m7.isnan(m) else f"{t}=nan")
    return "taxon: " + " ".join(parts)

print("OK M7 helpers ready (compute_per_species_auc, class_stats_str, taxon_str_m7)")
'''
    new_src = src.rstrip() + helpers
    eval_cell["source"] = new_src.splitlines(keepends=True)
    return True


def patch_train_loop_cell(train_cell):
    """Add M7-style epoch print with taxon + class stats + BEST tag."""
    src = "".join(train_cell["source"])
    if "taxon_str_m7" in src or "[M7-style log]" in src:
        print("  [train_loop] M7-style log already added, skipping")
        return False

    # Find the existing epoch summary print and replace with M7-style
    OLD_PRINT = '''    total_elapsed = time.time() - TRAIN_START
    print(f"\\n=== Ep {ep}/{N_TOTAL_EPOCHS}: "
          f"train_loss={train_loss_avg:.4f} cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
          f"val_ns22={val_metrics['ns22']:.4f} val_macro={val_metrics['macro']:.4f} "
          f"({ep_elapsed:.0f}s, total {total_elapsed/60:.1f}min) ===\\n")'''

    NEW_PRINT = '''    total_elapsed = time.time() - TRAIN_START
    # ★ M7-style log: epoch summary + taxon line + class stats + BEST tag
    cur_lr_log = optimizer.param_groups[0]["lr"]
    best_tag = ""
    if is_best_ns22 and is_best_macro:
        best_tag = "BEST(ns22+macro) "
    elif is_best_ns22:
        best_tag = "BEST(ns22) "
    elif is_best_macro:
        best_tag = "BEST(macro) "

    # Per-species AUC for class stats + taxon line (use ns22 mask = non-S22 evaluation set)
    try:
        _val_pred_blend_m7 = val_preds  # already computed above
        _val_true_m7 = Y_val
        if ns22_val is not None and ns22_val.any():
            _val_true_m7_ns22 = _val_true_m7[ns22_val]
            _val_pred_m7_ns22 = _val_pred_blend_m7[ns22_val]
        else:
            _val_true_m7_ns22 = _val_true_m7
            _val_pred_m7_ns22 = _val_pred_blend_m7
        _per_sp_aucs_m7 = compute_per_species_auc(_val_true_m7_ns22, _val_pred_m7_ns22)
        _cls_line_m7 = class_stats_str(_per_sp_aucs_m7)
        _tax_line_m7 = taxon_str_m7(_val_true_m7_ns22, _val_pred_m7_ns22, TAXON_MASKS)
    except Exception as _e_m7:
        _cls_line_m7 = f"(class stats err: {str(_e_m7)[:80]})"
        _tax_line_m7 = f"(taxon err: {str(_e_m7)[:80]})"

    print(f"\\n=== Ep {ep}/{N_TOTAL_EPOCHS}: "
          f"loss={train_loss_avg:.4f} (cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f}) "
          f"val_ns22={val_metrics['ns22']:.4f} val_macro={val_metrics['macro']:.4f} "
          f"{best_tag}lr={cur_lr_log:.2e} "
          f"({ep_elapsed:.0f}s, total {total_elapsed/60:.1f}min) ===")
    print(f"    {_tax_line_m7}")
    print(f"    class: {_cls_line_m7}\\n")'''

    if OLD_PRINT in src:
        new_src = src.replace(OLD_PRINT, NEW_PRINT)
        train_cell["source"] = new_src.splitlines(keepends=True)
        print("  [train_loop] Replaced epoch print with M7-style (taxon + class stats + BEST)")
        return True
    else:
        print(f"  [train_loop] WARN: OLD_PRINT not found, may need manual patch")
        return False


for nb_path in NB_PATHS:
    if not nb_path.exists():
        print(f"\nSKIP: {nb_path.name} not found")
        continue
    print(f"\n=== Patching {nb_path.name} ===")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    eval_cell = find_cell(nb["cells"], "eval")
    train_cell = find_cell(nb["cells"], "train_loop")

    e_changed = False
    t_changed = False
    if eval_cell is not None:
        e_changed = patch_eval_cell(eval_cell)
    if train_cell is not None:
        t_changed = patch_train_loop_cell(train_cell)

    if e_changed or t_changed:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  Saved {nb_path.name}")
    else:
        print(f"  No changes for {nb_path.name}")

print("\nDone.")
