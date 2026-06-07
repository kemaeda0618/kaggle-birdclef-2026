"""Augment train_r2 + eval cells to log per-taxon + per-class distribution metrics."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb")
nb = json.load(open(NB, encoding="utf-8"))


# (A) Augment eval cell: add per-class distribution helper
EVAL_AUGMENT = '''

def per_class_distribution(y_true, y_pred, mask=None, class_mask=None):
    """Return per-class AUC array and distribution stats."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            aucs.append(np.nan); continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            aucs.append(np.nan)
    aucs = np.array(aucs)
    valid = aucs[~np.isnan(aucs)]
    if len(valid) == 0:
        return {"n_valid": 0}
    return {
        "n_valid": len(valid),
        "median": float(np.median(valid)),
        "p25":    float(np.percentile(valid, 25)),
        "p75":    float(np.percentile(valid, 75)),
        "min":    float(valid.min()),
        "max":    float(valid.max()),
        "n_above_0.5": int((valid >= 0.5).sum()),
        "n_above_0.7": int((valid >= 0.7).sum()),
        "n_above_0.9": int((valid >= 0.9).sum()),
        "n_perfect_1.0": int((valid >= 0.999).sum()),
        "auc_array": aucs.tolist(),
    }


def rich_eval(y_true, y_pred, ns22_mask, taxon_masks):
    """Extended eval with per-class distribution + per-taxon AUC."""
    base = full_eval(y_true, y_pred, ns22_mask, taxon_masks)
    cls_dist = per_class_distribution(y_true, y_pred, mask=ns22_mask)
    base["per_class_dist"] = {k: v for k, v in cls_dist.items() if k != "auc_array"}
    base["per_class_auc"] = cls_dist.get("auc_array", [])
    return base
'''

for c in nb["cells"]:
    if c.get("id") == "eval":
        src = "".join(c["source"])
        if "rich_eval" not in src:
            # Append augmentation before final print
            if "print(\"OK eval helpers\")" in src:
                src = src.replace(
                    'print("OK eval helpers")',
                    EVAL_AUGMENT.lstrip() + '\nprint("OK eval helpers (with rich_eval + per_class_distribution)")',
                )
            else:
                src += EVAL_AUGMENT
            c["source"] = src.splitlines(keepends=True)
            print("[eval] OK augmented with rich_eval/per_class_distribution")
        else:
            print("[eval] already has rich_eval, skip")
        break


# (B) Augment train_r2 cell: use rich_eval, log per-taxon + per-class stats each epoch
for c in nb["cells"]:
    if c.get("id") == "train_r2":
        src = "".join(c["source"])

        # 1. Replace full_eval call with rich_eval (only the one in train_r2 loop)
        if "r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)" in src:
            src = src.replace(
                "r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)",
                "r = rich_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)",
            )
            print("[train_r2] OK full_eval -> rich_eval")

        # 2. Augment print statement (find existing format and expand)
        OLD_PRINT_PATTERN = '        print(f"\\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} "'
        if OLD_PRINT_PATTERN in src:
            # Find the multi-line print and replace with detailed version
            # Original: print(f"\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} val_ns22=... val_macro=... ===")
            # Insert extra per-taxon + per-class info after that
            new_print = '''        print(f"\\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} "'''
            assert new_print in src
            # Find the full multi-line print block end (look for ' ===")\\n')
            # Better: just insert NEW print AFTER the existing print
            anchor = '        print(f"\\n  === Ep {ep}/{N_TOTAL_EPOCHS}: loss={train_loss_avg:.4f} "'
            idx = src.find(anchor)
            if idx >= 0:
                # find end of print statement (closing of multi-line f-string)
                # look for ' ===")\n'
                end_marker_idx = src.find('===")', idx)
                if end_marker_idx >= 0:
                    insert_idx = src.find("\n", end_marker_idx) + 1
                    extra_logging = '''
        # ★ exp045: per-taxon + per-class distribution log
        if "per_taxon" in r:
            taxon_str = " ".join(f"{t}={a:.3f}" for t, a in r["per_taxon"].items())
            print(f"      taxon: {taxon_str}")
        if "per_class_dist" in r:
            d = r["per_class_dist"]
            if d.get("n_valid", 0) > 0:
                print(f"      class: n={d['n_valid']} median={d['median']:.3f} "
                      f"p25={d['p25']:.3f} p75={d['p75']:.3f} "
                      f"#>0.5={d['n_above_0.5']} #>0.7={d['n_above_0.7']} "
                      f"#>0.9={d['n_above_0.9']} #perfect={d['n_perfect_1.0']}")
'''
                    src = src[:insert_idx] + extra_logging + src[insert_idx:]
                    print("[train_r2] OK per-taxon + per-class log inserted")
                else:
                    print("[train_r2] WARN: ===\") marker not found, skip print augment")
            else:
                print("[train_r2] WARN: anchor not found")

        # 3. Save rich metrics to history dict
        # Find existing history.append and augment
        # Looking for the pattern: history.append({...,  "val_ns22": val_ns22, "val_macro": val_macro,...})
        if '"val_macro": val_macro,' in src:
            src = src.replace(
                '"val_macro": val_macro,',
                '"val_macro": val_macro,\n                        "per_taxon": r.get("per_taxon", {}),\n                        "per_class_dist": r.get("per_class_dist", {}),',
                1,
            )
            print("[train_r2] OK history augmented with per_taxon + per_class_dist")

        c["source"] = src.splitlines(keepends=True)
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {NB}")
