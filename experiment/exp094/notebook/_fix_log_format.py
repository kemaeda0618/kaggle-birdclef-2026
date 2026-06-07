"""Quick fix exp094 NB to match M7 (exp076) log format spec.

Spec (memory: feedback_train_logging_format, 2026-05-24 user 指定):
- step log: ep batch loss cls emb kd lr (already in place ✅)
- epoch summary: + BEST marker
- taxon line: per-taxon (Insecta/Reptilia/Amphibia/Mammalia/Aves) macro AUC
- class line: n/median/p25/p75/#>0.5/#>0.7/#>0.9/#perfect

Also: comment out Kaggle push section (user: Colab で training、push 不要)
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# ============================================================
# Patch 1: Add helpers (class_stats_str + per_species computation)
# Insert into eval cell (where full_eval is defined)
# ============================================================
# Find the cell with full_eval and inject helpers before it
HELPER_INSERT_OLD = "def full_eval"
HELPER_INSERT_NEW = '''def _exp094_class_stats_str(aucs):
    """★ exp094: Class stats per memo (n/median/p25/p75/#>0.5/#>0.7/#>0.9/#perfect)."""
    if len(aucs) == 0:
        return "n=0 median=nan p25=nan p75=nan #>0.5=0 #>0.7=0 #>0.9=0 #perfect=0"
    vals = np.array(aucs)
    return (f"n={len(vals)} median={np.median(vals):.3f} p25={np.percentile(vals,25):.3f} "
            f"p75={np.percentile(vals,75):.3f} "
            f"#>0.5={int((vals>0.5).sum())} #>0.7={int((vals>0.7).sum())} "
            f"#>0.9={int((vals>0.9).sum())} #perfect={int((vals>=1.0).sum())}")


def _exp094_taxon_str(per_taxon_dict):
    """★ exp094: Taxon line per memo (Insecta/Reptilia/Amphibia/Mammalia/Aves)."""
    parts = []
    for t in ["Insecta", "Reptilia", "Amphibia", "Mammalia", "Aves"]:
        v = per_taxon_dict.get(t, float("nan"))
        if isinstance(v, dict):
            v = v.get("macro", float("nan"))
        parts.append(f"{t}={v:.3f}" if not np.isnan(v) else f"{t}=nan")
    return "taxon: " + " ".join(parts)


def _exp094_per_species_aucs(y_true, y_pred):
    """★ exp094: Per-species AUC list for class_stats."""
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            from sklearn.metrics import roc_auc_score as _rcs
            aucs.append(float(_rcs(col, y_pred[:, c])))
        except ValueError:
            continue
    return aucs


def full_eval'''

# ============================================================
# Patch 2: Update epoch summary print (add BEST marker + taxon + class lines)
# ============================================================
PRINT_OLD = '''    total_elapsed = time.time() - TRAIN_START
    print(f"\\n=== Ep {ep}/{N_TOTAL_EPOCHS}: "
          f"train_loss={train_loss_avg:.4f} cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
          f"val_ns22={val_metrics['ns22']:.4f} val_macro={val_metrics['macro']:.4f} "
          f"({ep_elapsed:.0f}s, total {total_elapsed/60:.1f}min) ===\\n")'''

PRINT_NEW = '''    total_elapsed = time.time() - TRAIN_START
    # ★ exp094: BEST marker + taxon line + class stats per memo spec
    _best_tag = "BEST " if (is_best_ns22 or is_best_macro) else ""
    _tax_line = _exp094_taxon_str(val_metrics.get("per_taxon", {}))
    _per_sp = _exp094_per_species_aucs(Y_val, val_preds) if len(val_wavs) > 0 else []
    _cls_line = _exp094_class_stats_str(_per_sp)
    print(f"\\n=== Ep {ep}/{N_TOTAL_EPOCHS}: "
          f"train_loss={train_loss_avg:.4f} cls={cls_loss_avg:.4f} dist={dist_loss_avg:.4f} "
          f"val_ns22={val_metrics['ns22']:.4f} val_macro={val_metrics['macro']:.4f} "
          f"{_best_tag}({ep_elapsed:.0f}s, total {total_elapsed/60:.1f}min) ===")
    print(f"    {_tax_line}")
    print(f"    class: {_cls_line}")
    if is_best_ns22:
        print(f"    BEST ns22 saved val_ns22={val_metrics['ns22']:.4f}")
    if is_best_macro:
        print(f"    BEST macro saved val_macro={val_metrics['macro']:.4f}")
    print()'''

# ============================================================
# Patch 3: Comment out Kaggle push section (user: push 不要)
# ============================================================
# Find cells doing api.dataset_create_version / api.dataset_create_new
# We'll mark the cell with a skip flag at the top
KAGGLE_OLD = "api.dataset_view(f\"{USER}/{SLUG}\")"
KAGGLE_NEW = '''# ★ exp094: Push skipped (user: Colab で training、Kaggle push 不要)
    raise RuntimeError("exp094: Kaggle push disabled (manual Drive download instead). Comment out this line to enable.")
    api.dataset_view(f"{USER}/{SLUG}")'''

# ============================================================
# Apply patches
# ============================================================
patches = [
    (HELPER_INSERT_OLD, HELPER_INSERT_NEW),
    (PRINT_OLD, PRINT_NEW),
    (KAGGLE_OLD, KAGGLE_NEW),
]

print(f"Applying {len(patches)} log format patches...")
for old, new in patches:
    found = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if old in src:
            new_src = src.replace(old, new)
            c["source"] = new_src.splitlines(keepends=True)
            found = True
            print(f"  [OK] {old[:80]!r}")
            break
    if not found:
        print(f"  [MISS] {old[:80]!r}")

# Save
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB}")

# Verify
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verification ===")
checks = [
    ("_exp094_class_stats_str defined", "def _exp094_class_stats_str" in all_src),
    ("_exp094_taxon_str defined", "def _exp094_taxon_str" in all_src),
    ("_exp094_per_species_aucs defined", "def _exp094_per_species_aucs" in all_src),
    ("BEST marker in print", '"BEST "' in all_src),
    ("taxon line printed", "_exp094_taxon_str(val_metrics" in all_src),
    ("class line printed", "_exp094_class_stats_str(_per_sp)" in all_src),
    ("BEST ns22 saved log", "BEST ns22 saved" in all_src),
    ("BEST macro saved log", "BEST macro saved" in all_src),
    ("Kaggle push disabled", "exp094: Kaggle push disabled" in all_src),
    # Sanity
    ("USE_ALI_KD intact", "USE_ALI_KD     = True" in all_src),
    ("T_DISTILL = 1.0 intact", "T_DISTILL      = 1.0" in all_src),
    ("ALPHA_LOGIT = 0.1 intact", "ALPHA_LOGIT    = 0.1" in all_src),
    ("KL loss code intact", "distill_logit_loss = F.binary_cross_entropy_with_logits" in all_src),
    ("Step log has kd term", "kd={distill_logit_loss.item()" in all_src),
]
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok: all_ok = False

# Compile
print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
