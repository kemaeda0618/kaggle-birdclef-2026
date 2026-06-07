"""Add rich log to exp049 and exp050 (per-taxon AUC + class distribution).

Modeled after Colab log format used in exp045/046:
  taxon: Aves=0.839 Amphibia=0.944 Insecta=0.954 Mammalia=1.000 Reptilia=0.958
  class: n=29 median=0.972 p25=0.907 p75=1.000 #>0.5=28 ...
"""
import json
from pathlib import Path

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb"),
]

# Insert LABEL_TO_CLASS in data_load cell
DATA_LOAD_INSERT = """LABEL_TO_CLASS = dict(zip(tax['primary_label'], tax['class_name']))
print(f"  classes: {set(LABEL_TO_CLASS.values())}")
"""

# New rich_evaluate function
RICH_EVAL = '''# ============================================================
# Cell 9: rich_evaluate (val_ns22 + val_macro + per-taxon AUC + class distribution)
# ============================================================
from sklearn.metrics import roc_auc_score
from torch.amp import autocast

@torch.no_grad()
def rich_evaluate(model, mel_ex, loader):
    \"\"\"Return val_ns22, val_macro, taxon_aucs, class_stats.

    val_ns22: per-class AUC mean for classes with >=22 positive samples (BC2025 流儀)
    val_macro: per-class AUC simple mean
    taxon_aucs: dict {Aves, Amphibia, Insecta, Mammalia, Reptilia} -> mean AUC
    class_stats: distribution stats over per-class AUCs
    \"\"\"
    model.eval()
    all_logits, all_labels = [], []
    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        mel = mel_ex(wav)
        with autocast("cuda", dtype=torch.float16):
            logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    # Per-class AUC + positive count
    per_class = []  # (primary_label, auc, n_pos)
    for c in range(labels.shape[1]):
        n_pos = int(labels[:, c].sum())
        if n_pos >= 2 and n_pos < labels.shape[0]:
            try:
                auc = roc_auc_score(labels[:, c], logits[:, c])
                per_class.append((PRIMARY_LABELS[c], float(auc), n_pos))
            except Exception:
                pass

    # val_ns22 (n_pos >= 22 classes)
    aucs_n22 = [auc for _, auc, n in per_class if n >= 22]
    val_ns22 = float(np.mean(aucs_n22)) if aucs_n22 else 0.0

    # val_macro (all valid classes)
    aucs_all = [auc for _, auc, _ in per_class]
    val_macro = float(np.mean(aucs_all)) if aucs_all else 0.0

    # Per-taxon AUC
    taxon_aucs = {}
    for taxon in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        aucs_in = [auc for label, auc, _ in per_class if LABEL_TO_CLASS.get(label) == taxon]
        taxon_aucs[taxon] = float(np.mean(aucs_in)) if aucs_in else float("nan")

    # Class distribution stats (over per-class AUCs in val_ns22 subset)
    if aucs_n22:
        arr = np.array(aucs_n22)
        class_stats = {
            "n_valid": len(aucs_n22),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "n_above_05": int(np.sum(arr > 0.5)),
            "n_above_07": int(np.sum(arr > 0.7)),
            "n_above_09": int(np.sum(arr > 0.9)),
            "n_perfect": int(np.sum(arr >= 0.9999)),
        }
    else:
        class_stats = {"n_valid": 0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                       "n_above_05": 0, "n_above_07": 0, "n_above_09": 0, "n_perfect": 0}

    return val_ns22, val_macro, taxon_aucs, class_stats


# Backward-compatible alias for train_loop
def evaluate(model, mel_ex, loader):
    val_ns22, _, _, _ = rich_evaluate(model, mel_ex, loader)
    return val_ns22
'''

# Update train_loop val section to print rich log
NEW_TRAIN_LOOP_VAL_SECTION = '''    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    _p(f"[ep{epoch}] train done (loss={train_loss:.4f}), evaluating val...")
    val_ns22, val_macro, taxon_aucs, class_stats = rich_evaluate(model, mel_extractor, val_loader)
    lr_now = optimizer.param_groups[0]["lr"]
    elapsed_min = (time.time() - start_t) / 60
    ep_min = (time.time() - ep_start) / 60

    history["train_loss"].append(train_loss)
    history["val_ns22"].append(val_ns22)
    history["lr"].append(lr_now)
    history["elapsed_min"].append(elapsed_min)
    # Also save extras
    history.setdefault("val_macro", []).append(val_macro)
    history.setdefault("taxon_aucs", []).append(taxon_aucs)
    history.setdefault("class_stats", []).append(class_stats)

    is_best = val_ns22 > best_val
    if is_best:
        best_val = val_ns22
        torch.save(
            {"epoch": epoch, "val_ns22": val_ns22, "val_macro": val_macro,
             "taxon_aucs": taxon_aucs, "class_stats": class_stats,
             "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            },
            CFG.BEST_CKPT,
        )

    json.dump(history, open(CFG.HIST_JSON, "w"), indent=2)
    taxon_str = " ".join(f"{t}={v:.3f}" for t, v in taxon_aucs.items())
    cs = class_stats
    _p(f"=== Ep {epoch}/{CFG.EPOCHS}: loss={train_loss:.4f} val_ns22={val_ns22:.4f} val_macro={val_macro:.4f}"
       f" {'BEST' if is_best else ''} lr={lr_now:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min, nan_skips={nan_skip_count}) ===")
    _p(f"    taxon: {taxon_str}")
    _p(f"    class: n={cs['n_valid']} median={cs['median']:.3f} p25={cs['p25']:.3f} p75={cs['p75']:.3f}"
       f" #>0.5={cs['n_above_05']} #>0.7={cs['n_above_07']} #>0.9={cs['n_above_09']} #perfect={cs['n_perfect']}")'''

for NB in NBS:
    if not NB.exists():
        print(f"[SKIP] {NB.name}")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    n_changes = 0

    # 1. data_load: add LABEL_TO_CLASS
    for c in nb["cells"]:
        if c.get("id") == "data_load":
            src = "".join(c["source"])
            if "LABEL_TO_CLASS" not in src:
                # Add after PRIMARY_LABELS
                src = src.replace(
                    "LABEL_TO_IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}\n",
                    "LABEL_TO_IDX = {l: i for i, l in enumerate(PRIMARY_LABELS)}\n" + DATA_LOAD_INSERT
                )
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # 2. eval_fn cell -> replace with rich_evaluate
    for c in nb["cells"]:
        if c.get("id") == "eval_fn":
            c["source"] = RICH_EVAL.splitlines(keepends=True)
            n_changes += 1
            break

    # 3. train_loop: replace val section
    for c in nb["cells"]:
        if c.get("id") == "train_loop":
            src = "".join(c["source"])
            old_val_section = """    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    _p(f"[ep{epoch}] train done (loss={train_loss:.4f}), evaluating val...")
    val_ns22 = evaluate(model, mel_extractor, val_loader)
    lr_now = optimizer.param_groups[0]["lr"]
    elapsed_min = (time.time() - start_t) / 60
    ep_min = (time.time() - ep_start) / 60

    history["train_loss"].append(train_loss)
    history["val_ns22"].append(val_ns22)
    history["lr"].append(lr_now)
    history["elapsed_min"].append(elapsed_min)

    is_best = val_ns22 > best_val
    if is_best:
        best_val = val_ns22
        torch.save(
            {"epoch": epoch, "val_ns22": val_ns22,
             "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            },
            CFG.BEST_CKPT,
        )

    json.dump(history, open(CFG.HIST_JSON, "w"), indent=2)
    _p(f"=== Ep {epoch}/{CFG.EPOCHS}: loss={train_loss:.4f} val_ns22={val_ns22:.4f}"
       f" {'BEST' if is_best else ''} lr={lr_now:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min, nan_skips={nan_skip_count}) ===\")"""

            if old_val_section in src:
                src = src.replace(old_val_section, NEW_TRAIN_LOOP_VAL_SECTION)
                n_changes += 1
            else:
                print(f"  [WARN {NB.parent.parent.name}/train_loop] val section pattern not exact, attempting fuzzy replace")
                # Fallback: find "val_ns22 = evaluate(" line and replace from there
                import re
                pattern = re.compile(
                    r"    train_loss = float\(np\.mean\(train_losses\)\) if train_losses else float\(\"nan\"\)\n.*?_p\(f\"=== Ep \{epoch\}/\{CFG\.EPOCHS\}.*?\) ===\"\)",
                    re.DOTALL
                )
                src = pattern.sub(NEW_TRAIN_LOOP_VAL_SECTION, src, count=1)
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{NB.parent.parent.name}] {n_changes} changes (rich log added)")

print("\n=== Verify ===")
for NB in NBS:
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "eval_fn":
            src = "".join(c["source"])
            if "rich_evaluate" in src and "taxon_aucs" in src:
                print(f"  {NB.parent.parent.name} [eval_fn] OK rich_evaluate added")
        if c.get("id") == "train_loop":
            src = "".join(c["source"])
            if "taxon_str" in src and "rich_evaluate(model" in src:
                print(f"  {NB.parent.parent.name} [train_loop] OK rich log integrated")
        if c.get("id") == "data_load":
            src = "".join(c["source"])
            if "LABEL_TO_CLASS" in src:
                print(f"  {NB.parent.parent.name} [data_load] OK LABEL_TO_CLASS added")
