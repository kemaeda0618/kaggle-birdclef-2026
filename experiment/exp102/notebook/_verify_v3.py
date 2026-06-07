"""Verify exp102 v3: all patches (v1+v2+v3 early stop)."""
import json, io, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_r3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])}")
print()

checks = [
    # ---- v1 + v2 base ----
    ('FOLDS = [0, 1, 2]',                       "FOLDS = [0, 1, 2]" in all_src),
    ('N_TOTAL_EPOCHS = 30',                     "N_TOTAL_EPOCHS = 30" in all_src),
    ('WARMUP_EPOCHS = 6',                       "WARMUP_EPOCHS = 6" in all_src),
    ('BATCH = 640',                             "BATCH = 640" in all_src),
    ('NUM_WORKERS = 24',                        "NUM_WORKERS = 24" in all_src),
    ('WRS_WARMUP_EPOCHS = 5',                   "WRS_WARMUP_EPOCHS = 5" in all_src),
    ('USE_EMA = True',                          "USE_EMA = True" in all_src),
    ('EMA_DECAY = 0.999',                       "EMA_DECAY = 0.999" in all_src),
    ('class ModelEMA',                          "class ModelEMA:" in all_src),
    ('EMA update after opt',                    "ema.update(model)" in all_src),
    ('EMA val eval',                            "val_preds_ema = _predict_from_waveforms" in all_src),
    ('M7 helpers (taxon_str, class_stats)',     "def taxon_str" in all_src and "def class_stats_str" in all_src),
    ('BEST tag in print',                       'best_tag = "BEST "' in all_src),
    ('Bug fix: full_eval (no compute_metrics)', "full_eval(Y_val" in all_src and "compute_metrics(" not in all_src),
    # ---- v3 early stopping ----
    ('v3: EARLY_STOP_PATIENCE = 10',            "EARLY_STOP_PATIENCE = 10" in all_src),
    ('v3: MIN_EPOCH_BEFORE_STOP = 15',          "MIN_EPOCH_BEFORE_STOP = 15" in all_src),
    ('v3: no_improve_count init = 0',           "no_improve_count = 0" in all_src),
    ('v3: NaN guard',                           "if not math.isnan(cand_ns22)" in all_src),
    ('v3: reset on is_best_ns22',               "if is_best_ns22:\n                no_improve_count = 0" in all_src),
    ('v3: increment else',                      "no_improve_count += 1" in all_src),
    ('v3: print [es] status',                   "[es] no_improve=" in all_src),
    ('v3: stop condition has min_epoch guard',  "(epoch + 1) >= MIN_EPOCH_BEFORE_STOP and no_improve_count >= EARLY_STOP_PATIENCE" in all_src),
    ('v3: break statement',                     "[EARLY STOP] ep" in all_src),
    ('v3: cleanup before break',                "gc.collect()\n                torch.cuda.empty_cache()\n                break" in all_src),
    # ---- structural ----
    ('Babych spec preserved (MIXUP 1.0)',       "MIXUP_ALPHA = 1.0" in all_src),
    ('Babych spec preserved (dp 0.15)',         "drop_path_rate=0.15" in all_src),
    ('SHARES preserved',                        '"focal": 0.60' in all_src and '"pseudo_sc": 0.30' in all_src),
    ('Slug: exp102-l1-3fold-r3p',               "birdclef2026-exp102-l1-3fold-r3p" in all_src),
]
print("=== Invariants ===")
n_fail = 0
for n, ok in checks:
    if not ok: n_fail += 1
    print(f"  {'[OK]' if ok else '[FAIL]'} {n}")
print(f"\nFailed: {n_fail}/{len(checks)}")

# ---- Critical structural check: counter logic placement ----
print("\n=== Structural check (early stop logic ordering) ===")
# Find the early stop block and verify it's AFTER is_best_ns22 update but BEFORE next epoch
# Verify break is inside inner loop (epoch loop), not outer (fold loop)
es_block_match = re.search(
    r"if not math\.isnan\(cand_ns22\):.*?break",
    all_src, re.DOTALL
)
if es_block_match:
    es_block = es_block_match.group(0)
    print("[OK] Early stop block found, content:")
    for ln in es_block.split("\n")[:15]:
        print(f"    {ln}")
else:
    print("[FAIL] Early stop block pattern not found")

# Count breaks in epoch loop area
# Look for: "for epoch in range(N_TOTAL_EPOCHS):" ... "[EARLY STOP]" ... "break"
print()
epoch_loop_match = re.search(
    r"for epoch in range\(N_TOTAL_EPOCHS\):.*?(?=print\(f\"  \[R2 done\])",
    all_src, re.DOTALL
)
if epoch_loop_match:
    n_breaks = epoch_loop_match.group(0).count("break")
    print(f"[INFO] breaks inside epoch loop: {n_breaks}")
    # Should be 2: one existing (time budget) + one new (early stop)
else:
    print("[WARN] could not isolate epoch loop")

print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}-{c.get('id','?')}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} id={c.get('id','?')}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")

# ---- Simulation: re-play exp100 trajectory and check no early stop ----
print("\n=== Simulation: exp100 trajectory with patience=10, min_epoch=15 ===")
exp100_val = [0.6224, 0.7300, 0.8040, 0.8655, 0.8781, 0.9041, 0.9039, 0.9037,
              0.9119, 0.9154, 0.9226, 0.9218, 0.9177, 0.9202, 0.9218,
              0.9220, 0.9229, 0.9242, 0.9250, 0.9251]
best = -1.0
no_imp = 0
stopped_at = None
for ep, val in enumerate(exp100_val, 1):
    is_best = val > best
    if is_best:
        best = val
        no_imp = 0
    else:
        no_imp += 1
    if ep >= 15 and no_imp >= 10:
        stopped_at = ep
        break
    print(f"  ep{ep:2d}: val={val:.4f} {'BEST' if is_best else f'(no_imp={no_imp})'}")
print(f"\n  -> stopped at: {stopped_at or 'completed all epochs'}")
assert stopped_at is None, f"BUG: exp100 trajectory would have triggered early stop at ep {stopped_at}"
print("  [VERIFIED] exp100 pattern NOT killed by patience=10/min_epoch=15")
