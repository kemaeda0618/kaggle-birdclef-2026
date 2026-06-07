"""Patch v3: Add early stopping to exp102 v2 (patience=10, min_epoch=15).

Triggers:
  - per-fold scope only (inner epoch loop break, fold loop continues)
  - check at end of epoch (after val + ckpt save)
  - reset count when is_best_ns22=True (covers regular OR EMA improvement)
  - skip count update when val is NaN (defensive)
  - never stop before epoch 15 (protects Babych late jump pattern)

Run: python _patch_v3_early_stop.py
Then: python _gen_nb_train_r3.py
"""
import io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(__file__).with_name("_gen_nb_train_r3.py")
content = SRC.read_text(encoding="utf-8")
print(f"Original size: {len(content)} chars\n")

# ============================================================
# Patch A: Add config in Cell 3 (right after EMA_DECAY)
# ============================================================
OLD_CFG = """USE_EMA = True
EMA_DECAY = 0.999
STEP_LOG_INTERVAL = 50   # batch step log frequency
NS22_K = 22              # lowest 22 species AUC for non_s22_macro"""

NEW_CFG = """USE_EMA = True
EMA_DECAY = 0.999
STEP_LOG_INTERVAL = 50   # batch step log frequency
NS22_K = 22              # lowest 22 species AUC for non_s22_macro
# ★ exp102 v3: Early stopping config (Babych late jump 保護)
EARLY_STOP_PATIENCE = 10        # consecutive epochs without best_ns22 improvement
MIN_EPOCH_BEFORE_STOP = 15      # never stop before this epoch (protects late jump)"""

assert OLD_CFG in content, "config anchor not found"
assert content.count(OLD_CFG) == 1
content = content.replace(OLD_CFG, NEW_CFG)
print("[OK] Patch A: Early stop config added")

# ============================================================
# Patch B: Add no_improve_count init in train_r2 (alongside best_ns22)
# ============================================================
OLD_INIT = """    best_ns22 = -1.0
        best_macro = -1.0
        history = []
        epoch_times = []"""

# Check actual indentation in file
import re
# Try different indentation
for indent in ["    ", "\\t"]:
    candidate = OLD_INIT.replace("        ", "    ")  # fix indent
    if candidate in content:
        OLD_INIT = candidate
        break

# Direct search
OLD_INIT_PATTERNS = [
    "best_ns22 = -1.0\n    best_macro = -1.0\n    history = []\n    epoch_times = []",
    "best_ns22 = -1.0\n    best_macro = -1.0\n    history = []",
]
found = False
for pat in OLD_INIT_PATTERNS:
    if pat in content:
        OLD_INIT = pat
        found = True
        break
assert found, "best_ns22 init anchor not found"

# Compose NEW: add no_improve_count init right after best_macro
if "epoch_times = []" in OLD_INIT:
    NEW_INIT = """best_ns22 = -1.0
    best_macro = -1.0
    no_improve_count = 0          # ★ exp102 v3: early stop counter
    history = []
    epoch_times = []"""
else:
    NEW_INIT = """best_ns22 = -1.0
    best_macro = -1.0
    no_improve_count = 0          # ★ exp102 v3: early stop counter
    history = []"""

assert content.count(OLD_INIT) == 1
content = content.replace(OLD_INIT, NEW_INIT)
print("[OK] Patch B: no_improve_count init added")

# ============================================================
# Patch C: Add counter update + stop check at end of epoch loop
# Inject right before gc.collect() at end of epoch
# ============================================================
OLD_END = """        if is_best_ns22:
            print(f"    [BEST] ns22 updated to {best_ns22:.4f} (source: {best_source})")

        gc.collect()
        torch.cuda.empty_cache()"""

NEW_END = """        if is_best_ns22:
            print(f"    [BEST] ns22 updated to {best_ns22:.4f} (source: {best_source})")

        # ★ exp102 v3: Early stopping check
        # - Reset counter on improvement (regular OR EMA, captured by is_best_ns22)
        # - Skip update if val unavailable (NaN), to prevent false-triggers
        # - Never stop before MIN_EPOCH_BEFORE_STOP (Babych late jump 保護)
        # - Stop only when no_improve_count >= EARLY_STOP_PATIENCE
        if not math.isnan(cand_ns22):
            if is_best_ns22:
                no_improve_count = 0
            else:
                no_improve_count += 1
            print(f"    [es] no_improve={no_improve_count}/{EARLY_STOP_PATIENCE} (min_ep_to_stop={MIN_EPOCH_BEFORE_STOP})")
            if (epoch + 1) >= MIN_EPOCH_BEFORE_STOP and no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\\n  [EARLY STOP] ep {epoch+1}: no improvement for {no_improve_count} epochs "
                      f"(patience={EARLY_STOP_PATIENCE}, min_ep={MIN_EPOCH_BEFORE_STOP}) — abort fold {fold_k}, keep best ckpt\\n")
                gc.collect()
                torch.cuda.empty_cache()
                break

        gc.collect()
        torch.cuda.empty_cache()"""

assert OLD_END in content, "epoch end anchor not found"
assert content.count(OLD_END) == 1
content = content.replace(OLD_END, NEW_END)
print("[OK] Patch C: counter update + stop check added")

# ============================================================
# Save
# ============================================================
SRC.write_text(content, encoding="utf-8")
print(f"\n[OK] Saved {SRC} ({len(content)} chars)")
