"""Apply user's request:
1. BATCH: 64 → 128
2. LR: 3e-4 → 4.2e-4 (sqrt scale = 3e-4 * sqrt(2))
3. N_STEPS_PER_EP: restore original sum-based (steps preserved)
4. Keep LRU cache (helps either way)

Time estimate (BATCH=128, sum-based, 20 ep):
- N_STEPS_PER_EP = sum(SIZES)/128 = 178K/128 ≈ 1390 steps/ep
- Per step at BATCH=128: ~1.2-1.5s (LRU cache helps)
- Per ep: 28-35 min
- 20 ep: 9-12h
- ★ 1 session (11h limit) ギリギリ、time-aware self-stop で safely 中断する設計
"""
import json
from pathlib import Path
import ast

P = Path("experiment/exp083/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

# Fix 1: Update config — BATCH=128, LR=4.2e-4
if "config" in cell_map:
    i, c = cell_map["config"]
    src = "".join(c["source"])
    # BATCH change
    src = src.replace(
        "BATCH = 64              # ★ T4 16GB safe",
        "BATCH = 128             # ★ user request: steps preserved with bigger batch (T4 16GB tight)"
    )
    # LR change (sqrt scaling for convnext)
    src = src.replace(
        "LR = 3e-4\n",
        "LR = 4.2e-4   # ★ sqrt scaling: 3e-4 × sqrt(2) for BATCH 64→128 (convnext は linear/sqrt 適用可)\n"
    )
    c["source"] = src.splitlines(keepends=True)
    print(f"OK Fixed config: BATCH=128, LR=4.2e-4")

# Fix 2: Restore sum-based N_STEPS_PER_EP in resume_or_init
if "resume_or_init" in cell_map:
    i, c = cell_map["resume_or_init"]
    src = "".join(c["source"])
    old = """# ★ R2 fix: cap N_STEPS_PER_EP to focal-based (pseudo_sc adds 127K samples, would inflate to 2700 steps)
focal_size = SIZES[NAMES.index('focal')]
focal_share = SHARES.get('focal', 0.70)
N_STEPS_PER_EP = max(500, int(focal_size / BATCH / focal_share))"""
    new = """# ★ User request: sum-based steps preserved (BATCH=128 → ~1390 steps/ep, fits in 1 session)
N_STEPS_PER_EP = max(500, int(sum(SIZES) / BATCH))"""
    if old in src:
        src = src.replace(old, new)
        c["source"] = src.splitlines(keepends=True)
        print(f"OK Restored sum-based N_STEPS_PER_EP")
    else:
        print(f"WARN: focal-based pattern not found, checking alternatives...")
        # Try alternative pattern
        for alt_old in [
            "focal_size = SIZES[NAMES.index('focal')]",
            "N_STEPS_PER_EP = max(500, int(focal_size / BATCH / focal_share))",
        ]:
            if alt_old in src:
                print(f"  Found: {alt_old}")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Verify changes
nb = json.loads(P.read_text(encoding="utf-8"))
cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

# Show key changes
for cid in ["config", "resume_or_init"]:
    if cid in cell_map:
        i, c = cell_map[cid]
        src = "".join(c["source"])
        for ln in src.splitlines():
            if any(kw in ln for kw in ["BATCH = ", "LR =", "N_STEPS_PER_EP", "focal_size", "sum-based"]):
                print(f"  [{cid}] {ln.strip()[:150]}")

# Syntax check
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
