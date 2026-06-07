"""Build exp092 NB by cloning exp090 + lambda_prior 0.65 → 0.75 sweep.

exp092 = exp090 with single 1-line change:
  - apply_prior(...lambda_prior=0.65) → apply_prior(...lambda_prior=0.75)
  - (both call sites: line 2084 + line 2136)

Purpose: Value sweep on lambda_prior to test elasticity.
- anthony 0.950 NB uses 0.65 (up from 0.4 default)
- 我々 exp090 inherited 0.65 without sweep
- Try 0.75 → +0.001-0.003 expected if elasticity remains
"""
import json, io, sys, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp090\notebook\nb_exp090_tuned_tax.ipynb")
DST = Path(__file__).with_name("nb_exp092_lambda075.ipynb")

print(f"[clone] {SRC.name} -> {DST.name}")
shutil.copy(SRC, DST)

nb = json.loads(DST.read_text(encoding="utf-8"))

# ============================================================
# Patches
# ============================================================
patches = [
    # 1. lambda_prior in 2 call sites
    ("    lambda_prior=0.65,",
     "    lambda_prior=0.75,  # ★ exp092: sweep 0.65 → 0.75"),

    # 2. Header text: title line
    ("# exp090 — exp078 (LB 0.950) + lambda_prior 0.65 + rank_aware power 0.6 + TAX_SMOOTHING",
     "# exp092 — exp090 (LB 0.951) base + **lambda_prior sweep 0.65 → 0.75** (1 行変更 ablation)"),

    # 3. Header table row
    ("| `lambda_prior` (Hour/Site prior) | 0.4 (default) | **0.65** ★ | anthony Model_51 |",
     "| `lambda_prior` (Hour/Site prior) | 0.65 (exp090) | **0.75** ★ exp092 | anthony 0.4→0.65 inherited、自前 sweep |"),
]

print(f"\nApplying {len(patches)} patches...")
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
n_applied_total = 0
n_missed = 0

for c in nb["cells"]:
    src = "".join(c.get("source", []))
    new_src = src
    for old, new in patches:
        cnt = new_src.count(old)
        if cnt > 0:
            new_src = new_src.replace(old, new)
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

# Count applications
all_new = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for old, new in patches:
    n_remaining = all_new.count(old)
    n_replaced = all_src.count(old) - n_remaining
    status = "OK" if n_remaining == 0 else f"⚠️  {n_remaining} remaining"
    n_applied_total += n_replaced
    if n_replaced == 0 and old not in all_src:
        n_missed += 1
        print(f"  [MISS] {old[:80]!r} not found in source")
    else:
        print(f"  [{status}] replaced {n_replaced}x: {old[:70]!r}")

print(f"\nTotal replacements: {n_applied_total}, missed patches: {n_missed}")

# ============================================================
# Save
# ============================================================
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {DST}")

# ============================================================
# Verify
# ============================================================
print("\n=== Verification ===")
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
checks = [
    ("lambda_prior=0.75 present (× 2)", all_src.count("lambda_prior=0.75") == 2),
    ("lambda_prior=0.65 absent (param call)", "lambda_prior=0.65," not in all_src),
    ("Header updated (exp092)", "exp092" in all_src and "lambda_prior sweep 0.65 → 0.75" in all_src),
    ("rank_aware power=0.6 intact", "power=0.6" in all_src),
    ("FCS_TOP_K=2 intact", "FCS_TOP_K = 2" in all_src),
    ("TAX_SMOOTHING α=0.15/0.05 intact", "genus_alpha=0.15" in all_src and "class_alpha=0.05" in all_src),
    ("Blend weights intact (0.35/0.40/0.25)", "BLEND_W_E10  = 0.35" in all_src
                                              and "BLEND_W_SED  = 0.40" in all_src
                                              and "BLEND_W_E17  = 0.25" in all_src),
]
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok: all_ok = False

# ============================================================
# Compile-strict check
# ============================================================
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

if all_ok and n_err == 0:
    print("\n✅ exp092 NB ready to push")
else:
    print("\n⚠️  Issues detected — review before push")
