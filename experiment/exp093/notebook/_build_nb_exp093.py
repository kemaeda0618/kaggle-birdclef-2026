"""Build exp093 NB: exp090 base + Per-class blend (mapped/unmapped split).

Concept (lb-0948 NB-derived):
- Mapped 種 (Perch logit 持つ ~70-100 種): NB4 が direct Perch logit 持つので強め
- Unmapped 種 (Perch なし、genus proxy 経由 ~130-160 種): NB4 の proxy 不安定、Tucker 強め

Weights:
- Baseline (exp090):       NB4=0.35 / Tucker=0.40 / R3=0.25
- Mapped (exp093 新):     NB4=0.40 / Tucker=0.35 / R3=0.25
- Unmapped (exp093 新):   NB4=0.25 / Tucker=0.50 / R3=0.25

Pattern reusable from exp056/062/063 (Aves-only blend) but using MAPPED_MASK instead of AVES_MASK.

Implementation: blend cell only (~20 lines change), all other PP intact.
"""
import json, io, sys, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp090\notebook\nb_exp090_tuned_tax.ipynb")
DST = Path(__file__).with_name("nb_exp093_per_class_blend.ipynb")

print(f"[clone] {SRC.name} -> {DST.name}")
shutil.copy(SRC, DST)

nb = json.loads(DST.read_text(encoding="utf-8"))

# ============================================================
# Patch 1: Header
# ============================================================
HEADER_OLD = "# exp090 — exp078 (LB 0.950) + lambda_prior 0.65 + rank_aware power 0.6 + TAX_SMOOTHING"
HEADER_NEW = """# exp093 — exp090 (LB 0.951) base + **Per-class blend (mapped/unmapped split)** ★

## 新規変更
3-way blend weight を per-class で差別化 (lb-0948 NB-derived 概念):
- **Mapped 種** (Perch logit 持つ): NB4=**0.40** / Tucker=**0.35** / R3=**0.25** (NB4 強め、direct Perch logit)
- **Unmapped 種** (Perch なし、genus proxy 経由): NB4=**0.25** / Tucker=**0.50** / R3=**0.25** (Tucker 強め、Perch 系信頼度下)
- Baseline (exp090): 全 234 種 NB4=0.35/Tucker=0.40/R3=0.25 一律

## Pattern source
exp056/062/063 で `AVES_MASK` 使った per-class blend 実装あり。今回は `MAPPED_POS`/`UNMAPPED_POS` (exp090 既存 line 309/326) を流用。

## 期待
LB +0.002-0.005 (lb-0948 NB 実証由来)、Babych "full pp package" 0.020 lift の一部"""

patches = [
    (HEADER_OLD, HEADER_NEW),
]

# ============================================================
# Patch 2: Replace blend cell (else-branch) with per-class version
# ============================================================
# Find blend cell and replace the 3-way blend logic + add W_MAPPED/W_UNMAPPED defs

OLD_BLEND_BLOCK = '''BLEND_W_E10  = 0.35   # NB4 v7 weight
BLEND_W_SED  = 0.40   # Tucker public SED weight
BLEND_W_E17  = 0.25   # exp029 R3 (eca_nfnet_l1) weight — note var name kept "E17" for code reuse
USE_SED_PRE_AVG = False'''

NEW_BLEND_BLOCK = '''BLEND_W_E10  = 0.35   # NB4 v7 weight (legacy uniform、per-class blend では未使用)
BLEND_W_SED  = 0.40   # Tucker public SED weight (legacy uniform)
BLEND_W_E17  = 0.25   # exp029 R3 weight (legacy uniform)
USE_SED_PRE_AVG = False

# === exp093: Per-class blend weights (mapped vs unmapped split) ===
USE_PER_CLASS_BLEND = True
# Mapped 種 (Perch logit 持つ、~70-100 種): NB4 強め (direct Perch logit access)
W_MAPPED_E10 = 0.40    # NB4 (was 0.35) +0.05
W_MAPPED_SED = 0.35    # Tucker (was 0.40) -0.05
W_MAPPED_E17 = 0.25    # R3 (unchanged)
# Unmapped 種 (Perch なし、genus proxy 経由、~130-160 種): Tucker 強め
W_UNMAPPED_E10 = 0.25  # NB4 (was 0.35) -0.10 (genus proxy 不安定)
W_UNMAPPED_SED = 0.50  # Tucker (was 0.40) +0.10
W_UNMAPPED_E17 = 0.25  # R3 (unchanged)
assert abs(W_MAPPED_E10 + W_MAPPED_SED + W_MAPPED_E17 - 1.0) < 1e-6, "Mapped weights must sum to 1.0"
assert abs(W_UNMAPPED_E10 + W_UNMAPPED_SED + W_UNMAPPED_E17 - 1.0) < 1e-6, "Unmapped weights must sum to 1.0"'''

# Replace the 3-way else-branch with per-class version
OLD_ELSE_BRANCH = '''else:
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    blend_flat = (BLEND_W_E10 * rank_e10
                  + BLEND_W_SED * rank_sed
                  + BLEND_W_E17 * rank_e17)
    print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e29={BLEND_W_E17}")'''

NEW_ELSE_BRANCH = '''else:
    rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
    if USE_PER_CLASS_BLEND:
        # === exp093: Per-class blend (mapped/unmapped split) ===
        # MAPPED_POS / UNMAPPED_POS defined at top (line 309 / 326)
        blend_flat = np.zeros_like(rank_e10, dtype=np.float32)
        blend_flat[:, MAPPED_POS] = (W_MAPPED_E10 * rank_e10[:, MAPPED_POS]
                                     + W_MAPPED_SED * rank_sed[:, MAPPED_POS]
                                     + W_MAPPED_E17 * rank_e17[:, MAPPED_POS]).astype(np.float32)
        blend_flat[:, UNMAPPED_POS] = (W_UNMAPPED_E10 * rank_e10[:, UNMAPPED_POS]
                                       + W_UNMAPPED_SED * rank_sed[:, UNMAPPED_POS]
                                       + W_UNMAPPED_E17 * rank_e17[:, UNMAPPED_POS]).astype(np.float32)
        print(f"  Per-class blend ENABLED:")
        print(f"    Mapped   ({len(MAPPED_POS)} sp): NB4={W_MAPPED_E10}/Tucker={W_MAPPED_SED}/e29={W_MAPPED_E17}")
        print(f"    Unmapped ({len(UNMAPPED_POS)} sp): NB4={W_UNMAPPED_E10}/Tucker={W_UNMAPPED_SED}/e29={W_UNMAPPED_E17}")
    else:
        blend_flat = (BLEND_W_E10 * rank_e10
                      + BLEND_W_SED * rank_sed
                      + BLEND_W_E17 * rank_e17)
        print(f"  3-way uniform rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e29={BLEND_W_E17}")'''

patches.extend([
    (OLD_BLEND_BLOCK, NEW_BLEND_BLOCK),
    (OLD_ELSE_BRANCH, NEW_ELSE_BRANCH),
])

# Also update the assertion check to not fail for unequal weights (each set sums to 1.0 individually)
# The old assertion is `assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6`
# We keep it for backward compat (legacy uniform weights still sum to 1)

print(f"\nApplying {len(patches)} patches...")
all_src_before = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
n_total = 0
n_missed = 0

for c in nb["cells"]:
    src = "".join(c.get("source", []))
    new_src = src
    for old, new in patches:
        if old in new_src:
            new_src = new_src.replace(old, new)
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

all_src_after = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for i, (old, new) in enumerate(patches):
    n_remaining = all_src_after.count(old)
    n_replaced = all_src_before.count(old) - n_remaining
    n_total += n_replaced
    status = "OK" if n_remaining == 0 and n_replaced > 0 else f"⚠️ remaining={n_remaining}, replaced={n_replaced}"
    label = old[:60].replace("\n", " ")
    print(f"  Patch {i+1}: [{status}] {label!r}...")
    if n_remaining > 0 and n_replaced == 0:
        n_missed += 1

print(f"\nTotal replacements: {n_total}, missed patches: {n_missed}")

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
    ("exp093 header present", "exp093 — exp090" in all_src and "Per-class blend" in all_src),
    ("USE_PER_CLASS_BLEND = True", "USE_PER_CLASS_BLEND = True" in all_src),
    ("W_MAPPED defined (3 weights)", all(w in all_src for w in ["W_MAPPED_E10 = 0.40", "W_MAPPED_SED = 0.35", "W_MAPPED_E17 = 0.25"])),
    ("W_UNMAPPED defined (3 weights)", all(w in all_src for w in ["W_UNMAPPED_E10 = 0.25", "W_UNMAPPED_SED = 0.50", "W_UNMAPPED_E17 = 0.25"])),
    ("Per-class blend cell applies MAPPED_POS slicing", "blend_flat[:, MAPPED_POS]" in all_src),
    ("Per-class blend cell applies UNMAPPED_POS slicing", "blend_flat[:, UNMAPPED_POS]" in all_src),
    ("Sum-to-1 assertions present", "Mapped weights must sum to 1.0" in all_src),
    # Unchanged params
    ("lambda_prior=0.65 intact", all_src.count("lambda_prior=0.65") == 2),
    ("rank_aware power=0.6 intact", "power=0.6" in all_src),
    ("FCS_TOP_K=2 intact", "FCS_TOP_K = 2" in all_src),
    ("TAX_SMOOTHING α=0.15/0.05 intact", "genus_alpha=0.15" in all_src and "class_alpha=0.05" in all_src),
    ("Legacy BLEND_W values intact (for fallback)", "BLEND_W_E10  = 0.35" in all_src and "BLEND_W_SED  = 0.40" in all_src and "BLEND_W_E17  = 0.25" in all_src),
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
    print("\n✅ exp093 NB ready to push")
else:
    print("\n⚠️  Issues detected — review before push")
