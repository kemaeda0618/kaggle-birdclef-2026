"""Patch exp095 gen script: l1 → l0, exp029 R3 spec → exp017 R2 spec.

Changes:
- BACKBONE: eca_nfnet_l1 → eca_nfnet_l0
- WARMUP_EPOCHS: 4 → 3 (R2 spec)
- SHARES_R2: focal 0.65/labeled 0.10/pseudo 0.25 (R3 stronger pseudo)
       → focal 0.70/labeled 0.10/pseudo 0.20 (R2 exact spec)
- Slug: l1-single → l0-r2spec

Then re-run gen to produce updated nb_train_r3.ipynb.
"""
import shutil
from pathlib import Path

SRC = Path(__file__).with_name("_gen_nb_train_r3.py")
content = SRC.read_text(encoding="utf-8")

# === Patches ===
patches = [
    # 1. Top docstring
    ('"""Generate exp095: exp017 R2 single teacher → l1 student (single fold) NB.',
     '"""Generate exp095: exp017 R2 teacher → l0 student (R2 exact spec、single fold) NB.'),

    # 2. Markdown header
    ('# exp095 R3 (l1 single fold, exp017 R2 teacher) — exp029 と同 spec の baseline replica',
     '# exp095 R3 (l0 + R2 exact spec, exp017 R2 teacher) — l0 で val→LB gap 改善狙い'),

    # 3. BACKBONE
    ('BACKBONE = "eca_nfnet_l1"   # ★ exp095 (= exp029 same): l0 → l1 (capacity 増、Option B 検証)',
     'BACKBONE = "eca_nfnet_l0"   # ★ exp095: l0 維持 (exp017 R2 success spec、val→LB gap -0.004)'),

    # 4. WARMUP_EPOCHS
    ('WARMUP_EPOCHS = 4         # ★ exp095 (= exp029 same): l1 で +1 ep (exp022 同設定)',
     'WARMUP_EPOCHS = 3         # ★ exp095: R2 spec (l0 なので +1 不要)'),

    # 5. SHARES_R2 (R3 specific → R2 exact spec)
    ('SHARES_R2 = {"focal": 0.65, "labeled_sc": 0.10, "pseudo_sc": 0.25}   # ★ exp095 (= exp029 Option C): pseudo 0.20 → 0.25、distill 強化',
     'SHARES_R2 = {"focal": 0.70, "labeled_sc": 0.10, "pseudo_sc": 0.20}   # ★ exp095: R2 spec exact (focal 0.70 / pseudo 0.20)'),

    # 6. Output naming: l1-single → l0-r2spec
    ('birdclef2026-exp095-l1-single',
     'birdclef2026-exp095-l0-r2spec'),
    ('"birdclef2026 exp095 l1 single',
     '"birdclef2026 exp095 l0 r2spec'),
    ('exp095 R3 (l1 student, e17 teacher, replica of exp029)',
     'exp095 R3 (l0 student + R2 exact spec, e17 teacher)'),
    ('exp095 l1 single (R3 from e17)',
     'exp095 l0 R3 (R2 exact spec, e17 teacher)'),
    ('exp095 l1 single fold R3 summary (replica of exp029)',
     'exp095 l0 R3 summary (R2 exact spec)'),
]

print(f"Applying {len(patches)} patches...")
applied = 0
for old, new in patches:
    if old in content:
        n = content.count(old)
        content = content.replace(old, new)
        applied += n
        print(f"  [{n}x OK] {old[:80]!r}")
    else:
        print(f"  [MISS] {old[:80]!r}")

print(f"\nTotal applied: {applied}")
SRC.write_text(content, encoding="utf-8")
print(f"Saved: {SRC}")
