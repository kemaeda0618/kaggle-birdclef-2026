"""Patch exp100 gen script: output ipynb name + dataset slug to reflect Babych spec."""
import io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(__file__).with_name("_gen_nb_train_r3.py")
content = SRC.read_text(encoding="utf-8")

patches = [
    # Output ipynb name: nb_train_r3_l1_single.ipynb -> nb_train_r3.ipynb
    ('"nb_train_r3_l1_single.ipynb"', '"nb_train_r3.ipynb"'),
    # Slug rename: l1-single -> l1-babych (reflects Babych spec)
    ('birdclef2026-exp100-l1-single', 'birdclef2026-exp100-l1-babych'),
    # Title
    ('birdclef2026 exp100 l1 single (R3 from e17)', 'birdclef2026 exp100 l1 babych (R3 from e17, mixup 1.0 + dp 0.15 + WRS)'),
    # Run directive (cosmetic)
    ('_gen_nb_train_r3_l1_single.py', '_gen_nb_train_r3.py'),
]

n_applied = 0
for old, new in patches:
    if old in content:
        cnt = content.count(old)
        content = content.replace(old, new)
        n_applied += cnt
        print(f"  [{cnt}x OK] {old[:80]!r}")
    else:
        print(f"  [MISS] {old[:80]!r}")

SRC.write_text(content, encoding="utf-8")
print(f"\nTotal applied: {n_applied}, saved {SRC}")
