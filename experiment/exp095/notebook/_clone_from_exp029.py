"""Clone exp029 R3 gen script to exp095, replace identifiers.

Purpose: Create exp095 as identical baseline replica of exp029 R3.
- Same architecture (eca_nfnet_l1)
- Same params (epochs/lr/batch/mixup/etc)
- Same teacher (exp017 R2 pseudo_e17.csv)
- Only path/slug/title changed to exp095

Then run this gen script to produce nb_train_r3.ipynb for Colab upload.
"""
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp029\notebook\_gen_nb_train_r3_l1_single.py")
DST = Path(__file__).with_name("_gen_nb_train_r3.py")

print(f"[copy] {SRC.name} -> {DST.name}")
content = SRC.read_text(encoding="utf-8")

# Replace identifiers
replacements = [
    # 1. Top-level docstring
    ("\"\"\"Generate exp029: exp017 R2 single teacher", "\"\"\"Generate exp095: exp017 R2 single teacher"),
    # 2. Markdown header
    ("# exp029 R3 (l1 single fold, exp017 R2 teacher)",
     "# exp095 R3 (l1 single fold, exp017 R2 teacher) — exp029 と同 spec の baseline replica"),
    # 3. Kaggle Dataset slug references
    ("`birdclef2026-exp029-l1-single`", "`birdclef2026-exp095-l1-single`"),
    ("birdclef2026-exp029-l1-single", "birdclef2026-exp095-l1-single"),
    # 4. Drive output path
    ('"output" / "exp029"', '"output" / "exp095"'),
    # 5. Config comments (exp029 → exp095 in inline annotations)
    ("# ★ exp029:", "# ★ exp095 (= exp029 same):"),
    ("# ★ exp029 Option B", "# ★ exp095 (= exp029 Option B)"),
    ("# ★ exp029 Option C", "# ★ exp095 (= exp029 Option C)"),
    # 6. Print messages / TITLE
    ("\"birdclef2026 exp029 l1 single", "\"birdclef2026 exp095 l1 single"),
    ("exp029 R3 (l1 student, e17 teacher)", "exp095 R3 (l1 student, e17 teacher, replica of exp029)"),
    ("exp029 l1 single fold R3 summary", "exp095 l1 single fold R3 summary (replica of exp029)"),
    ("exp029 l1 single (R3 from e17)", "exp095 l1 single (R3 from e17)"),
    # 7. Drive folder note
    ("output/exp029/", "output/exp095/"),
]

for old, new in replacements:
    if old in content:
        n = content.count(old)
        content = content.replace(old, new)
        print(f"  [{n}x] {old[:80]!r}")
    else:
        print(f"  [MISS] {old[:80]!r}")

# Update the output NB filename (where generated NB is saved)
# Find the json.dump line
if "nb_train_r3_l1_single.ipynb" in content:
    content = content.replace(
        "nb_train_r3_l1_single.ipynb",
        "nb_train_r3.ipynb"
    )
    print("  [OK] Output NB filename: nb_train_r3.ipynb")

DST.write_text(content, encoding="utf-8")
print(f"\nSaved: {DST}")
print(f"Size: {DST.stat().st_size} bytes")

# Sanity verify no remaining exp029 (except in fallback comments that are intentional)
remaining = content.count("exp029")
print(f"\nRemaining 'exp029' references (should be minimal): {remaining}")
import re
# Show context of remaining
for m in re.finditer(r".{0,40}exp029.{0,40}", content):
    print(f"  ... {m.group(0)!r} ...")
