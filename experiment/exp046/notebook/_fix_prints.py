"""Fix leftover exp029 references in print statements (cosmetic but misleading)."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp046\notebook\nb_train_r3_l1_gamma12.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

FIXES = [
    # (cell_id, old, new)
    ("fold_loop",
     '# exp029 R3 (l1 student, e17 teacher) Fold {FOLD_K} (of {N_FOLDS}-fold split, single fold training)',
     '# exp046 R3 + γ=1.2 (l1 student, e17 teacher) Fold {FOLD_K} (of {N_FOLDS}-fold split, single fold training)'),
    ("summary",
     'print(f"exp029 l1 single fold R3 summary")',
     'print(f"exp046 l1 single fold R3 + γ={POWER_GAMMA} Power Transform summary")'),
    ("summary",
     'print(f"\\n  Kaggle Dataset: https://www.kaggle.com/datasets/maekeso/birdclef2026-exp029-l1-single")',
     'print(f"\\n  Kaggle Dataset: https://www.kaggle.com/datasets/maekeso/birdclef2026-exp046-l1-gamma12")'),
]

for cid, old, new in FIXES:
    found = False
    for c in nb["cells"]:
        if c.get("id") == cid:
            src = "".join(c["source"])
            if old in src:
                src = src.replace(old, new, 1)
                c["source"] = src.splitlines(keepends=True)
                print(f"  [{cid}] OK fixed: {old[:80]}")
                found = True
            break
    if not found:
        print(f"  [{cid}] WARN: pattern not found: {old[:80]}")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {NB}")

# Re-verify
print("\n=== Re-verify ===")
for c in nb["cells"]:
    cid = c.get("id", "?")
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    for i, line in enumerate(src.split("\n"), 1):
        if "print" not in line:
            continue
        ls = line.strip()
        # Look for residual exp029 / exp045 in PRINT statements (excluding baseline references in hdr)
        if ("exp029" in ls or "exp045" in ls) and "print" in ls:
            print(f"  [{cid}:L{i}] REMAINING: {ls[:130]}")
