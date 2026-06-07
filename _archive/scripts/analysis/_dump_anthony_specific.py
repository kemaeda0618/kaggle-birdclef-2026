"""Extract specific high-value cells from anthonytherrien 0.950 NB:
- Cell 2 (solutions dict): which models + weights
- Cell 3 (parameters)
- Cell 23 (blend logic = direct_add)
- Cell 24 (f_TAX_SMOOTHING_POSTPROC) - the special post-processing
- Cell 25 (dispatcher)
- Cell 27 (submit)

Also extract key code patterns from Model_5 (POWEROPT) cells.
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PULL_ROOT = Path("C:/Users/maeke/work/kaggle/birdclef-2026/_public_nb_pulls")
nb = json.loads((PULL_ROOT / "anthonytherrien__birdclef-2026-ensemble-0-950" /
                  "birdclef-2026-ensemble-0-950.ipynb").read_text(encoding="utf-8"))

cells_to_dump = [2, 3, 23, 24, 25, 27]
for idx in cells_to_dump:
    c = nb["cells"][idx]
    print(f"\n{'='*70}\n# CELL {idx} ({c['cell_type']})\n{'='*70}")
    print("".join(c.get("source", [])))

# Also extract key snippets from Model_51 (Model_5 family) - POWEROPT
print(f"\n\n{'='*70}\n# MODEL_51 CELL — Key snippets (POWEROPT body)\n{'='*70}")
m51 = "".join(nb["cells"][13]["source"])
lines = m51.split("\n")

# Find rank_aware_scaling definition
print("\n--- rank_aware_scaling function ---")
in_def = False
indent_target = None
for i, ln in enumerate(lines):
    if "def rank_aware_scaling" in ln:
        in_def = True
        indent_target = len(ln) - len(ln.lstrip())
        print(f"L{i}: {ln}")
        continue
    if in_def:
        if ln.strip() == "" and i < len(lines)-1 and lines[i+1].strip():
            # check if next line is dedented (def end)
            next_ln = lines[i+1]
            next_indent = len(next_ln) - len(next_ln.lstrip())
            if next_indent <= indent_target and not next_ln.strip().startswith("#"):
                print(f"L{i}: {ln}")
                break
        print(f"L{i}: {ln}")
        if len(ln) > 0 and not ln[0].isspace() and ln.strip() and not ln.strip().startswith("#"):
            break

# Find lambda_prior usage
print("\n--- lambda_prior usage (first 20 mentions) ---")
count = 0
for i, ln in enumerate(lines):
    if "lambda_prior" in ln:
        print(f"L{i}: {ln.strip()[:120]}")
        count += 1
        if count >= 20: break

# Find Hour/Site prior code
print("\n--- Hour prior code ---")
count = 0
for i, ln in enumerate(lines):
    if "hour" in ln.lower() and ("prior" in ln.lower() or "_utc" in ln or "_hour" in ln):
        print(f"L{i}: {ln.strip()[:120]}")
        count += 1
        if count >= 15: break

print("\n--- Cross-Attention code ---")
count = 0
for i, ln in enumerate(lines):
    if "cross" in ln.lower() and "attention" in ln.lower():
        print(f"L{i}: {ln.strip()[:120]}")
        count += 1
        if count >= 10: break

# Find blend weight in Model_51
print("\n--- Final blend / weight (Model_5*) ---")
count = 0
for i, ln in enumerate(lines):
    s = ln.strip()
    if ("blend" in s.lower() or "weight" in s.lower()) and "=" in s and any(c.isdigit() for c in s):
        if not s.startswith("#") and "lambda_prior" not in s:
            print(f"L{i}: {s[:120]}")
            count += 1
            if count >= 15: break
