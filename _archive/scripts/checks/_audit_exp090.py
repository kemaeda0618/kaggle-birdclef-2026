"""Comprehensive audit of exp090 vs exp078.
1. Full diff to confirm ONLY intended changes
2. Inspect changed lines IN CONTEXT
3. Verify TAX_SMOOTHING placement
4. Check for syntax issues
5. Verify dependency parity with exp078
"""
import json, ast, io, sys, difflib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb78 = json.loads(Path("experiment/exp078/notebook/nb_blend_v313_swap.ipynb").read_text(encoding="utf-8"))
nb90 = json.loads(Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb").read_text(encoding="utf-8"))

def by_id(nb):
    return {c.get("id", f"__noid_{i}__"): c for i, c in enumerate(nb["cells"])}

d78 = by_id(nb78)
d90 = by_id(nb90)

# Step 1: Cell-by-cell diff
print("=== Cell-by-cell diff (exp078 vs exp090) ===")
print(f"exp078 cells: {len(nb78['cells'])}, exp090 cells: {len(nb90['cells'])}")

added = set(d90.keys()) - set(d78.keys())
removed = set(d78.keys()) - set(d90.keys())
common = set(d78.keys()) & set(d90.keys())

print(f"\nAdded cells (new in exp090): {added}")
print(f"Removed cells: {removed}")

diff_cells = []
for cid in sorted(common):
    s78 = "".join(d78[cid].get("source", []))
    s90 = "".join(d90[cid].get("source", []))
    if s78 != s90:
        diff_cells.append(cid)

print(f"Differing cells: {len(diff_cells)}/{len(common)} = {diff_cells}")

# Step 2: Inspect each differing cell
for cid in diff_cells:
    s78 = "".join(d78[cid].get("source", []))
    s90 = "".join(d90[cid].get("source", []))
    print(f"\n{'='*70}")
    print(f"DIFF in cell '{cid}'")
    print(f"  exp078: {len(s78)} chars, exp090: {len(s90)} chars (Δ={len(s90)-len(s78):+d})")
    print('='*70)
    diff = list(difflib.unified_diff(
        s78.splitlines(),
        s90.splitlines(),
        lineterm="",
        n=3,
        fromfile=f"exp078/{cid}",
        tofile=f"exp090/{cid}",
    ))
    # Show context (limit per cell)
    for ln in diff[:120]:
        print(ln[:200])
    if len(diff) > 120:
        print(f"  ... ({len(diff) - 120} more diff lines truncated)")

# Step 3: Inspect actual apply_prior + rank_aware_scaling + tax_smoothing IN CONTEXT
print(f"\n\n{'='*70}")
print("=== CRITICAL CONTEXT CHECK: changed calls in exp090 ===")
print('='*70)

for c in nb90["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        # Find each apply_prior call and show context (3 lines before + 5 lines after)
        lines = src.split("\n")
        for i, ln in enumerate(lines):
            if "apply_prior" in ln and "def apply_prior" not in ln:
                print(f"\n--- apply_prior call at L{i} ---")
                start = max(0, i - 3)
                end = min(len(lines), i + 8)
                for j in range(start, end):
                    marker = " >>>" if "lambda_prior=0.65" in lines[j] else "    "
                    print(f"  {marker} L{j:3d}: {lines[j][:140]}")
        # Find rank_aware_scaling call
        for i, ln in enumerate(lines):
            if "rank_aware_scaling" in ln and "def rank_aware_scaling" not in ln:
                print(f"\n--- rank_aware_scaling call at L{i} ---")
                start = max(0, i - 2)
                end = min(len(lines), i + 6)
                for j in range(start, end):
                    marker = " >>>" if "power=0.6" in lines[j] else "    "
                    print(f"  {marker} L{j:3d}: {lines[j][:140]}")
        # Also verify file_confidence_scale UNCHANGED
        for i, ln in enumerate(lines):
            if "file_confidence_scale" in ln and "def file_confidence_scale" not in ln:
                print(f"\n--- file_confidence_scale call at L{i} (should be UNCHANGED) ---")
                start = max(0, i - 2)
                end = min(len(lines), i + 6)
                for j in range(start, end):
                    print(f"      L{j:3d}: {lines[j][:140]}")
        break

# Step 4: TAX_SMOOTHING placement in blend cell
print(f"\n\n{'='*70}")
print("=== TAX_SMOOTHING placement check (blend cell) ===")
print('='*70)

for c in nb90["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        lines = src.split("\n")
        # Find tax_smoothing related lines + context
        for i, ln in enumerate(lines):
            if "tax_smoothing" in ln or "f_tax_smoothing" in ln or "submission.to_csv" in ln or "Sonotype mirror" in ln:
                print(f"  L{i:3d}: {ln[:140]}")
        break

# Step 5: Verify no remaining lambda_prior=0.4 in CALL sites (function def can keep default)
print(f"\n\n{'='*70}")
print("=== Verify no lambda_prior=0.4 CALL sites remain ===")
print('='*70)
for c in nb90["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        n_call = 0
        n_def = 0
        for ln in src.split("\n"):
            s = ln.strip()
            if "lambda_prior=0.4" in ln:
                if s.startswith("def "):
                    n_def += 1
                    print(f"  [DEF] OK keep: {ln.strip()[:140]}")
                else:
                    n_call += 1
                    print(f"  [CALL] ⚠ WARNING: {ln.strip()[:140]}")
        print(f"  Summary: {n_call} call sites with 0.4, {n_def} def with 0.4")
        if n_call > 0:
            print("  [FAIL] CALL sites still use 0.4!")

# Step 6: Syntax check
print(f"\n{'='*70}")
print("=== Syntax check ===")
print('='*70)
n_ok, n_err = 0, 0
for i, c in enumerate(nb90["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  cell {i} (id={c.get('id','?')}): {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")

# Step 7: Final summary
print(f"\n{'='*70}")
print("=== AUDIT SUMMARY ===")
print('='*70)
expected_diff_cells = {"hdr", "exp037_full_pipeline", "blend"}
unexpected_added = added - set()  # No expected new cells (only existing modified)
print(f"  Diff cells: {set(diff_cells)} (expected: {expected_diff_cells})")
print(f"  Unexpected added cells: {unexpected_added}")
print(f"  Unexpected removed cells: {removed}")
print(f"  Syntax: {n_ok} OK")
