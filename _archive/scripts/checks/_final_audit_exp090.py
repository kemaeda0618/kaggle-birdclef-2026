"""Final audit of exp090 after bug fix.
Use compile() (strict) instead of ast.parse().
"""
import json, io, sys, difflib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb78 = json.loads(Path("experiment/exp078/notebook/nb_blend_v313_swap.ipynb").read_text(encoding="utf-8"))
nb90 = json.loads(Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb").read_text(encoding="utf-8"))

def by_id(nb):
    return {c.get("id", f"__noid_{i}__"): c for i, c in enumerate(nb["cells"])}

d78 = by_id(nb78)
d90 = by_id(nb90)

# Compile-check all cells
print("=== Compile-time syntax check (strict) ===")
n_ok = n_err = 0
for i, c in enumerate(nb90["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} (id={c.get('id','?')}): {e}")
print(f"  {n_ok} OK, {n_err} ERRORS")

# Show diff of changed cells (just key lines)
print("\n=== Changed cells: critical lines ===")
for cid in ["exp037_full_pipeline", "blend", "hdr"]:
    s78 = "".join(d78[cid].get("source", []))
    s90 = "".join(d90[cid].get("source", []))
    print(f"\n--- {cid} ---")
    if cid == "exp037_full_pipeline":
        # Show only the apply_prior + rank_aware lines for confirmation
        for keyword in ["apply_prior(", "rank_aware_scaling(", "file_confidence_scale("]:
            print(f"\n  {keyword}:")
            for ln in s90.split("\n"):
                if keyword in ln or (keyword in s90 and ("lambda_prior" in ln or "power=" in ln or "top_k=" in ln) and len(ln.strip()) < 80):
                    print(f"    {ln.rstrip()[:140]}")
            # Show first 5 matched lines only
    elif cid == "blend":
        # Show TAX_SMOOTHING related lines
        n = 0
        for ln in s90.split("\n"):
            if any(k in ln for k in ["tax_smoothing", "f_tax_smoothing", "submission.to_csv", "Sonotype mirror applied"]):
                print(f"    {ln.rstrip()[:140]}")
                n += 1
            if n > 10: break

# Detailed rank_aware_scaling context
print("\n=== rank_aware_scaling FULL call (4 lines context) ===")
for c in nb90["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        lines = "".join(c["source"]).split("\n")
        for i, ln in enumerate(lines):
            if "rank_aware_scaling(" in ln and "def " not in ln:
                for j in range(max(0,i-1), min(len(lines), i+4)):
                    print(f"  L{j:3d}: {repr(lines[j])}")
        break

# Verify apply_prior lambda values
print("\n=== apply_prior lambda values ===")
for c in nb90["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        for ln in "".join(c["source"]).split("\n"):
            if "lambda_prior" in ln:
                print(f"  {ln.strip()[:140]}")
        break

# Verify FCS unchanged
print("\n=== file_confidence_scale call (should still use power=0.4) ===")
for c in nb90["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        lines = "".join(c["source"]).split("\n")
        for i, ln in enumerate(lines):
            if "file_confidence_scale(" in ln and "def " not in ln:
                for j in range(max(0,i-1), min(len(lines), i+3)):
                    print(f"  L{j:3d}: {lines[j].rstrip()[:140]}")
        break

# Final OK
print("\n=== FINAL ===")
print(f"  Syntax: {'PASS' if n_err == 0 else 'FAIL'}")
print(f"  Diff scope: {[cid for cid in sorted(set(d78) & set(d90)) if ''.join(d78[cid].get('source',[])) != ''.join(d90[cid].get('source',[]))]}")
