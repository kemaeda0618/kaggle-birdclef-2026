"""Verify exp090: syntax + content checks."""
import json, ast, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp090_tuned_tax.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# 1. Syntax check
n_ok, n_err = 0, 0
errs = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        errs.append(f"  cell {i} (id={c.get('id','?')}): {e}")
print(f"Syntax: {n_ok} OK, {n_err} ERRORS")
for e in errs:
    print(e)

# 2. Content checks
all_src = "\n\n".join("".join(c.get("source", [])) for c in nb["cells"])

checks = [
    # lambda_prior 0.65
    ("lambda_prior=0.65 (1 occurrence at least)", all_src.count("lambda_prior=0.65") >= 2),
    ("lambda_prior=0.4 removed", all_src.count("lambda_prior=0.4") == 0),
    # rank_aware power=0.6
    ("rank_aware_scaling power=0.6", "power=0.6," in all_src),
    # TAX_SMOOTHING
    ("f_tax_smoothing function defined", "def f_tax_smoothing" in all_src),
    ("genus_alpha=0.15", "genus_alpha=0.15" in all_src),
    ("class_alpha=0.05", "class_alpha=0.05" in all_src),
    ("species_to_genus", "species_to_genus" in all_src),
    ("species_to_class", "species_to_class" in all_src),
    ("Tax smoothing call", "submission = f_tax_smoothing(" in all_src),
    # FCS NOT changed
    ("FCS_POWER = 0.4 retained", "FCS_POWER = 0.4" in all_src),
    ("FCS_POWER not changed to 0.6", "FCS_POWER = 0.6" not in all_src),
    # Submission save retained
    ("submission.to_csv retained", 'submission.to_csv("submission.csv", index=False)' in all_src),
    # exp078 stack intact
    ("BLEND_W_E10 = 0.35 (exp037)", "BLEND_W_E10  = 0.35" in all_src),
    ("BLEND_W_SED = 0.40 (Tucker)", "BLEND_W_SED  = 0.40" in all_src),
    ("BLEND_W_E17 = 0.25 (exp029)", "BLEND_W_E17  = 0.25" in all_src),
    # Sonotype mirror retained
    ("Sonotype mirror retained", "MIRROR_PAIRS" in all_src),
]
print("\nContent checks:")
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[MISS]'} {name}")
    if not ok: all_ok = False

# 3. Specific cell verification
print("\n=== exp037_full_pipeline cell (find apply_prior + rank_aware_scaling lines) ===")
for c in nb["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        for ln in src.split("\n"):
            if "lambda_prior" in ln or "rank_aware_scaling" in ln:
                print(f"  {ln.strip()[:140]}")
        break

print("\n=== blend cell — TAX_SMOOTHING section ===")
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Find tax smoothing section
        if "f_tax_smoothing" in src:
            idx = src.index("# === ★ exp090: Taxonomy")
            section = src[idx:idx+500]
            for ln in section.split("\n")[:15]:
                print(f"  {ln[:140]}")
        break

print(f"\n=== Overall: {'PASSED' if all_ok else 'FAILED'} ===")
