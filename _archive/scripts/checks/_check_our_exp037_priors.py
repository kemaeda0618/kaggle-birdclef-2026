"""Check our exp078 (= exp037 v313) for Hour/Site prior, TAX smoothing, rank_aware_scaling, etc.
to determine which anthonytherrien techniques are truly NEW for us."""
import json, io, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb = json.loads(Path("experiment/exp078/notebook/nb_blend_v313_swap.ipynb").read_text(encoding="utf-8"))

all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

# Check for each technique
techniques = {
    "Hour prior (apply_prior with hours)": ["apply_prior", "hour_utc", "hour_prior", "hour_keys"],
    "Site prior": ["site_prior", "S22", "site_p", "site_keys"],
    "lambda_prior": ["lambda_prior"],
    "rank_aware_scaling": ["rank_aware_scaling", "rank_aware"],
    "FCS (File Confidence Scale)": ["FCS_TOP_K", "_topk_mean"],
    "f_TAX_SMOOTHING_POSTPROC": ["TAX_SMOOTHING", "f_TAX_SMOOTHING"],
    "genus smoothing": ["genus_α", "genus_alpha", "species_to_genus"],
    "class smoothing": ["class_α", "class_alpha", "species_to_class"],
    "Circular Gaussian smoothing": ["circular", "Circular Gaussian"],
    "Cross-Attention LightProtoSSM": ["cross_attn", "CrossAttention", "cross_attention_heads"],
    "Per-taxon temperature": ["per_taxon", "event_temp", "texture_temp"],
}

print("=== Techniques in our exp078 ===\n")
for tech, patterns in techniques.items():
    matches = []
    for p in patterns:
        if p in all_src:
            matches.append(p)
    status = "✓ FOUND" if matches else "✗ NOT FOUND"
    print(f"{status}: {tech}")
    if matches:
        # Show context (first occurrence)
        for p in matches[:2]:
            idx = all_src.index(p)
            ctx = all_src[max(0, idx-50):idx+100].replace("\n", " | ")
            print(f"     [{p}] ... {ctx[:150]} ...")
    print()

# Specific check: our FCS implementation
print("\n=== Our FCS (File Confidence Scaling) implementation ===")
for ln in all_src.split("\n"):
    if "FCS_TOP_K" in ln or "_topk_mean" in ln or "FCS_POWER" in ln:
        print(f"  {ln.strip()[:150]}")

# Specific check: our Prior table builder (v313_10)
print("\n=== Our 'Prior table builder' cell (v313_10) content ===")
for c in nb["cells"]:
    if c.get("id") == "v313_10":
        src = "".join(c["source"])
        # First 60 lines
        for ln in src.split("\n")[:80]:
            print(f"  {ln[:140]}")
        break

# Specific check: our File-level confidence scaling (v313_11)
print("\n=== Our 'File-level confidence scaling' cell (v313_11) content ===")
for c in nb["cells"]:
    if c.get("id") == "v313_11":
        src = "".join(c["source"])
        for ln in src.split("\n")[:60]:
            print(f"  {ln[:140]}")
        break
