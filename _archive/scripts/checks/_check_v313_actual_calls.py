"""Check exp078 (exp037 v313) for ACTUAL calls (not just definitions) of:
- apply_prior(lambda_prior=?)
- rank_aware_scaling(power=?)
- file_confidence_scale(power=?)
- Cross-Attention LightProtoSSM (use_cross_attn=?)
"""
import json, io, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb = json.loads(Path("experiment/exp078/notebook/nb_blend_v313_swap.ipynb").read_text(encoding="utf-8"))

# exp037_full_pipeline cell (the actual driver)
for c in nb["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        print("=== exp037_full_pipeline cell ===")
        # Look for actual calls
        print("\n--- apply_prior CALLS (not definitions) ---")
        for ln in src.split("\n"):
            if "apply_prior" in ln and "def apply_prior" not in ln:
                print(f"  {ln.strip()[:160]}")

        print("\n--- rank_aware_scaling CALLS ---")
        for ln in src.split("\n"):
            if "rank_aware_scaling" in ln and "def rank_aware_scaling" not in ln:
                print(f"  {ln.strip()[:160]}")

        print("\n--- file_confidence_scale CALLS ---")
        for ln in src.split("\n"):
            if "file_confidence_scale" in ln and "def file_confidence_scale" not in ln:
                print(f"  {ln.strip()[:160]}")

        print("\n--- LightProtoSSM CALLS (constructor with use_cross_attn) ---")
        in_block = False
        for i, ln in enumerate(src.split("\n")):
            if "LightProtoSSM(" in ln and "class" not in ln:
                # Print this line + next 5 lines (constructor args)
                lines = src.split("\n")[i:i+8]
                for sl in lines:
                    print(f"  {sl[:160]}")
                print("  ---")

        print("\n--- All lambda_prior= occurrences ---")
        for ln in src.split("\n"):
            if "lambda_prior=" in ln:
                print(f"  {ln.strip()[:160]}")
        break

# Also check the v313_18 / v313_19 cells (Cross-Attention LightProtoSSM def + usage)
print("\n\n=== v313_18 / v313_19 / v313_20 (LightProtoSSM + ResSSM + train) ===")
for cid in ["v313_18", "v313_19", "v313_20", "v313_22"]:
    for c in nb["cells"]:
        if c.get("id") == cid:
            src = "".join(c["source"])
            print(f"\n--- {cid} (first 30 lines) ---")
            for ln in src.split("\n")[:30]:
                print(f"  {ln[:140]}")
            break

# Look for the FCS_TOP_K usage in exp078 final blend
print("\n\n=== Cell 'blend' — actual blend logic ===")
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        for ln in src.split("\n"):
            if "FCS" in ln or "rank_aware" in ln or "apply_prior" in ln or "TAX_SMOOTH" in ln:
                print(f"  {ln.strip()[:160]}")
        break
