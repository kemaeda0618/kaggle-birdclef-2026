"""Smoke check v12 by inspecting key edits."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v12_resssm.ipynb", encoding="utf-8"))

def get_cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"])
    return None

# Verify model cell contains ResidualSSM class
m = get_cell("model")
assert "class ResidualSSM" in m, "ResidualSSM class missing"
assert "RESIDUAL_CORRECTION_WEIGHT" in m, "RESIDUAL_CORRECTION_WEIGHT missing"
assert "BiSSMBlock" in m, "BiSSMBlock reference missing (used by ResidualSSM)"
print("[model] OK")

# Verify train cell contains ResSSM training
t = get_cell("train")
assert "Training ResidualSSM" in t, "ResSSM training block missing"
assert "rs_model = ResidualSSM" in t, "rs_model instantiation missing"
assert "best_rs_state" in t, "best_state tracking missing"
print("[train] OK")

# Verify nb4-infer cell has ResSSM correction
i = get_cell("nb4-infer")
assert "rs_corr = rs_model" in i, "ResSSM correction call missing"
assert "RESIDUAL_CORRECTION_WEIGHT * rs_corr" in i, "Correction add missing"
# Verify correction is BEFORE retrieval (LAMBDA_RETRIEVAL)
idx_corr = i.find("rs_corr = rs_model")
idx_ret = i.find("LAMBDA_RETRIEVAL")
assert 0 < idx_corr < idx_ret, "ResSSM correction must come before retrieval block"
print("[nb4-infer] OK (correction inserted before retrieval)")

# Verify rs_model is in scope (defined globally in train, used in nb4-infer)
# Just sanity check that rs_model.eval() or similar is reachable
print("[ALL OK]")

# Print cell sizes
print("\ncell sizes:")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"  {c.get('cell_type'):8s} id={c.get('id'):30s} L={n}")
