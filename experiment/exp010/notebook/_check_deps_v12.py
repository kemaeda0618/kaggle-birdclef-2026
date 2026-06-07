"""Confirm BiSSMBlock and N_WINDOWS exist in v12 prior cells (needed by ResidualSSM)."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v12_resssm.ipynb", encoding="utf-8"))

def get_cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"])
    return None

# BiSSMBlock and N_WINDOWS should be defined before ResidualSSM (in model cell)
# or earlier (config cell for N_WINDOWS).
for cid in ("config", "load", "model"):
    src = get_cell(cid)
    for needle in ("class BiSSMBlock", "N_WINDOWS"):
        if needle in src:
            print(f"  [{cid}] contains: {needle}")

# Verify ResidualSSM in v12 model cell references these (uses them)
model_src = get_cell("model")
proto_idx = model_src.find("class ProtoSSM")
resssm_idx = model_src.find("class ResidualSSM")
bissm_idx = model_src.find("class BiSSMBlock")
print(f"\n  Order: ProtoSSM={proto_idx}, BiSSMBlock={bissm_idx}, ResidualSSM={resssm_idx}")
assert bissm_idx >= 0, "BiSSMBlock must be defined in model cell"
assert bissm_idx < resssm_idx, "BiSSMBlock must come BEFORE ResidualSSM"

# Verify rs_emb_t / rs_fp_t etc. won't collide with rs_model usage
# Also: confirm train cell uses lab_emb_files (defined in load)
train_src = get_cell("train")
load_src = get_cell("load")
for needle in ("lab_emb_files", "lab_scores_files", "lab_site_ids", "lab_hours", "lab_labels_files"):
    if needle not in load_src:
        print(f"  [WARN] {needle} not in load cell")
    if needle in train_src:
        print(f"  [train uses] {needle}")

# Verify lab_prior_files - it's used in train cell but might not be in load.
if "lab_prior_files" in load_src:
    print("  [load] has lab_prior_files")
else:
    print("  [WARN] lab_prior_files NOT in load cell - might break")

print("\n[deps OK]")
