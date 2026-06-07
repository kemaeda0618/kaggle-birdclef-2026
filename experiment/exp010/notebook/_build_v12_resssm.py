"""Build nb_blend_tucker_v12_resssm.ipynb from v11 by porting ResSSM cells from exp040.

Changes:
  - model cell: append ResidualSSM class (from exp040)
  - train cell: append ResidualSSM training block (from exp040)
  - nb4-infer cell: insert ResSSM correction inside inference loop (from exp040)
  - hdr cell: update title to reflect v12 spec
"""
import json
from pathlib import Path

V11_PATH = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v11_g26.ipynb")
E40_PATH = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp040\notebook\nb_blend.ipynb")
V12_PATH = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v12_resssm.ipynb")

v11 = json.load(open(V11_PATH, encoding="utf-8"))
e40 = json.load(open(E40_PATH, encoding="utf-8"))


def get_cell(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


# ---- model: append ResidualSSM class definition ----
v11_model = get_cell(v11, "model")
e40_model = get_cell(e40, "model")
e40_model_src = "".join(e40_model["source"])
# extract everything from the "# === exp040: ResidualSSM" marker onwards
marker = "# === exp040: ResidualSSM"
idx = e40_model_src.find(marker)
assert idx > 0, "ResidualSSM marker not found in exp040 model cell"
resssm_class_block = e40_model_src[idx:]
v11_model_src = "".join(v11_model["source"])
new_model_src = v11_model_src.rstrip() + "\n\n" + resssm_class_block
v11_model["source"] = new_model_src.splitlines(keepends=True)


# ---- train: append ResidualSSM training block ----
v11_train = get_cell(v11, "train")
e40_train = get_cell(e40, "train")
e40_train_src = "".join(e40_train["source"])
marker_t = "# === exp040: Train ResidualSSM"
idx_t = e40_train_src.find(marker_t)
assert idx_t > 0, "ResidualSSM train marker not found"
resssm_train_block = e40_train_src[idx_t:]
v11_train_src = "".join(v11_train["source"])
new_train_src = v11_train_src.rstrip() + "\n\n" + resssm_train_block
v11_train["source"] = new_train_src.splitlines(keepends=True)


# ---- nb4-infer: insert ResSSM correction inside inference loop ----
v11_infer = get_cell(v11, "nb4-infer")
v11_infer_src = "".join(v11_infer["source"])

# The exp040 patch inserts these 6 lines AFTER `l_blend = W_PROTO * l_proto + W_MLP * l_mlp`
# and BEFORE the retrieval/PP block.
insert_lines = (
    "\n"
    "        # === v12: ResidualSSM correction (second-pass, from exp040) ===\n"
    "        with torch.no_grad():\n"
    "            rs_corr = rs_model(emb_t, l_blend.float(),\n"
    "                               site_ids=site_t, hours=hour_t)\n"
    "            l_blend = l_blend + RESIDUAL_CORRECTION_WEIGHT * rs_corr\n"
)
needle = "        l_blend = W_PROTO * l_proto + W_MLP * l_mlp\n"
assert needle in v11_infer_src, "anchor 'l_blend = W_PROTO ...' not found in v11 nb4-infer cell"
new_infer_src = v11_infer_src.replace(needle, needle + insert_lines, 1)
# update header comment
new_infer_src = new_infer_src.replace(
    "# === Stage A: NB4 v7 inference + konbu17 LinearHead ===",
    "# === Stage A: NB4 v12 inference + ResSSM correction (v11 + exp040 ResidualSSM) ===",
)
v11_infer["source"] = new_infer_src.splitlines(keepends=True)


# ---- hdr: update header ----
v11_hdr = get_cell(v11, "hdr")
new_hdr = (
    "# v12 = v11 + ResidualSSM (second-pass correction, from exp040)\n"
    "\n"
    "v11 (NB4 ProtoSSM+MLP+KD+Retrieval+E19+FCS+Season prior+Mirror+G26 calibration) を keep し、\n"
    "exp040 で実装した ResidualSSM (BiSSMBlock 流用、in-NB train ~20-30 sec) を NB4 stream に追加。\n"
    "Inference: `l_blend += 0.30 * rs_model(emb, l_blend)` (logit-space correction)\n"
    "\n"
    "## 期待 LB\n"
    "- v11 standalone NB4 LB 0.926 → +ResSSM で 0.927-0.928 想定 (mtoshidesu 経験則 +0.001-0.002)\n"
    "- Tucker blend 込み v11 full LB に対して same gain transfer 期待\n"
)
v11_hdr["source"] = new_hdr.splitlines(keepends=True)


# ---- Write ----
json.dump(v11, open(V12_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Wrote {V12_PATH}")
print(f"cells: {len(v11['cells'])}")
for c in v11["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"  {c.get('cell_type'):8s} id={c.get('id'):30s} L={n}")
