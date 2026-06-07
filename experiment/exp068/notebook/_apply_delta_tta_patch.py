"""Apply Delta TTA modifications to nb_blend_delta_tta.ipynb.

Changes:
  1. Title/markdown cell (cell 0): mention Delta TTA
  2. CFG cell (cell 3): add TTA_DELTA = 1 + helpers _delta_max_np / _delta_max_torch
  3. Tucker SED cell (cell 10): swap frame_max to _delta_max_np
  4. exp029 cell (cell 11): swap _frame_max to _delta_max_torch
"""
import json, sys, io, os
from pathlib import Path

if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_blend_delta_tta.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# ---- 1. Title (cell 0) ----
title_cell = nb["cells"][0]
title_src = "".join(title_cell["source"])
# Replace title and lineage description
title_src_new = title_src.replace(
    "# exp048: TopN N=1 PP (paper 256 spec, K=1, POW=1.0)",
    "# exp068: Delta TTA (Babych BC25 1st frame-axis shift, on top of exp048)"
).replace(
    "# exp040 = NB4 v11 + ResSSM (0.30) + Tucker SED (0.40) + exp029 (0.30)",
    "# exp068 = exp048 base + Delta TTA (frame-axis ±1 shift) on Tucker SED + exp029 streams"
).replace(
    "- exp048 (TopN K=1 spec) over exp040",
    "- exp068: exp048 base + Babych-style delta-shift TTA on framewise output (Slot 2 + Slot 3)"
)
title_cell["source"] = title_src_new.splitlines(keepends=True)
print(f"  Cell 0 title updated ({len(title_src)} → {len(title_src_new)} chars)")

# ---- 2. CFG cell (cell 3): add TTA_DELTA + helpers ----
cfg_cell = nb["cells"][3]
cfg_src = "".join(cfg_cell["source"])
ANCHOR = "# TTA shifts\nTTA_SHIFTS = [-1, 0, 1]\n"
ADDITION = (
    "# TTA shifts (window-axis np.roll on NB4 stream, untouched in exp068)\n"
    "TTA_SHIFTS = [-1, 0, 1]\n"
    "\n"
    "# Delta TTA (exp068, Babych BC25 1st place style): frame-axis shift max on framewise output\n"
    "# Applied in Slot 2 (Tucker SED ONNX) and Slot 3 (exp029 PyTorch) inference\n"
    "# delta=1 = safe / delta=2 = Babych default (frame T sometimes <16 in our setup → start with 1)\n"
    "TTA_DELTA = 1\n"
    "\n"
    "def _delta_max_np(framewise, delta=None):\n"
    "    '''Babych-style delta shift max on (B, T, C) numpy logits.'''\n"
    "    if delta is None: delta = TTA_DELTA\n"
    "    if delta <= 0 or framewise.shape[1] <= 2 * delta + 1:\n"
    "        return framewise.max(axis=1)\n"
    "    base  = framewise.max(axis=1) * 0.5\n"
    "    left  = framewise[:, delta:].max(axis=1) * 0.25\n"
    "    right = framewise[:, :-delta].max(axis=1) * 0.25\n"
    "    return base + left + right\n"
    "\n"
    "def _delta_max_torch(framewise, delta=None):\n"
    "    '''Babych-style delta shift max on (B, T, C) torch logits.'''\n"
    "    if delta is None: delta = TTA_DELTA\n"
    "    if delta <= 0 or framewise.size(1) <= 2 * delta + 1:\n"
    "        return framewise.max(dim=1).values\n"
    "    base  = framewise.max(dim=1).values * 0.5\n"
    "    left  = framewise[:, delta:].max(dim=1).values * 0.25\n"
    "    right = framewise[:, :-delta].max(dim=1).values * 0.25\n"
    "    return base + left + right\n"
)
if ANCHOR not in cfg_src:
    raise SystemExit("CFG anchor not found, abort")
cfg_src_new = cfg_src.replace(ANCHOR, ADDITION)
cfg_cell["source"] = cfg_src_new.splitlines(keepends=True)
print(f"  Cell 3 CFG: TTA_DELTA + helpers inserted ({len(cfg_src)} → {len(cfg_src_new)} chars)")

# ---- 3. Tucker SED cell (cell 10): swap frame_max ----
tucker_cell = nb["cells"][10]
tucker_src = "".join(tucker_cell["source"])
OLD_T = "        frame_max = outs[1].max(axis=1)\n"
NEW_T = "        frame_max = _delta_max_np(outs[1], TTA_DELTA)   # ★ exp068 Delta TTA (frame ±delta shift max)\n"
if OLD_T not in tucker_src:
    raise SystemExit("Tucker SED anchor not found, abort")
tucker_src_new = tucker_src.replace(OLD_T, NEW_T)
tucker_cell["source"] = tucker_src_new.splitlines(keepends=True)
print(f"  Cell 10 Tucker SED: frame_max → _delta_max_np")

# ---- 4. exp029 cell (cell 11): swap _frame_max ----
exp029_cell = nb["cells"][11]
exp029_src = "".join(exp029_cell["source"])
OLD_E = "        _frame_max = _frame.max(dim=1).values\n"
NEW_E = "        _frame_max = _delta_max_torch(_frame, TTA_DELTA)   # ★ exp068 Delta TTA (frame ±delta shift max)\n"
if OLD_E not in exp029_src:
    raise SystemExit("exp029 anchor not found, abort")
exp029_src_new = exp029_src.replace(OLD_E, NEW_E)
exp029_cell["source"] = exp029_src_new.splitlines(keepends=True)
print(f"  Cell 11 exp029: _frame_max → _delta_max_torch")

# ---- Save ----
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nPatched NB saved: {NB}")
