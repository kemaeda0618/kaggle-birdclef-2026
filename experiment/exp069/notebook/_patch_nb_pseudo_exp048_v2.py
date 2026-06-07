"""v2 patch for exp069b exp048 pseudo gen NB.

Fixes:
  1. Remove api.dataset_create_new() — kernel auth 不可
  2. Save NPZ to /kaggle/working/ (kernel output)

Note: Cell 9 already uses TRAIN_SC_DIR from exp048's CFG which handles BASE auto-detect.
      So no path fix needed here (exp048 NB already has proper BASE fallback).
"""
import json, sys, io, os
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Find Cell 12 and replace tail (remove dataset upload)
src12 = "".join(nb["cells"][12]["source"])

# Find anchor: probs_blend = ... + print
ANCHOR = "probs_blend = blend_flat.reshape(probs_exp010.shape)\n"
if ANCHOR not in src12:
    raise SystemExit("Cell 12 anchor not found")
idx = src12.index(ANCHOR) + len(ANCHOR)
print_line = "print(f\"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}\")\n"
if print_line in src12[idx:idx+200]:
    idx = src12.index(print_line, idx) + len(print_line)

prefix = src12[:idx]
NEW_TAIL = """
# === exp069b v2: Save probs_blend NPZ to /kaggle/working/ (kernel output) ===
# NOTE: api.dataset_create_new() を削除。Kaggle kernel 内で auth 不可のため。
#       /kaggle/working/ に save した時点で kernel output として保存される。
#       M-R3 NB は kernel_sources で this kernel を attach すれば参照可。

import json as _json
import numpy as _np

OUT_DIR_EXP069B = Path("/kaggle/working")

file_ids = [Path(f).stem for f in test_files]
print(f"Saving probs_blend: shape={probs_blend.shape}, n_files={len(file_ids)}")

_np.savez_compressed(
    OUT_DIR_EXP069B / "exp048_raw_234.npz",
    probs=probs_blend.astype(_np.float16),
    file_ids=_np.array(file_ids),
)
print(f"  NPZ size: {(OUT_DIR_EXP069B / 'exp048_raw_234.npz').stat().st_size/1e6:.1f} MB")

with open(OUT_DIR_EXP069B / "primary_labels.json", "w") as _f:
    _json.dump(list(PRIMARY_LABELS), _f, indent=2)
with open(OUT_DIR_EXP069B / "file_index.json", "w") as _f:
    _json.dump({fid: i for i, fid in enumerate(file_ids)}, _f)

print(f"OK exp069b pseudo gen DONE: outputs at {OUT_DIR_EXP069B}")
print(f"  - exp048_raw_234.npz")
print(f"  - primary_labels.json (234 species)")
print(f"  - file_index.json")
print(f"Total time: {(time.time()-START)/60:.1f} min")
"""

src12_new = prefix + NEW_TAIL
nb["cells"][12]["source"] = src12_new.splitlines(keepends=True)
nb["cells"][12]["outputs"] = []
nb["cells"][12]["execution_count"] = None

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v2 patched: {NB_PATH}")
print(f"Cell 12: {len(src12)} → {len(src12_new)} chars")
