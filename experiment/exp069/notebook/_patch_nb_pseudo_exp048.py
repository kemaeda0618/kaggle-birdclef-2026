"""Patch exp048 blend NB → exp069b pseudo gen NB for BC2026 train_soundscapes.

Changes:
  - Cell 0 (md): Title update
  - Cell 9 (audio loading): test_files = all train_soundscapes (not test_soundscapes)
  - Cell 12 (probs_blend tail): Save NPZ + Kaggle Dataset upload, skip TopN PP / submission CSV
"""
import json, sys, io, os
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def set_md(i, text):
    nb["cells"][i]["cell_type"] = "markdown"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    if "outputs" in nb["cells"][i]: del nb["cells"][i]["outputs"]
    if "execution_count" in nb["cells"][i]: del nb["cells"][i]["execution_count"]


# ─────────── Cell 0: Title ───────────
set_md(0, """# exp069b: exp048 blend pseudo gen on BC2026 train_soundscapes

**Purpose**: exp048 blend (NB4 v11 + Tucker SED + exp029 R3) を BC2026 train_soundscapes (~10,658 files) に推論、
3-way rank blend + sonotype mirror 後の raw probability (n_files × 12 chunks × 234 class) を NPZ 保存。

**Output**: maekeso/birdclef2026-exp069b-exp048-pseudo
  - exp048_raw_234.npz: blend probability (n_files × 12 × 234)
  - primary_labels.json: 234 species column order
  - file_index.json: file_id → row index

**後段**: exp069c で Babych pseudo と per-class merge → 3 pseudo CSV
""")


# ─────────── Cell 9: train_soundscapes loading ───────────
# Replace only the first 5 lines (test_files block)
src9 = "".join(nb["cells"][9]["source"])
OLD_BLOCK = (
    "# === Stage A: NB4 v11 + ResSSM inference ===\n"
    "test_files = sorted(glob.glob(str(TEST_DIR / \"*.ogg\")))\n"
    "if len(test_files) == 0:\n"
    "    print(\"No test files, using train_soundscapes as fallback\")\n"
    "    test_files = sorted(glob.glob(str(TRAIN_SC_DIR / \"*.ogg\")))[:8]\n"
    "print(f\"Test files: {len(test_files)}\")\n"
)
NEW_BLOCK = (
    "# === exp069b: Stage A NB4 v11 + ResSSM inference on TRAIN_SC ===\n"
    "test_files = sorted(glob.glob(str(TRAIN_SC_DIR / \"*.ogg\")))\n"
    "if len(test_files) == 0:\n"
    "    raise RuntimeError(f\"No train_soundscapes files at {TRAIN_SC_DIR}\")\n"
    "print(f\"exp069b: {len(test_files)} train_soundscape files for pseudo gen\")\n"
)
if OLD_BLOCK not in src9:
    raise SystemExit("Cell 9 OLD_BLOCK anchor not found")
src9_new = src9.replace(OLD_BLOCK, NEW_BLOCK)
nb["cells"][9]["source"] = src9_new.splitlines(keepends=True)
nb["cells"][9]["outputs"] = []
nb["cells"][9]["execution_count"] = None
print(f"  Cell 9 patched ({len(src9)} → {len(src9_new)} chars)")


# ─────────── Cell 12: Replace TopN/submission tail with NPZ save + upload ───────────
src12 = "".join(nb["cells"][12]["source"])
# Find "probs_blend = blend_flat.reshape(probs_exp010.shape)" — anchor
ANCHOR = "probs_blend = blend_flat.reshape(probs_exp010.shape)\n"
if ANCHOR not in src12:
    raise SystemExit("Cell 12 probs_blend anchor not found")
# Keep everything up to and including the anchor + print, replace the rest
idx = src12.index(ANCHOR) + len(ANCHOR)
# Also include the print line right after
print_line = "print(f\"blend shape: {probs_blend.shape}, mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}\")\n"
if print_line in src12[idx:idx+200]:
    idx = src12.index(print_line, idx) + len(print_line)

prefix = src12[:idx]
# Replace tail with NPZ save + upload
NEW_TAIL = """
# === exp069b: Save probs_blend NPZ + upload to Kaggle Dataset ===
import json as _json
from kaggle.api.kaggle_api_extended import KaggleApi as _KaggleApi

OUT_DIR_EXP069B = Path("/kaggle/working/exp069b_output")
OUT_DIR_EXP069B.mkdir(parents=True, exist_ok=True)

file_ids = [Path(f).stem for f in test_files]
print(f"Saving probs_blend: shape={probs_blend.shape}, n_files={len(file_ids)}")

import numpy as _np
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

# Upload to Kaggle Dataset
_api = _KaggleApi(); _api.authenticate()
DATASET_SLUG = "maekeso/birdclef2026-exp069b-exp048-pseudo"
_meta = {
    "id": DATASET_SLUG,
    "title": "BC2026 exp069b exp048 blend pseudo (234 BC26 class)",
    "licenses": [{"name": "CC0-1.0"}],
}
with open(OUT_DIR_EXP069B / "dataset-metadata.json", "w") as _f:
    _json.dump(_meta, _f)

try:
    _api.dataset_status(DATASET_SLUG)
    print(f"Dataset {DATASET_SLUG} exists → new version")
    _api.dataset_create_version(folder=str(OUT_DIR_EXP069B),
                                 version_notes="exp069b exp048 pseudo", quiet=False)
except Exception as _e:
    print(f"Dataset {DATASET_SLUG} new → creating ({str(_e)[:60]})")
    _api.dataset_create_new(folder=str(OUT_DIR_EXP069B), public=False, quiet=False)

print(f"OK exp069b pseudo gen DONE: {DATASET_SLUG}")
print(f"Total time: {(time.time()-START)/60:.1f} min")
"""

src12_new = prefix + NEW_TAIL
nb["cells"][12]["source"] = src12_new.splitlines(keepends=True)
nb["cells"][12]["outputs"] = []
nb["cells"][12]["execution_count"] = None
print(f"  Cell 12 patched ({len(src12)} → {len(src12_new)} chars)")


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nPatched: {NB_PATH}")
print(f"Cells: {len(nb['cells'])}")
