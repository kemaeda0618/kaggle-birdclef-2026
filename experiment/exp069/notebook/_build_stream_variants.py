"""Build 3 stream-specific NB variants from exp069b base for Pattern B-CPU parallel.

Variants:
  nb_pseudo_nb4.ipynb     - STREAM_ID="nb4"    - NB4 v11 (Perch + ProtoSSM + ResidualSSM)
  nb_pseudo_tucker.ipynb  - STREAM_ID="tucker" - Tucker SED 5-fold ONNX
  nb_pseudo_exp029.ipynb  - STREAM_ID="exp029" - exp029 R3 eca_nfnet_l1

Each variant:
  - Cell 3 (CFG) に STREAM_ID = "{stream}" 追加
  - Cell 7 (ProtoSSM training): wrap with `if STREAM_ID == "nb4":` (他 stream は skip)
  - Cell 9 (exp010 inference loop): wrap with `if STREAM_ID == "nb4":`
  - Cell 10 (Tucker SED): wrap with `if STREAM_ID == "tucker":`
  - Cell 11 (exp029 R3): wrap with `if STREAM_ID == "exp029":`
  - Cell 12 (blend): replace with per-stream NPZ save
"""
import json, sys, io, os, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_NB = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
STREAMS = ["nb4", "tucker", "exp029"]


def build_variant(stream_id):
    """Build 1 variant NB from base, with STREAM_ID hard-coded."""
    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))

    # ─── Cell 0: title md ───
    title_md = f"""# exp069b-{stream_id}: Single stream pseudo gen on BC2026 train_soundscapes (CPU)

**Pattern B-CPU parallel**: 3 stream を別 NB で独立並列実行、CPU mode で各 ~8.6h。
**STREAM_ID**: `{stream_id}` (この NB は {stream_id} stream のみ実行)

**Output**: /kaggle/working/{stream_id}_raw_234.npz (n_files × 12 × 234 BC26 class)
  + primary_labels.json + file_index.json + stream_meta.json

**他 2 NB (3 並列実行)**:
  - exp069b-nb4: NB4 v11 (ProtoSSM + ResidualSSM)
  - exp069b-tucker: Tucker SED 5-fold
  - exp069b-exp029: exp029 R3 (eca_nfnet_l1)

**後段**: exp069c で 3 NPZ を blend (rank-avg + sonotype mirror) → full pseudo
"""
    nb["cells"][0]["source"] = title_md.splitlines(keepends=True)
    nb["cells"][0]["cell_type"] = "markdown"
    nb["cells"][0].pop("outputs", None)
    nb["cells"][0].pop("execution_count", None)

    # ─── Cell 3 (CFG): add STREAM_ID ───
    cfg_src = "".join(nb["cells"][3]["source"])
    # Insert STREAM_ID at top of CFG cell
    stream_marker = f'\n# ★ Pattern B-CPU: this NB runs ONLY one stream\nSTREAM_ID = "{stream_id}"  # "nb4" / "tucker" / "exp029"\nprint(f"[exp069b-{stream_id}] STREAM_ID = {{STREAM_ID}}")\n\n'
    cfg_src_new = stream_marker + cfg_src
    nb["cells"][3]["source"] = cfg_src_new.splitlines(keepends=True)
    nb["cells"][3]["outputs"] = []
    nb["cells"][3]["execution_count"] = None

    # ─── Cell 7 (ProtoSSM training): guard with STREAM_ID == "nb4" ───
    cell7_src = "".join(nb["cells"][7]["source"])
    guarded = (
        'if STREAM_ID == "nb4":\n'
        + "\n".join("    " + l for l in cell7_src.split("\n"))
        + '\nelse:\n    print(f"[Skip] ProtoSSM training (STREAM_ID={STREAM_ID})")\n'
    )
    nb["cells"][7]["source"] = guarded.splitlines(keepends=True)
    nb["cells"][7]["outputs"] = []
    nb["cells"][7]["execution_count"] = None

    # ─── Cell 8 (Perch ONNX for test inference): guard with nb4 ───
    cell8_src = "".join(nb["cells"][8]["source"])
    guarded8 = (
        'if STREAM_ID == "nb4":\n'
        + "\n".join("    " + l for l in cell8_src.split("\n"))
        + '\nelse:\n    print(f"[Skip] Perch ONNX setup (STREAM_ID={STREAM_ID})")\n'
    )
    nb["cells"][8]["source"] = guarded8.splitlines(keepends=True)
    nb["cells"][8]["outputs"] = []
    nb["cells"][8]["execution_count"] = None

    # ─── Cell 9: split audio_cache load (always) vs exp010 inference (nb4 only) ───
    # The cell does: test_files setup + audio_cache build + exp010 inference loop
    # For non-nb4 variants, we need audio_cache but not exp010 inference
    cell9_src = "".join(nb["cells"][9]["source"])
    # Find the exp010 inference loop anchor
    EXP010_START = "for fi, fpath in enumerate(test_files):"
    if EXP010_START in cell9_src:
        idx = cell9_src.index(EXP010_START)
        # Find the indentation level of this line
        line_start = cell9_src.rfind("\n", 0, idx) + 1
        indent = cell9_src[line_start:idx]
        # The exp010 loop ends at end of cell 9
        # Wrap from EXP010_START to end with STREAM_ID check
        prefix = cell9_src[:idx]
        suffix = cell9_src[idx:]
        # Add wrapper
        wrapper_open = f'{indent}if STREAM_ID == "nb4":\n'
        # Indent suffix lines
        suffix_indented = "\n".join(("    " + l) if l.strip() else l for l in suffix.split("\n"))
        cell9_new = prefix + wrapper_open + suffix_indented + f'\n{indent}else:\n{indent}    print(f"[Skip] exp010 inference loop (STREAM_ID={{STREAM_ID}})")\n'
        nb["cells"][9]["source"] = cell9_new.splitlines(keepends=True)
        nb["cells"][9]["outputs"] = []
        nb["cells"][9]["execution_count"] = None

    # ─── Cell 10 (Tucker SED): guard with STREAM_ID == "tucker" ───
    cell10_src = "".join(nb["cells"][10]["source"])
    guarded10 = (
        'if STREAM_ID == "tucker":\n'
        + "\n".join("    " + l for l in cell10_src.split("\n"))
        + '\nelse:\n    print(f"[Skip] Tucker SED stream (STREAM_ID={STREAM_ID})")\n    probs_sed = None\n'
    )
    nb["cells"][10]["source"] = guarded10.splitlines(keepends=True)
    nb["cells"][10]["outputs"] = []
    nb["cells"][10]["execution_count"] = None

    # ─── Cell 11 (exp029 R3): guard with STREAM_ID == "exp029" ───
    cell11_src = "".join(nb["cells"][11]["source"])
    guarded11 = (
        'if STREAM_ID == "exp029":\n'
        + "\n".join("    " + l for l in cell11_src.split("\n"))
        + '\nelse:\n    print(f"[Skip] exp029 R3 stream (STREAM_ID={STREAM_ID})")\n    probs_e17 = None\n'
    )
    nb["cells"][11]["source"] = guarded11.splitlines(keepends=True)
    nb["cells"][11]["outputs"] = []
    nb["cells"][11]["execution_count"] = None

    # ─── Cell 12: replace blend with per-stream save ───
    save_logic = f"""# === exp069b-{stream_id}: Per-stream NPZ save ===
import json as _json
import numpy as _np
from pathlib import Path

OUT_DIR_STREAM = Path("/kaggle/working")
STREAM_ID_LOCAL = STREAM_ID  # for safety
print(f"[Save] STREAM_ID = {{STREAM_ID_LOCAL}}")

# Pick stream-specific probability array
if STREAM_ID_LOCAL == "nb4":
    stream_probs = probs_exp010  # (n_files, 12, 234) from exp010 stream
    save_name = "nb4_raw_234.npz"
elif STREAM_ID_LOCAL == "tucker":
    stream_probs = probs_sed  # (n_files, 12, 234) from Tucker SED
    save_name = "tucker_raw_234.npz"
elif STREAM_ID_LOCAL == "exp029":
    stream_probs = probs_e17  # (n_files, 12, 234) from exp029 R3
    save_name = "exp029_raw_234.npz"
else:
    raise ValueError(f"Unknown STREAM_ID: {{STREAM_ID_LOCAL}}")

assert stream_probs is not None, f"stream_probs is None for STREAM_ID={{STREAM_ID_LOCAL}}"
print(f"  stream_probs shape: {{stream_probs.shape}}")

file_ids = [Path(f).stem for f in test_files]

_np.savez_compressed(
    OUT_DIR_STREAM / save_name,
    probs=stream_probs.astype(_np.float16),
    file_ids=_np.array(file_ids),
)
print(f"  Saved: {{save_name}} ({{(OUT_DIR_STREAM / save_name).stat().st_size/1e6:.1f}} MB)")

with open(OUT_DIR_STREAM / "primary_labels.json", "w") as _f:
    _json.dump(list(PRIMARY_LABELS), _f, indent=2)
with open(OUT_DIR_STREAM / "file_index.json", "w") as _f:
    _json.dump({{fid: i for i, fid in enumerate(file_ids)}}, _f)
with open(OUT_DIR_STREAM / "stream_meta.json", "w") as _f:
    _json.dump({{"stream_id": STREAM_ID_LOCAL, "n_files": len(file_ids)}}, _f)

print(f"OK exp069b-{{STREAM_ID_LOCAL}} DONE: total time {{(time.time()-START)/60:.1f}} min")
"""
    nb["cells"][12]["source"] = save_logic.splitlines(keepends=True)
    nb["cells"][12]["outputs"] = []
    nb["cells"][12]["execution_count"] = None

    # Write variant NB
    out_path = BASE_NB.with_name(f"nb_pseudo_{stream_id}.ipynb")
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Built: {out_path.name}")
    return out_path


for stream in STREAMS:
    build_variant(stream)

print(f"\n3 variant NBs built")
