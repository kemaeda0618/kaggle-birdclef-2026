"""Patch nb_pseudo_nb4.ipynb: convert exp069b base to NB4 v11 only streaming.

Changes:
  1. Revert GPU/CUDA patches → CPU mode (avoid device error)
  2. Remove audio_cache.append → streaming (memory OOM 回避)
  3. Replace cells 10 (Tucker) and 11 (exp029) with no-op
  4. Replace cell 12 (blend) with save probs_e10 NPZ only
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_nb4.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def set_code(i, text):
    nb["cells"][i]["cell_type"] = "code"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None


def set_md(i, text):
    nb["cells"][i]["cell_type"] = "markdown"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    if "outputs" in nb["cells"][i]: del nb["cells"][i]["outputs"]
    if "execution_count" in nb["cells"][i]: del nb["cells"][i]["execution_count"]


# ─────────── Cell 0: Title update ───────────
set_md(0, """# exp069b-nb4: NB4 v11 (Perch + ProtoSSM + Konbu) pseudo gen (streaming, CPU)

**Pattern B-CPU parallel stream NB**: NB4 v11 stream のみ実行、streaming で full 10,658 files カバー。

**Output**: `/kaggle/working/nb4_raw_234.npz` (10658, 12, 234) float16

**並行 NB**:
  - nb_pseudo_tucker (Tucker SED 5-fold) — pushed
  - nb_pseudo_exp029 (exp029 R3) — pushed

**後段**: exp069c で 3 NPZ blend
""")


# ─────────── Cell 2: Revert DEVICE to CPU ───────────
src2 = "".join(nb["cells"][2]["source"])
src2_new = src2
GPU_BLOCK_OLD = ('import torch as _torch_init\n'
                 'if _torch_init.cuda.is_available():\n'
                 '    _torch_init.set_default_device("cuda")\n'
                 '    DEVICE = "cuda"\n'
                 'else:\n'
                 '    DEVICE = "cpu"\n'
                 'print(f"[exp069b v8] DEVICE = {DEVICE} (set_default_device set)")\n')
CPU_BLOCK_NEW = 'DEVICE = "cpu"\nprint(f"[nb_pseudo_nb4] DEVICE = {DEVICE} (CPU streaming mode)")\n'
if GPU_BLOCK_OLD in src2_new:
    src2_new = src2_new.replace(GPU_BLOCK_OLD, CPU_BLOCK_NEW)
    print("  Cell 2: GPU → CPU reverted")
nb["cells"][2]["source"] = src2_new.splitlines(keepends=True)
nb["cells"][2]["outputs"] = []
nb["cells"][2]["execution_count"] = None


# ─────────── Cell 7: numpy RNG patch fix (CPU compatible) ───────────
# Keep np.random.RandomState (works on CPU)


# ─────────── Cell 8: ONNX providers revert to CPU only ───────────
src8 = "".join(nb["cells"][8]["source"])
src8_new = src8.replace(
    'providers=["CUDAExecutionProvider", "CPUExecutionProvider"])',
    'providers=["CPUExecutionProvider"])',
)
if src8 != src8_new:
    nb["cells"][8]["source"] = src8_new.splitlines(keepends=True)
    nb["cells"][8]["outputs"] = []
    nb["cells"][8]["execution_count"] = None
    print("  Cell 8: ONNX CUDA → CPU only")


# ─────────── Cell 9: Streaming refactor (remove audio_cache.append) ───────────
src9 = "".join(nb["cells"][9]["source"])
src9_new = src9
# Remove audio_cache list initialization (keep it as empty list for downstream cells, OR replace with None)
# The issue: audio_cache.append(raw_60s) accumulates audio. Just SKIP that line.
# Find and comment out the audio_cache.append line(s).
OLD_APPEND = "        audio_cache.append(raw_60s)\n"
NEW_NO_APPEND = "        # ★ streaming: skip audio_cache.append (memory OOM avoidance for full 10,658 files)\n"
if OLD_APPEND in src9_new:
    src9_new = src9_new.replace(OLD_APPEND, NEW_NO_APPEND)
    print("  Cell 9: audio_cache.append removed (streaming mode)")
# Also remove any other audio_cache references that might be problematic
# audio_cache = [] init stays (other cells expect this variable but it'll just be empty)
nb["cells"][9]["source"] = src9_new.splitlines(keepends=True)
nb["cells"][9]["outputs"] = []
nb["cells"][9]["execution_count"] = None


# ─────────── Cell 10: Tucker SED — replace with no-op ───────────
set_code(10, """# === nb_pseudo_nb4: Tucker SED SKIPPED (this NB is NB4 v11 only) ===
print("[Skip] Tucker SED stream (this NB runs NB4 v11 only)")
""")


# ─────────── Cell 11: exp029 R3 — replace with no-op ───────────
set_code(11, """# === nb_pseudo_nb4: exp029 R3 SKIPPED (this NB is NB4 v11 only) ===
print("[Skip] exp029 R3 stream (this NB runs NB4 v11 only)")
""")


# ─────────── Cell 12: Replace blend with save probs_e10 NPZ ───────────
set_code(12, """# === nb_pseudo_nb4: Save probs_exp010 NPZ ===
import json as _json
import numpy as _np

OUT_DIR = Path("/kaggle/working")
print(f"Saving probs_exp010: shape={probs_exp010.shape}")

file_ids = [Path(f).stem for f in test_files]
_np.savez_compressed(
    OUT_DIR / "nb4_raw_234.npz",
    probs=probs_exp010.astype(_np.float16),
    file_ids=_np.array(file_ids),
)
print(f"Saved nb4_raw_234.npz: {(OUT_DIR / 'nb4_raw_234.npz').stat().st_size/1e6:.1f} MB")

with open(OUT_DIR / "primary_labels.json", "w") as _f:
    _json.dump(list(PRIMARY_LABELS), _f, indent=2)
with open(OUT_DIR / "file_index.json", "w") as _f:
    _json.dump({fid: i for i, fid in enumerate(file_ids)}, _f)
with open(OUT_DIR / "stream_meta.json", "w") as _f:
    _json.dump({"stream_id": "nb4", "n_files": len(file_ids), "shape": list(probs_exp010.shape)}, _f)

print(f"OK exp069b-nb4 DONE: total {(time.time()-START)/60:.1f} min")
""")


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"nb4 NB patched: {NB_PATH}")
