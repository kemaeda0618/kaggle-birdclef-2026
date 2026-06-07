"""Build Kaggle NB: convert Tucker SED 5-fold ONNX -> OpenVINO IR.

Input: tuckerarrants/bc2026-distilled-sed-public (sed_fold0-4.onnx)
Output: /kaggle/working/sed_fold{0..4}.xml + .bin
Verify: onnxruntime vs OpenVINO max diff < 1e-3 per fold
Benchmark: total speedup across 5 folds
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_ov_tucker_sed.ipynb")


def cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1_INSTALL = """# ttahara's wheel pack (openvino 2026.0.0 + onnxruntime + onnx + all deps)
# kernel_sources: ttahara/birdclef-2026-download-wheels
import os, glob
WHEEL_DIR = "/kaggle/input/birdclef-2026-download-wheels/wheels"
assert os.path.isdir(WHEEL_DIR), f"wheel dir missing — attach kernel_sources: ttahara/birdclef-2026-download-wheels  ({WHEEL_DIR})"
print(f"wheels: {len(glob.glob(WHEEL_DIR+'/*.whl'))} files")

try:
    import openvino as ov
    print(f"openvino already available: {ov.__version__}")
except ImportError:
    !pip install -q --no-deps {WHEEL_DIR}/openvino-*.whl {WHEEL_DIR}/openvino_telemetry-*.whl
    import openvino as ov
    print(f"openvino installed from wheel: {ov.__version__}")

try:
    import onnx; print(f"onnx already: {onnx.__version__}")
except ImportError:
    !pip install -q --no-deps {WHEEL_DIR}/onnx-*.whl {WHEEL_DIR}/onnx_ir-*.whl {WHEEL_DIR}/ml_dtypes-*.whl
    import onnx; print(f"onnx installed: {onnx.__version__}")

try:
    import onnxruntime as ort; print(f"onnxruntime already: {ort.__version__}")
except ImportError:
    !pip install -q --no-deps {WHEEL_DIR}/onnxruntime-*.whl {WHEEL_DIR}/flatbuffers-*.whl
    import onnxruntime as ort; print(f"onnxruntime installed: {ort.__version__}")
"""

CELL2_IMPORTS = """import os, sys, json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import openvino as ov

print(f"onnxruntime={ort.__version__}, openvino={ov.__version__}")
"""

CELL3_LOCATE = """# === locate Tucker SED ONNX files ===
TUCKER_CANDIDATES = [
    Path("/kaggle/input/bc2026-distilled-sed-public"),
    Path("/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public"),
]
TUCKER_DIR = None
for p in TUCKER_CANDIDATES:
    if p.exists():
        TUCKER_DIR = p; break
assert TUCKER_DIR is not None, f"tucker dir not found in: {TUCKER_CANDIDATES}"
print(f"TUCKER_DIR: {TUCKER_DIR}")

ONNX_PATHS = []
for fold in range(5):
    hits = list(TUCKER_DIR.rglob(f"sed_fold{fold}.onnx"))
    assert hits, f"sed_fold{fold}.onnx not found"
    ONNX_PATHS.append(hits[0])
    print(f"  fold {fold}: {hits[0]} ({hits[0].stat().st_size/1e6:.1f}MB)")
"""

CELL4_INSPECT = """# === inspect ONNX input/output shapes (fold 0 as reference) ===
sess_ref = ort.InferenceSession(str(ONNX_PATHS[0]), providers=["CPUExecutionProvider"])
print("Inputs:")
for i in sess_ref.get_inputs():
    print(f"  {i.name}: shape={i.shape}, dtype={i.type}")
print("Outputs:")
for o in sess_ref.get_outputs():
    print(f"  {o.name}: shape={o.shape}, dtype={o.type}")

INPUT_NAME = sess_ref.get_inputs()[0].name
INPUT_SHAPE = sess_ref.get_inputs()[0].shape

# Resolve dynamic axes -> concrete shape for testing
def resolve_shape(s, batch=1):
    out = []
    for x in s:
        if isinstance(x, int) and x > 0:
            out.append(x)
        elif x in (None, "batch_size", "batch", "N"):
            out.append(batch)
        else:
            # unknown dynamic dim — default to 1
            out.append(1 if x is None else (int(x) if isinstance(x, int) else 1))
    return tuple(out)

TEST_SHAPE = resolve_shape(INPUT_SHAPE, batch=12)
print(f"Resolved test shape (batch=12): {TEST_SHAPE}")
del sess_ref
"""

CELL5_CONVERT = """# === convert all 5 folds: ONNX -> OpenVINO IR ===
OUT_DIR = Path("/kaggle/working")
IR_PATHS = []
for fold, onnx_path in enumerate(ONNX_PATHS):
    t0 = time.time()
    ov_model = ov.convert_model(str(onnx_path))
    ir_path = OUT_DIR / f"sed_fold{fold}.xml"
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    IR_PATHS.append(ir_path)
    xml_size = ir_path.stat().st_size / 1e6
    bin_size = (OUT_DIR / f"sed_fold{fold}.bin").stat().st_size / 1e6
    print(f"  fold {fold}: .xml={xml_size:.2f}MB .bin={bin_size:.2f}MB ({time.time()-t0:.1f}s)")
print(f"All 5 folds converted.")
"""

CELL6_VERIFY = """# === verify per fold: onnxruntime vs OpenVINO ===
core = ov.Core()
max_diff_per_fold = []
for fold in range(5):
    sess = ort.InferenceSession(str(ONNX_PATHS[fold]), providers=["CPUExecutionProvider"])
    compiled = core.compile_model(str(IR_PATHS[fold]), "CPU")

    max_diff = 0.0
    N_TEST = 2
    for i in range(N_TEST):
        x_np = np.random.randn(*TEST_SHAPE).astype(np.float32)
        ort_outs = sess.run(None, {INPUT_NAME: x_np})
        ov_outs = compiled(x_np)
        # Compare each output (Tucker SED returns 1 or more)
        for j, ort_out in enumerate(ort_outs):
            ov_out = ov_outs[compiled.outputs[j]]
            d = float(np.abs(ort_out - ov_out).max())
            max_diff = max(max_diff, d)
    print(f"  fold {fold}: max diff = {max_diff:.6f} {'[OK]' if max_diff < 1e-3 else '[FAIL]'}")
    max_diff_per_fold.append(max_diff)
    del sess, compiled

assert all(d < 1e-3 for d in max_diff_per_fold), f"some folds failed: {max_diff_per_fold}"
print(f"\\n[OK] all 5 folds verified (max diff < 1e-3)")
"""

CELL7_BENCH = """# === benchmark: onnxruntime vs OpenVINO (1 fold avg) ===
N_ITER = 5
x_np = np.random.randn(*TEST_SHAPE).astype(np.float32)

ort_times = []
ov_times = []
for fold in range(5):
    sess = ort.InferenceSession(str(ONNX_PATHS[fold]), providers=["CPUExecutionProvider"])
    compiled = core.compile_model(str(IR_PATHS[fold]), "CPU")
    # warmup
    _ = sess.run(None, {INPUT_NAME: x_np})
    _ = compiled(x_np)

    t0 = time.time()
    for _ in range(N_ITER):
        _ = sess.run(None, {INPUT_NAME: x_np})
    ort_t = (time.time() - t0) / N_ITER

    t0 = time.time()
    for _ in range(N_ITER):
        _ = compiled(x_np)
    ov_t = (time.time() - t0) / N_ITER

    ort_times.append(ort_t)
    ov_times.append(ov_t)
    print(f"  fold {fold}: ort={ort_t*1000:6.1f}ms, ov={ov_t*1000:6.1f}ms, speedup={ort_t/ov_t:.2f}x")
    del sess, compiled

total_ort = sum(ort_times)
total_ov = sum(ov_times)
print(f"\\n5-fold total per file (12-window batch):")
print(f"  ONNX:     {total_ort*1000:.1f}ms")
print(f"  OpenVINO: {total_ov*1000:.1f}ms")
print(f"  Speedup:  {total_ort/total_ov:.2f}x")
print(f"\\nProjected for 700 test files:")
print(f"  ONNX:     {total_ort*700/60:.1f} min")
print(f"  OpenVINO: {total_ov*700/60:.1f} min")
"""

CELL8_LIST = """# === list output files ===
out_dir = Path("/kaggle/working")
print(f"Files in {out_dir}:")
total = 0
for f in sorted(out_dir.iterdir()):
    if f.is_file():
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:30s} {sz/1e6:8.2f}MB")
print(f"  {'TOTAL':30s} {total/1e6:8.2f}MB")
"""

CELLS = [
    (CELL1_INSTALL, "cell-install"),
    (CELL2_IMPORTS, "cell-imports"),
    (CELL3_LOCATE, "cell-locate"),
    (CELL4_INSPECT, "cell-inspect"),
    (CELL5_CONVERT, "cell-convert"),
    (CELL6_VERIFY, "cell-verify"),
    (CELL7_BENCH, "cell-bench"),
    (CELL8_LIST, "cell-list"),
]

nb = {
    "cells": [cell(s, cid) for s, cid in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes, {len(CELLS)} cells)")

for i, (src, cid) in enumerate(CELLS):
    if cid == "cell-install":
        continue
    try:
        compile(src, f"<{cid}>", "exec")
        print(f"  [OK] {cid}")
    except SyntaxError as e:
        print(f"  [FAIL] {cid}: {e}")
