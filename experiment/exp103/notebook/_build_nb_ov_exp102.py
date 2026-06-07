"""Build Kaggle NB: convert exp102 3-fold .pth ckpts -> ONNX -> OpenVINO IR.

Input: maekeso/birdclef2026-exp102-l1-3fold-r3p (r3_fold{0,1,2}_ckpt_best_ns22.pth)
       ttahara/birdclef-2026-download-wheels (openvino wheels)
Output: /kaggle/working/exp102_fold{0,1,2}.xml + .bin

Architecture identical to exp029 R3 (eca_nfnet_l1 + Perch distill 1536d, GeMFreqPool, dense+att+cla)
Only diff: drop_path_rate=0.15 in training (no effect on inference)

Critical fix carried over from OV1:
- ScaledStdConv2d monkey-patch (avoid BatchNormalization training_mode=1 in ONNX)
- training=TrainingMode.EVAL + dynamo=False
- partial_shape for compiled.inputs/outputs (dynamic batch)
- Fold missing skip (run with partial folds 0/1 if fold 2 not yet trained)
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_ov_exp102.ipynb")


def cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1_INSTALL = """# openvino install (offline via ttahara wheels OR fallback to internet pip)
import os, glob
from pathlib import Path

# Debug: list all input mounts
print("=== /kaggle/input/ contents ===")
for p in sorted(Path("/kaggle/input").iterdir()):
    print(f"  {p.name}/")
print()

# Auto-search for openvino wheel (kernel_sources mount path may vary)
WHEEL_DIR = None
hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
if hits:
    WHEEL_DIR = str(Path(hits[0]).parent)
    print(f"[wheels] found at: {WHEEL_DIR}")
    print(f"  ({len(glob.glob(WHEEL_DIR + '/*.whl'))} wheels total)")
else:
    print("[wheels] not found anywhere under /kaggle/input/")
    print("[wheels] will fallback to internet pip install (enable_internet=True)")

try:
    import openvino as ov
    print(f"openvino already available: {ov.__version__}")
except ImportError:
    if WHEEL_DIR:
        !pip install -q --no-deps {WHEEL_DIR}/openvino-*.whl {WHEEL_DIR}/openvino_telemetry-*.whl
    else:
        !pip install -q openvino==2026.0.0
    import openvino as ov
    print(f"openvino installed: {ov.__version__}")

try:
    import onnx; print(f"onnx already: {onnx.__version__}")
except ImportError:
    if WHEEL_DIR:
        !pip install -q --no-deps {WHEEL_DIR}/onnx-*.whl {WHEEL_DIR}/onnx_ir-*.whl {WHEEL_DIR}/ml_dtypes-*.whl
    else:
        !pip install -q onnx
    import onnx; print(f"onnx installed: {onnx.__version__}")

try:
    import onnxruntime as ort; print(f"onnxruntime already: {ort.__version__}")
except ImportError:
    if WHEEL_DIR:
        !pip install -q --no-deps {WHEEL_DIR}/onnxruntime-*.whl {WHEEL_DIR}/flatbuffers-*.whl
    else:
        !pip install -q onnxruntime
    import onnxruntime as ort; print(f"onnxruntime installed: {ort.__version__}")
"""

CELL2_IMPORTS = """import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import timm
import openvino as ov
import onnx

print(f"torch={torch.__version__}, openvino={ov.__version__}, onnx={onnx.__version__}")
"""

CELL3_CONST = """# === constants (matches exp102 training config) ===
SR              = 32_000
N_CLASSES       = 234
BACKBONE        = "eca_nfnet_l1"
N_MELS          = 256
N_FFT           = 2048
HOP             = 512
FMIN            = 20
FMAX            = 16000
TRAIN_SAMPLES   = SR * 5
USE_DISTILL     = True
PERCH_DIM       = 1536
N_TF_DIM        = TRAIN_SAMPLES // HOP + 1   # 313
N_WINDOWS       = 12
FOLDS           = [0, 1, 2]                  # try all 3, skip if missing
print(f"N_TF_DIM={N_TF_DIM}, mel input shape per chunk = (1, 1, {N_MELS}, {N_TF_DIM})")
"""

CELL4_LOCATE = """# === locate ckpts (skip fold if missing — supports partial upload) ===
CKPT_DIR_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp102-l1-3fold-r3p"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp102-l1-3fold-r3p"),
]
CKPT_DIR = None
for p in CKPT_DIR_CANDIDATES:
    if p.exists():
        CKPT_DIR = p; break
assert CKPT_DIR is not None, f"ckpt dir not found in: {CKPT_DIR_CANDIDATES}"
print(f"CKPT_DIR: {CKPT_DIR}")

CKPT_PATHS = {}
for k in FOLDS:
    name = f"r3_fold{k}_ckpt_best_ns22.pth"
    hits = list(CKPT_DIR.rglob(name))
    if hits:
        CKPT_PATHS[k] = hits[0]
        print(f"  fold {k}: {hits[0]} ({hits[0].stat().st_size/1e6:.1f}MB)")
    else:
        print(f"  fold {k}: [MISSING] {name} — will skip")

assert CKPT_PATHS, "no ckpts found — check dataset"
print(f"\\nFolds to convert: {sorted(CKPT_PATHS.keys())}")
"""

CELL5_ARCH = """# === architecture (matches exp102 BirdSEDModel = exp029 _E17SED) ===
class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.15, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, N_MELS, N_TF_DIM)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if USE_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if USE_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


class _ExportWrap(nn.Module):
    \"\"\"ONNX export wrapper: always return (clip, frame).\"\"\"
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, mel):
        clip, frame = self.m(mel, return_framewise=True)
        return clip, frame
"""

CELL6_PATCH = """# === Monkey-patch ScaledStdConv2d (NFNet Weight Standardization) ===
# Reason: timm's ScaledStdConv2d uses F.batch_norm(training=True) internally,
# which ONNX exports as BatchNormalization(training_mode=1) — OpenVINO refuses.
import torch.nn.functional as F
from timm.layers.std_conv import ScaledStdConv2d
_ssc_classes = [ScaledStdConv2d]
try:
    from timm.layers.std_conv import ScaledStdConv2dSame
    _ssc_classes.append(ScaledStdConv2dSame)
except ImportError:
    pass

def _ssc_forward_no_bn(self, x):
    w = self.weight.reshape(self.out_channels, -1)
    w_mean = w.mean(dim=1, keepdim=True)
    w_var = ((w - w_mean) ** 2).mean(dim=1, keepdim=True)
    w_norm = (w - w_mean) * torch.rsqrt(w_var + self.eps)
    gain_scale = (self.gain * self.scale).view(-1, 1)
    weight = (w_norm * gain_scale).reshape_as(self.weight)
    return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

for cls in _ssc_classes:
    cls.forward = _ssc_forward_no_bn
print(f"Patched {len(_ssc_classes)} ScaledStdConv2d classes (no F.batch_norm in forward)")
"""

CELL7_CONVERT = """# === per-fold loop: load -> ONNX -> IR -> verify -> benchmark ===
OUT_DIR = Path("/kaggle/working")
IR_PATHS = {}
metrics_per_fold = []
core = ov.Core()

for fold in sorted(CKPT_PATHS.keys()):
    print(f"\\n{'='*60}\\n  fold {fold}\\n{'='*60}")
    t_fold_start = time.time()
    ckpt_path = CKPT_PATHS[fold]

    # --- load ckpt ---
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    val = state.get('best_ns22', float('nan'))
    print(f"  ckpt epoch={state.get('epoch')}, best_ns22={val:.4f}")

    model = BirdSEDModel().cpu()
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    for sm in model.modules(): sm.eval()
    assert sum(1 for m in model.modules() if m.training) == 0

    export_model = _ExportWrap(model).eval()

    # --- sanity forward ---
    with torch.no_grad():
        sanity_out = export_model(torch.randn(2, 1, N_MELS, N_TF_DIM))
        assert sanity_out[0].shape == (2, N_CLASSES) and sanity_out[1].shape[0] == 2

    # --- export ONNX ---
    onnx_path = str(OUT_DIR / f"exp102_fold{fold}.onnx")
    dummy = torch.randn(1, 1, N_MELS, N_TF_DIM)
    with torch.no_grad():
        torch.onnx.export(
            export_model, dummy, onnx_path,
            input_names=["mel"], output_names=["clip", "frame"],
            dynamic_axes={"mel": {0: "batch"}, "clip": {0: "batch"}, "frame": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
    onnx_mb = Path(onnx_path).stat().st_size / 1e6
    print(f"  ONNX: {onnx_path} ({onnx_mb:.1f}MB)")

    # --- verify ONNX has no BN with training_mode=1 ---
    _m = onnx.load(onnx_path)
    _bn_t1 = [n for n in _m.graph.node if n.op_type == "BatchNormalization"
              and any(a.name == "training_mode" and a.i == 1 for a in n.attribute)]
    assert len(_bn_t1) == 0, f"fold {fold}: {len(_bn_t1)} BN with training_mode=1 in ONNX"

    # --- convert to OpenVINO IR ---
    ov_model = ov.convert_model(onnx_path)
    ir_path = str(OUT_DIR / f"exp102_fold{fold}.xml")
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    bin_path = ir_path.replace(".xml", ".bin")
    print(f"  IR: {ir_path}")
    print(f"    .xml={Path(ir_path).stat().st_size/1e6:.2f}MB")
    print(f"    .bin={Path(bin_path).stat().st_size/1e6:.2f}MB")
    IR_PATHS[fold] = ir_path

    # --- verify torch vs OV (max diff < 1e-3) ---
    compiled = core.compile_model(ir_path, "CPU")
    max_diff_clip = 0.0; max_diff_frame = 0.0
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(N_WINDOWS, 1, N_MELS, N_TF_DIM)
            clip_pt, frame_pt = export_model(x)
            ov_out = compiled(x.numpy())
            clip_ov = ov_out[compiled.outputs[0]]
            frame_ov = ov_out[compiled.outputs[1]]
            max_diff_clip = max(max_diff_clip, float(np.abs(clip_pt.numpy() - clip_ov).max()))
            max_diff_frame = max(max_diff_frame, float(np.abs(frame_pt.numpy() - frame_ov).max()))
    print(f"  verify: max diff clip={max_diff_clip:.6f}, frame={max_diff_frame:.6f}")
    assert max_diff_clip < 1e-3 and max_diff_frame < 1e-3, f"fold {fold}: diff too large"

    # --- benchmark ---
    N_ITER = 5
    x_np = np.random.randn(N_WINDOWS, 1, N_MELS, N_TF_DIM).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    with torch.no_grad():
        _ = export_model(x_t)
    _ = compiled(x_np)

    with torch.no_grad():
        t0 = time.time()
        for _ in range(N_ITER):
            _ = export_model(x_t)
        pt_t = (time.time() - t0) / N_ITER

    t0 = time.time()
    for _ in range(N_ITER):
        _ = compiled(x_np)
    ov_t = (time.time() - t0) / N_ITER
    print(f"  bench: PyTorch={pt_t*1000:.1f}ms, OpenVINO={ov_t*1000:.1f}ms, speedup={pt_t/ov_t:.2f}x")

    # cleanup (free RAM)
    del compiled, ov_model, model, export_model, state
    import gc; gc.collect()

    metrics_per_fold.append({
        "fold": fold, "val_ns22": val,
        "onnx_mb": onnx_mb,
        "pt_ms": pt_t * 1000, "ov_ms": ov_t * 1000,
        "speedup": pt_t / ov_t,
        "max_diff_clip": max_diff_clip, "max_diff_frame": max_diff_frame,
        "elapsed_s": time.time() - t_fold_start,
    })

print(f"\\n{'='*60}\\nAll {len(IR_PATHS)} folds converted")
"""

CELL8_SUMMARY = """# === summary table ===
print(f"\\n{'fold':>6} {'val_ns22':>10} {'pt_ms':>10} {'ov_ms':>10} {'speedup':>8} {'diff_clip':>10} {'elapsed_s':>10}")
for r in metrics_per_fold:
    print(f"{r['fold']:>6d} {r['val_ns22']:>10.4f} {r['pt_ms']:>10.1f} {r['ov_ms']:>10.1f} "
          f"{r['speedup']:>8.2f} {r['max_diff_clip']:>10.6f} {r['elapsed_s']:>10.1f}")
print(f"\\nTotal time (all folds): {sum(r['elapsed_s'] for r in metrics_per_fold):.1f}s")
"""

CELL9_LIST = """# === list output files ===
out_dir = Path("/kaggle/working")
print(f"\\nFiles in {out_dir}:")
total = 0
for f in sorted(out_dir.iterdir()):
    if f.is_file() and f.suffix in (".onnx", ".xml", ".bin"):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:35s} {sz/1e6:8.2f}MB")
print(f"  {'TOTAL':35s} {total/1e6:8.2f}MB")
"""

CELLS = [
    (CELL1_INSTALL, "cell-install"),
    (CELL2_IMPORTS, "cell-imports"),
    (CELL3_CONST, "cell-const"),
    (CELL4_LOCATE, "cell-locate"),
    (CELL5_ARCH, "cell-arch"),
    (CELL6_PATCH, "cell-patch"),
    (CELL7_CONVERT, "cell-convert"),
    (CELL8_SUMMARY, "cell-summary"),
    (CELL9_LIST, "cell-list"),
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
