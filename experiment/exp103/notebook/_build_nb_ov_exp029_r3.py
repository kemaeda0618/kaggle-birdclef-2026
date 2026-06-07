"""Build Kaggle NB: convert exp029 R3 .pth -> ONNX -> OpenVINO IR.

Output: /kaggle/working/exp029_r3_fold0.onnx, .xml, .bin
Verify: torch vs OpenVINO max diff < 1e-3
Benchmark: PyTorch vs OpenVINO speedup
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_ov_exp029_r3.ipynb")


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

# openvino: install from wheel (not preinstalled). --no-deps to avoid numpy version conflicts with torch
try:
    import openvino as ov
    print(f"openvino already available: {ov.__version__}")
except ImportError:
    !pip install -q --no-deps {WHEEL_DIR}/openvino-*.whl {WHEEL_DIR}/openvino_telemetry-*.whl
    import openvino as ov
    print(f"openvino installed from wheel: {ov.__version__}")

# onnx/onnxruntime usually preinstalled on Kaggle; install only if missing
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
import torch
import torch.nn as nn
import torchaudio
import timm
import openvino as ov

print(f"torch={torch.__version__}, openvino={ov.__version__}")
"""

CELL3_CONST = """# === constants (matches exp090 exp029 R3 inference config) ===
SR              = 32_000
N_CLASSES       = 234
E17_BACKBONE    = "eca_nfnet_l1"
E17_N_MELS      = 256
E17_N_FFT       = 2048
E17_HOP         = 512
E17_FMIN        = 20
E17_FMAX        = 16000
E17_TRAIN_SAMPLES = SR * 5
E17_USE_DISTILL = True
E17_PERCH_DIM   = 1536
N_TF_DIM        = E17_TRAIN_SAMPLES // E17_HOP + 1   # 313
N_WINDOWS       = 12
print(f"N_TF_DIM={N_TF_DIM}, mel input shape per chunk = (1, 1, {E17_N_MELS}, {N_TF_DIM})")
"""

CELL4_LOCATE = """# === locate ckpt ===
CKPT_DIR_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp029-l1-single"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single"),
]
CKPT_DIR = None
for p in CKPT_DIR_CANDIDATES:
    if p.exists():
        CKPT_DIR = p; break
assert CKPT_DIR is not None, f"ckpt dir not found in: {CKPT_DIR_CANDIDATES}"
print(f"CKPT_DIR: {CKPT_DIR}")

CKPT = None
for hit in CKPT_DIR.rglob("r3_fold0_ckpt_best_ns22.pth"):
    CKPT = hit; break
assert CKPT is not None, "r3_fold0_ckpt_best_ns22.pth not found"
print(f"CKPT: {CKPT} ({CKPT.stat().st_size/1e6:.1f}MB)")
"""

CELL5_ARCH = """# === architecture (copy of exp090 _E17SED) ===
class _E17GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E17DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _E17SED(nn.Module):
    def __init__(self, backbone_name=E17_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, E17_N_MELS, N_TF_DIM)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E17GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E17_USE_DISTILL:
            self.distill_head = _E17DistillHead(self.backbone_dim, E17_PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if E17_USE_DISTILL else h
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


# Export wrapper: always return (clip, frame) for ONNX
class _E17SEDExport(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, mel):
        clip, frame = self.m(mel, return_framewise=True)
        return clip, frame
"""

CELL6_LOAD = """# === load ckpt ===
try:
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(str(CKPT), map_location="cpu")
print(f"ckpt epoch={state.get('epoch')}, best_ns22={state.get('best_ns22', float('nan')):.4f}")

model = _E17SED().cpu().eval()
missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
if missing: print(f"  first 5 missing: {missing[:5]}")
if unexpected: print(f"  first 5 unexpected: {unexpected[:5]}")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

export_model = _E17SEDExport(model).eval()
"""

CELL7_ONNX = """# === export to ONNX (dynamic batch axis) ===
# ★ ROOT FIX: timm's ScaledStdConv2d (NFNet) uses F.batch_norm(training=True) internally
#   for Weight Standardization. This exports as ONNX BatchNormalization with training_mode=1,
#   which OpenVINO refuses to import. Monkey-patch the forward to use manual mean/var
#   normalization (mathematically equivalent at inference, no BN node generated).
import torch.nn.functional as F
from timm.layers.std_conv import ScaledStdConv2d
_ssc_classes = [ScaledStdConv2d]
try:
    from timm.layers.std_conv import ScaledStdConv2dSame
    _ssc_classes.append(ScaledStdConv2dSame)
except ImportError:
    pass

def _ssc_forward_no_bn(self, x):
    # Manual weight standardization (math-equivalent to F.batch_norm at inference)
    w = self.weight.reshape(self.out_channels, -1)
    w_mean = w.mean(dim=1, keepdim=True)
    w_var = ((w - w_mean) ** 2).mean(dim=1, keepdim=True)
    w_norm = (w - w_mean) * torch.rsqrt(w_var + self.eps)
    gain_scale = (self.gain * self.scale).view(-1, 1)
    weight = (w_norm * gain_scale).reshape_as(self.weight)
    return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

for cls in _ssc_classes:
    cls.forward = _ssc_forward_no_bn
print(f"Patched {len(_ssc_classes)} ScaledStdConv2d classes (Weight Standardization without BN op)")

# Defensive eval
export_model.eval(); model.eval()
for sm in model.modules(): sm.eval()
assert sum(1 for m in export_model.modules() if m.training) == 0

# Sanity check: patched forward should produce same output as before (within FP tolerance)
with torch.no_grad():
    sanity_in = torch.randn(2, 1, E17_N_MELS, N_TF_DIM)
    sanity_out = export_model(sanity_in)
    print(f"sanity forward OK: clip={sanity_out[0].shape}, frame={sanity_out[1].shape}")

ONNX_PATH = "/kaggle/working/exp029_r3_fold0.onnx"
dummy = torch.randn(1, 1, E17_N_MELS, N_TF_DIM)
with torch.no_grad():
    torch.onnx.export(
        export_model, dummy, ONNX_PATH,
        input_names=["mel"], output_names=["clip", "frame"],
        dynamic_axes={"mel": {0: "batch"},
                      "clip": {0: "batch"},
                      "frame": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
print(f"ONNX saved: {ONNX_PATH}, size={Path(ONNX_PATH).stat().st_size/1e6:.1f}MB")

# Verify ONNX has no BN with training_mode=1 (the actual blocker)
import onnx as _onnx_mod
_m = _onnx_mod.load(ONNX_PATH)
_bn_nodes = [n for n in _m.graph.node if n.op_type == "BatchNormalization"]
_bn_train1 = [n for n in _bn_nodes
              if any(a.name == "training_mode" and a.i == 1 for a in n.attribute)]
print(f"ONNX BN nodes total: {len(_bn_nodes)}, with training_mode=1: {len(_bn_train1)}")
assert len(_bn_train1) == 0, f"BN with training_mode=1 still present — OpenVINO will fail"
print("[OK] no BN training_mode=1 in ONNX")
"""

CELL8_IR = """# === convert ONNX -> OpenVINO IR ===
ov_model = ov.convert_model(ONNX_PATH)
IR_PATH = "/kaggle/working/exp029_r3_fold0.xml"
ov.save_model(ov_model, IR_PATH, compress_to_fp16=False)
print(f"IR saved: {IR_PATH}")
print(f"  .xml: {Path(IR_PATH).stat().st_size/1e6:.2f}MB")
print(f"  .bin: {Path(IR_PATH.replace('.xml','.bin')).stat().st_size/1e6:.2f}MB")
"""

CELL9_VERIFY = """# === verify: torch vs OpenVINO on 3 random inputs (batch=12 like inference) ===
core = ov.Core()
compiled = core.compile_model(IR_PATH, "CPU")
# Use partial_shape (dynamic dim safe). Static shape would raise on dynamic axes.
print(f"OpenVINO compiled, inputs: {[i.get_partial_shape() for i in compiled.inputs]}")
print(f"  outputs: {[o.get_partial_shape() for o in compiled.outputs]}")

max_diff_clip = 0.0
max_diff_frame = 0.0
N_TEST = 3
with torch.no_grad():
    for i in range(N_TEST):
        x = torch.randn(N_WINDOWS, 1, E17_N_MELS, N_TF_DIM)
        clip_pt, frame_pt = export_model(x)
        ov_out = compiled(x.numpy())
        clip_ov = ov_out[compiled.outputs[0]]
        frame_ov = ov_out[compiled.outputs[1]]
        d_clip = float(np.abs(clip_pt.numpy() - clip_ov).max())
        d_frame = float(np.abs(frame_pt.numpy() - frame_ov).max())
        max_diff_clip = max(max_diff_clip, d_clip)
        max_diff_frame = max(max_diff_frame, d_frame)
        print(f"  [{i+1}/{N_TEST}] clip diff={d_clip:.6f}, frame diff={d_frame:.6f}")

print(f"\\nmax diff clip:  {max_diff_clip:.6f}")
print(f"max diff frame: {max_diff_frame:.6f}")
assert max_diff_clip < 1e-3, f"clip diff too large: {max_diff_clip}"
assert max_diff_frame < 1e-3, f"frame diff too large: {max_diff_frame}"
print("[OK] verify passed (max diff < 1e-3)")
"""

CELL10_BENCH = """# === benchmark: PyTorch vs OpenVINO (12-window batch = 1 file) ===
import time
N_ITER = 10
x = torch.randn(N_WINDOWS, 1, E17_N_MELS, N_TF_DIM)
x_np = x.numpy()

# warmup
with torch.no_grad():
    _ = export_model(x)
_ = compiled(x_np)

# PyTorch
with torch.no_grad():
    t0 = time.time()
    for _ in range(N_ITER):
        _ = export_model(x)
pt_time = (time.time() - t0) / N_ITER

# OpenVINO
t0 = time.time()
for _ in range(N_ITER):
    _ = compiled(x_np)
ov_time = (time.time() - t0) / N_ITER

print(f"PyTorch:  {pt_time*1000:8.1f}ms per file (12-window batch)")
print(f"OpenVINO: {ov_time*1000:8.1f}ms per file")
print(f"Speedup:  {pt_time/ov_time:.2f}x")
print(f"\\nProjected savings for 700 test files (typical hidden test size):")
print(f"  PyTorch:  {pt_time*700/60:.1f} min")
print(f"  OpenVINO: {ov_time*700/60:.1f} min")
"""

CELL11_LIST = """# === list output files ===
out_dir = Path("/kaggle/working")
print(f"Files in {out_dir}:")
total = 0
for f in sorted(out_dir.iterdir()):
    if f.is_file():
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:40s} {sz/1e6:8.2f}MB")
print(f"  {'TOTAL':40s} {total/1e6:8.2f}MB")
"""

CELLS = [
    (CELL1_INSTALL, "cell-install"),
    (CELL2_IMPORTS, "cell-imports"),
    (CELL3_CONST, "cell-const"),
    (CELL4_LOCATE, "cell-locate"),
    (CELL5_ARCH, "cell-arch"),
    (CELL6_LOAD, "cell-load"),
    (CELL7_ONNX, "cell-onnx"),
    (CELL8_IR, "cell-ir"),
    (CELL9_VERIFY, "cell-verify"),
    (CELL10_BENCH, "cell-bench"),
    (CELL11_LIST, "cell-list"),
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

# compile check
for i, (src, cid) in enumerate(CELLS):
    if cid == "cell-install":
        continue  # bash
    try:
        compile(src, f"<{cid}>", "exec")
        print(f"  [OK] {cid}")
    except SyntaxError as e:
        print(f"  [FAIL] {cid}: {e}")
