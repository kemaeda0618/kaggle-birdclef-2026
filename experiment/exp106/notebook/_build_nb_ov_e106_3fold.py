"""Build Kaggle NB: convert 3 ckpts (exp029 R3 fold 0 + exp106 fold 1, 2) -> OpenVINO IR.

All 3 ckpts share exp029 R3 architecture (eca_nfnet_l1 + Perch distill).
Unified naming for downstream use: exp106_fold{0,1,2}.xml + .bin

Output Dataset slug: birdclef2026-e106-3fold-ov
Used by exp112 inference NB (e102 fold 1+2 → e106 fold 1+2 swap).

Critical fixes (carried from OV1):
- ScaledStdConv2d monkey-patch (avoid BatchNormalization training_mode=1 in ONNX)
- training=TrainingMode.EVAL + dynamo=False
- partial_shape for dynamic batch
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path("experiment/exp106/notebook/nb_ov_e106_3fold.ipynb")


def cell(src, cid):
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELL1_INSTALL = """# openvino install (offline via ttahara wheels)
import os, glob
from pathlib import Path

print("=== /kaggle/input/ contents ===")
for p in sorted(Path("/kaggle/input").iterdir()):
    print(f"  {p.name}/")

WHEEL_DIR = None
hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
if hits:
    WHEEL_DIR = str(Path(hits[0]).parent)
    print(f"[wheels] found at: {WHEEL_DIR}  ({len(glob.glob(WHEEL_DIR + '/*.whl'))} wheels)")
else:
    print("[wheels] not found, fallback to internet pip")

try:
    import openvino as ov
    print(f"openvino preinstalled: {ov.__version__}")
except ImportError:
    if WHEEL_DIR:
        !pip install -q --no-deps {WHEEL_DIR}/openvino-*.whl {WHEEL_DIR}/openvino_telemetry-*.whl
    else:
        !pip install -q openvino==2026.0.0
    import openvino as ov
    print(f"openvino installed: {ov.__version__}")

try:
    import onnx; print(f"onnx: {onnx.__version__}")
except ImportError:
    if WHEEL_DIR:
        !pip install -q --no-deps {WHEEL_DIR}/onnx-*.whl {WHEEL_DIR}/onnx_ir-*.whl {WHEEL_DIR}/ml_dtypes-*.whl
    else:
        !pip install -q onnx
    import onnx
"""

CELL2_IMPORTS = """import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
import openvino as ov
import onnx

print(f"torch={torch.__version__}, openvino={ov.__version__}, onnx={onnx.__version__}")
"""

CELL3_CONST = """# === Constants (matches exp029 R3 / exp106 training) ===
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
print(f"N_TF_DIM={N_TF_DIM}")
"""

CELL4_LOCATE = """# === Locate 3 ckpts ===
CKPT_SPECS = [
    # (out_fold_id, label, search_paths, ckpt_filename_candidates)
    (0, "exp029_R3_fold0", [
        Path("/kaggle/input/birdclef2026-exp029-l1-single"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single"),
    ], ["r3_fold0_ckpt_best_ns22.pth"]),
    (1, "exp106_fold1", [
        Path("/kaggle/input/birdclef2026-exp106-fold1"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp106-fold1"),
    ], ["r3_fold1_ckpt_best_ns22.pth"]),
    (2, "exp106_fold2", [
        Path("/kaggle/input/birdclef2026-exp106-fold2"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-exp106-fold2"),
    ], ["r3_fold2_ckpt_best_ns22.pth"]),
]

CKPTS = {}
for out_id, label, search_paths, names in CKPT_SPECS:
    found = None
    for p in search_paths:
        if not p.exists():
            continue
        for name in names:
            hits = list(p.rglob(name))
            if hits:
                found = hits[0]; break
        if found: break
    assert found is not None, f"ckpt not found for {label}"
    CKPTS[out_id] = (label, found)
    print(f"  out_id={out_id}  {label}: {found} ({found.stat().st_size/1e6:.1f}MB)")
"""

CELL5_ARCH = """# === Architecture (exp029 R3 BirdSEDModel = _E17SED) ===
class _GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class _BirdSED(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, N_MELS, N_TF_DIM)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if USE_DISTILL:
            self.distill_head = _DistillHead(self.backbone_dim, PERCH_DIM)
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
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, mel):
        clip, frame = self.m(mel, return_framewise=True)
        return clip, frame
"""

CELL6_PATCH = """# === ScaledStdConv2d monkey-patch (NFNet Weight Standardization) ===
# Reason: F.batch_norm(training=True) -> ONNX BatchNormalization(training_mode=1) -> OpenVINO rejects.
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
print(f"Patched {len(_ssc_classes)} ScaledStdConv2d classes")
"""

CELL7_CONVERT = """# === Per-ckpt loop: load -> ONNX -> IR -> verify -> benchmark ===
OUT_DIR = Path("/kaggle/working")
IR_PATHS = {}
metrics = []
core = ov.Core()

for out_id in sorted(CKPTS.keys()):
    label, ckpt_path = CKPTS[out_id]
    print(f"\\n{'='*60}\\n  out_id={out_id}  {label}\\n{'='*60}")
    t_start = time.time()

    # Load ckpt
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    val = state.get('best_ns22', float('nan'))
    print(f"  ckpt epoch={state.get('epoch')}, best_ns22={val:.4f}")

    model = _BirdSED().cpu()
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    for sm in model.modules(): sm.eval()
    assert sum(1 for m in model.modules() if m.training) == 0
    export_model = _ExportWrap(model).eval()

    # Sanity forward
    with torch.no_grad():
        sanity = export_model(torch.randn(2, 1, N_MELS, N_TF_DIM))
        assert sanity[0].shape == (2, N_CLASSES)

    # ONNX export
    onnx_path = str(OUT_DIR / f"exp106_fold{out_id}.onnx")
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
    print(f"  ONNX: exp106_fold{out_id}.onnx ({onnx_mb:.1f}MB)")

    # Verify ONNX has no BN training_mode=1
    _m = onnx.load(onnx_path)
    _bn_t1 = [n for n in _m.graph.node if n.op_type == "BatchNormalization"
              and any(a.name == "training_mode" and a.i == 1 for a in n.attribute)]
    assert len(_bn_t1) == 0, f"{label}: {len(_bn_t1)} BN training_mode=1 in ONNX"

    # ONNX -> OpenVINO IR
    ov_model = ov.convert_model(onnx_path)
    ir_path = str(OUT_DIR / f"exp106_fold{out_id}.xml")
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    print(f"  IR: exp106_fold{out_id}.xml ({Path(ir_path).stat().st_size/1e6:.2f}MB) "
          f"+ .bin ({Path(ir_path.replace('.xml','.bin')).stat().st_size/1e6:.2f}MB)")
    IR_PATHS[out_id] = ir_path

    # Verify (3 random inputs)
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
    assert max_diff_clip < 1e-3 and max_diff_frame < 1e-3

    # Benchmark
    N_ITER = 5
    x_np = np.random.randn(N_WINDOWS, 1, N_MELS, N_TF_DIM).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    with torch.no_grad(): _ = export_model(x_t)
    _ = compiled(x_np)
    with torch.no_grad():
        t0 = time.time()
        for _ in range(N_ITER): _ = export_model(x_t)
        pt_t = (time.time() - t0) / N_ITER
    t0 = time.time()
    for _ in range(N_ITER): _ = compiled(x_np)
    ov_t = (time.time() - t0) / N_ITER
    print(f"  bench: PyTorch={pt_t*1000:.1f}ms, OV={ov_t*1000:.1f}ms, speedup={pt_t/ov_t:.2f}x")

    # Cleanup
    del compiled, ov_model, model, export_model, state
    import gc; gc.collect()

    metrics.append({"out_id": out_id, "label": label, "val_ns22": val,
                    "max_diff_clip": max_diff_clip, "max_diff_frame": max_diff_frame,
                    "pt_ms": pt_t * 1000, "ov_ms": ov_t * 1000,
                    "speedup": pt_t / ov_t, "elapsed_s": time.time() - t_start})

print(f"\\n{'='*60}\\nAll {len(IR_PATHS)} ckpts converted")
"""

CELL8_SUMMARY = """# === Summary table ===
print(f"\\n{'out_id':>7} {'label':>20} {'val':>8} {'pt_ms':>8} {'ov_ms':>8} {'speedup':>8} {'diff':>10}")
for r in metrics:
    print(f"{r['out_id']:>7d} {r['label']:>20} {r['val_ns22']:>8.4f} "
          f"{r['pt_ms']:>8.1f} {r['ov_ms']:>8.1f} {r['speedup']:>8.2f} "
          f"{r['max_diff_clip']:>10.6f}")
print(f"\\nTotal time: {sum(r['elapsed_s'] for r in metrics):.1f}s")
"""

CELL9_LIST = """# === List output files ===
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
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes)")

for src, cid in CELLS:
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<{cid}>", "exec")
        print(f"  [OK] {cid}")
    except SyntaxError as e:
        print(f"  [FAIL] {cid}: {e}")
