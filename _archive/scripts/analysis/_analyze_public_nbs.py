"""Analyze each pulled NB: extract markdown cells (description), count cells, identify key techniques."""
import json, io, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PULL_ROOT = Path("C:/Users/maeke/work/kaggle/birdclef-2026/_public_nb_pulls")

# Techniques to detect in code
TECH_PATTERNS = {
    "Tucker SED 5-fold ONNX": ["sed_fold", "bc2026-distilled-sed"],
    "Perch v2 ONNX": ["perch_v2", "perch_meta"],
    "LightProtoSSM": ["LightProtoSSM", "light_proto_ssm", "LightProto"],
    "ResidualSSM": ["ResidualSSM", "residual_ssm", "ResidualS"],
    "MLP probes": ["mlp_probe", "MLP probe", "probe_models"],
    "ProtoSSM v5": ["ProtoSSM v5", "ProtoSSMv5", "proto_ssm_v5"],
    "BirdNET": ["birdnet", "BirdNET", "birdnet_global"],
    "Sonotype mirror": ["MIRROR_PAIRS", "47158son", "sonotype"],
    "Rank blend": ["rank(axis=0", "rank(axis=0, pct=True)", "rank_aware"],
    "Power Transform": ["power_gamma", "POWER_GAMMA", "np.power(prob"],
    "Temperature scaling": ["temperature", "per_taxon_temp", "Temperature"],
    "Adaptive smoothing": ["adaptive_delta", "delta_smoothing"],
    "Isotonic Calibration": ["IsotonicRegression", "isotonic"],
    "Per-class Threshold": ["per_class_thresh", "threshold_opt"],
    "TTA (TimeShift)": ["circular_shift", "TTA", "tta_shifts"],
    "FCS (file conf scale)": ["FCS_TOP_K", "file_confidence", "_topk_mean"],
    "Hour prior": ["hour_prior", "HourPrior", "hour_utc"],
    "Site prior": ["site_prior", "S22"],
    "ConvNeXt": ["convnext", "ConvNeXt"],
    "NFNet": ["eca_nfnet", "nfnet"],
    "EfficientNet": ["efficientnet", "effv2", "efficientnetv2"],
    "Transformer": ["transformer", "Transformer", "Attention", "attention_head"],
    "Waveform cache": ["waveform_cache", "wave_cache", "waveform-cache"],
    "Sidecar": ["sidecar", "Sidecar", "SIDECAR"],
    "5s simple chunk": ["WINDOW_SAMPLES = SR * 5", "n_samples = SR * 5"],
    "10s sliding": ["DURATION_SEC = 10", "WINDOW_SEC = 10", "10s"],
    "20s sliding": ["DURATION_SEC = 20", "WINDOW_SEC = 20", "20s"],
    "Hierarchical Taxonomy": ["hier_tax", "HierTax", "hierarchical"],
    "Cross-Attention": ["cross_attention", "cross-attention", "CrossAttention"],
    "OOF eval": ["oof_n_splits", "n_oof", "OOF"],
}

for nb_path in sorted(PULL_ROOT.glob("*/*.ipynb")):
    name = nb_path.parent.name
    print(f"\n{'='*80}")
    print(f"NB: {name}")
    print(f"  File: {nb_path.name} ({nb_path.stat().st_size/1024:.1f}KB)")
    print('='*80)

    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  Parse ERR: {e}")
        continue

    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"  Cells: {n_code} code + {n_md} markdown")

    # Extract all source
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    all_md = "\n".join("".join(c.get("source", []))
                        for c in nb["cells"] if c["cell_type"] == "markdown")

    # First markdown cell (usually description)
    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    if md_cells:
        first_md = "".join(md_cells[0]["source"])
        first_md_lines = first_md.split("\n")[:20]
        print(f"\n  --- First markdown (intro/description) ---")
        for ln in first_md_lines:
            if ln.strip():
                print(f"    {ln[:120]}")
        if len(first_md.split("\n")) > 20:
            print(f"    ... (+{len(first_md.split(chr(10)))-20} more lines)")

    # Look for LB / weight / blend ratio mentions
    print(f"\n  --- LB / weight / blend ratio mentions ---")
    for pattern in ["LB", "blend", "weight", "ratio", "W_E10", "W_SED", "BLEND_W"]:
        for line in all_src.split("\n"):
            if pattern in line and "=" in line and not line.strip().startswith("#"):
                # likely an assignment
                if any(x in line for x in ["0.3", "0.4", "0.5", "0.2", "0.6", "BLEND", "rank"]):
                    print(f"    {line.strip()[:120]}")
                    break

    # Detect techniques
    print(f"\n  --- Techniques present ---")
    for tech, patterns in TECH_PATTERNS.items():
        present = any(p in all_src for p in patterns)
        if present:
            print(f"    [YES] {tech}")
    print(f"\n  --- Techniques ABSENT ---")
    for tech, patterns in TECH_PATTERNS.items():
        present = any(p in all_src for p in patterns)
        if not present:
            pass  # too long, skip

    # Check for specific blend weights
    blend_lines = [ln for ln in all_src.split("\n")
                    if "BLEND" in ln and "=" in ln and any(c.isdigit() for c in ln)]
    if blend_lines:
        print(f"\n  --- BLEND weight assignments ---")
        for ln in blend_lines[:10]:
            print(f"    {ln.strip()[:120]}")
