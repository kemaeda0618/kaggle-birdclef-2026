"""Build exp081 R1 inference NB (R1 ckpt 専用).

Difference from nb_infer.ipynb:
- ckpt search 優先: R1 ckpt (ckpt_best_ns22.pth WITHOUT r2_ prefix)
- R2 ckpt は fallback として残す

Use case: R1 完走後すぐに sub したい (R2 待たない)
"""
import json
from pathlib import Path

OUT_PATH = Path(__file__).with_name("nb_infer_r1.ipynb")
SRC_PATH = Path(__file__).with_name("nb_infer.ipynb")

# Load base R2-aware NB
nb = json.loads(SRC_PATH.read_text(encoding="utf-8"))

# Modify cells
for c in nb["cells"]:
    src = "".join(c.get("source", []))

    # 1. Update header (cell 0)
    if "exp081 inference: 10s sliding window" in src and c["cell_type"] == "markdown":
        new_src = """# exp081 **R1** inference: 10s sliding window (Tucker mel)

**Setup**:
- Window: 10s
- Mel: Tucker spec (n_mels=256, n_fft=2048, hop=512)
- Sliding step: 5s → 12 windows/file = 12 row outputs
- Pad audio: 2.5s + 60s + 2.5s = 65s
- Smoothing: [0.1, 0.2, 0.4, 0.2, 0.1] across 12 windows
- Inference: PyTorch CPU (eca_nfnet_l0、Kaggle 90min limit)

**Input**:
- Kaggle Dataset: `maekeso/birdclef2026-exp081-weights` (R1 ckpt: `ckpt_best_ns22.pth`)
- Competition: `birdclef-2026/test_soundscapes/`

**Output**: `/kaggle/working/submission.csv`

**★ R1 専用**: ckpt search で `ckpt_best_ns22.pth` (no `r2_` prefix) を優先
"""
        c["source"] = new_src.splitlines(keepends=True)

    # 2. Modify ckpt search list — R1 ckpts FIRST
    if "PTH_PATH = None" in src and 'for cand_name in [' in src:
        # Replace search order
        new_src = src.replace(
            'for cand_name in ["ckpt_best_ns22.pth", "r2_ckpt_best_ns22.pth", "ckpt_best_macro.pth", "r2_ckpt_best_macro.pth", "ckpt_latest.pth"]:',
            'for cand_name in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth"]:  # ★ R1 専用: r2_ prefix 探さない'
        )
        if new_src != src:
            c["source"] = new_src.splitlines(keepends=True)

OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Built: {OUT_PATH}")
print(f"  cells: {len(nb['cells'])}")
