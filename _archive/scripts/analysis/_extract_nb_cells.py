"""Extract specific cells from public NBs for deep analysis.

Focus: anthonytherrien 0.950 — get all code cell sources, identify:
1. POWEROPT (Model_5) implementation
2. lambda_prior / rank_aware_scaling
3. Hour/Site prior
4. f_TAX_SMOOTHING_POSTPROC
5. Cross-Attention LightProtoSSM
6. Final blend logic
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PULL_ROOT = Path("C:/Users/maeke/work/kaggle/birdclef-2026/_public_nb_pulls")

# Target 1: anthonytherrien 0.950
target_nb = PULL_ROOT / "anthonytherrien__birdclef-2026-ensemble-0-950" / "birdclef-2026-ensemble-0-950.ipynb"
nb = json.loads(target_nb.read_text(encoding="utf-8"))

# Dump all cells with id and snippet
print(f"=== {target_nb.parent.name} ===")
print(f"Total cells: {len(nb['cells'])}\n")

for i, c in enumerate(nb["cells"]):
    cell_type = c["cell_type"]
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    if cell_type == "code":
        # First 2 non-empty lines + first comment if any
        lines = [ln for ln in src.split("\n") if ln.strip()][:5]
        size = len(src)
        print(f"[{i:2d}] CODE id={cid:30s} ({size:6d} chars)")
        for ln in lines:
            print(f"     {ln[:100]}")
        print()
    else:
        # Markdown: first 2 non-empty lines
        lines = [ln for ln in src.split("\n") if ln.strip()][:3]
        size = len(src)
        print(f"[{i:2d}] MD   id={cid:30s} ({size:6d} chars)")
        for ln in lines:
            print(f"     {ln[:100]}")
        print()
