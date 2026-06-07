"""Patch M1 NB → M2 NB.

M2 = M1 と同 paradigm、backbone のみ nfnet_l0 + 3-fold (軽量化版)。

Patches:
  - backbone: eca_nfnet_l1 → eca_nfnet_l0
  - N_FOLDS: 5 → 3
  - FOLDS: [0..4] → [0..2]
  - exp070 → exp071
  - m1 → m2
  - title: M1 → M2
  - BATCH=32 維持 (M1 batch 32 ミスを再現しない、L4 24GB なので 96 でも可)
    → 念のため Note 追加だけで触らず
"""
import json
from pathlib import Path

SRC_NB = Path("experiment/exp070/notebook/nb_train_m1.ipynb")
DST_NB = Path(__file__).with_name("nb_train_m2.ipynb")

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

# Patches: text replace pairs (順序重要)
patches = [
    # Title (Cell 0)
    ("# exp070 (M1): eca_nfnet_l1 + 5s + Perch ON + 5-fold R3 (Colab Pro+ G4)",
     "# exp071 (M2): eca_nfnet_l0 + 5s + Perch ON + 3-fold R3 (Colab L4)"),
    ("**M-R3 Perch-axis core (5s + nfnet_l1 + ImageNet + Perch distill)**",
     "**M-R3 Perch-axis lite (5s + nfnet_l0 + ImageNet + Perch distill、M1 軽量版)**"),
    ("Backbone: `eca_nfnet_l1.ra2_in1k` (★ M1: L1 = capacity 増、Perch ON 専用)",
     "Backbone: `eca_nfnet_l0.ra2_in1k` (★ M2: L0 = M1 と異なる backbone 軸)"),
    ("5-fold StratifiedKFold + 20 epoch", "3-fold StratifiedKFold + 20 epoch"),
    ("Output**: m1_fold{0..4}_ckpt_best_ns22.pth",
     "Output**: m2_fold{0..2}_ckpt_best_ns22.pth"),

    # CFG (Cell 3)
    ('BACKBONE = "eca_nfnet_l1"            # ★ M1: nfnet_l1 + ImageNet',
     'BACKBONE = "eca_nfnet_l0"            # ★ M2: nfnet_l0 + ImageNet (M1 軽量版)'),
    ("N_FOLDS = 5                           # ★ M1: 5-fold (full)\nFOLDS = [0, 1, 2, 3, 4]",
     "N_FOLDS = 3                           # ★ M2: 3-fold\nFOLDS = [0, 1, 2]"),
    ("BATCH = 32                            # ★ M1: nfnet_l1 + 96GB G4 で batch 32",
     "BATCH = 96                            # ★ M2: nfnet_l0 + L4 24GB で batch 96 (軽量 backbone)"),

    # Drive path (Cell 1)
    ('DRIVE_EXP_DIR    = Path("/content/drive/MyDrive/kaggle/birdclef2026") / "output" / "exp070"',
     'DRIVE_EXP_DIR    = Path("/content/drive/MyDrive/kaggle/birdclef2026") / "output" / "exp071"'),

    # Output prefix (cells 8, 9, 10, 11)
    ("m1_fold", "m2_fold"),
    ('"m1_fold{fold_k}_', '"m2_fold{fold_k}_'),
    ('m1_summary.json', 'm2_summary.json'),

    # Kaggle Dataset slug (Cell 10)
    ('SLUG  = "birdclef2026-exp070-m1-nfnet-l1-perch"',
     'SLUG  = "birdclef2026-exp071-m2-nfnet-l0-perch"'),
    ('TITLE = "birdclef2026 exp070 m1 nfnet l1 perch"',
     'TITLE = "birdclef2026 exp071 m2 nfnet l0 perch"'),
    ("exp070 M1 nfnet_l1 Perch ON 5-fold",
     "exp071 M2 nfnet_l0 Perch ON 3-fold"),
    ('https://www.kaggle.com/datasets/maekeso/birdclef2026-exp070-m1-nfnet-l1-perch',
     'https://www.kaggle.com/datasets/maekeso/birdclef2026-exp071-m2-nfnet-l0-perch'),

    # Header prints (Cell 8, 9, 11)
    ("[Fold {fold_k}] M1 R3 training", "[Fold {fold_k}] M2 R3 training"),
    ("# exp070 M1 (nfnet_l1 + Perch ON)",
     "# exp071 M2 (nfnet_l0 + Perch ON)"),
    ("exp070 M1 (nfnet_l1 + Perch ON) 5-fold R3 summary",
     "exp071 M2 (nfnet_l0 + Perch ON) 3-fold R3 summary"),
    ("M1 model:", "M2 model:"),
    ("All M1 R3 folds complete", "All M2 R3 folds complete"),
]

total = 0
for cell in nb["cells"]:
    if cell.get("cell_type") not in ("code", "markdown"):
        continue
    src = "".join(cell["source"])
    new_src = src
    for old, new in patches:
        if old in new_src:
            new_src = new_src.replace(old, new)
            total += 1
    if new_src != src:
        cell["source"] = new_src.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

print(f"Total patches applied: {total}")
DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {DST_NB}")

# Inspect
import ast
nb2 = json.loads(DST_NB.read_text(encoding="utf-8"))
errs = 0
SKIP_CELLS = {1, 2, 10, 12}
for i, c in enumerate(nb2["cells"]):
    if c["cell_type"] != "code" or i in SKIP_CELLS:
        continue
    src = "".join(c["source"])
    src_clean = "\n".join(l for l in src.splitlines() if not l.strip().startswith("!"))
    try:
        ast.parse(src_clean)
    except SyntaxError as e:
        errs += 1
        print(f"Cell {i}: SyntaxError line {e.lineno}: {e.msg}")
print(f"Syntax errors: {errs}")
