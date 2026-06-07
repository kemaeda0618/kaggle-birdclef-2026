"""Patch M3 NB → M4 NB (backbone + ckpt + slug 変更のみ).

M4 = tf_efficientnet_b3 + 20s Babych iter3+ init (M3 と同 paradigm、backbone のみ変更)
N_FOLDS: 3 → 1
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("nb_train_m4.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Patches: text replace pairs
patches = [
    # Title
    ("# exp072 (M3): eca_nfnet_l0", "# exp073 (M4): tf_efficientnet_b3"),
    # Subtitle
    ("**M-R3 mel軸 core (Babych pretrain transfer 検証 model)**", "**M-R3 EffNet family軸 (Babych b3 iter3+ ckpt transfer)**"),
    # Spec line in title
    ("Backbone: `eca_nfnet_l0.ra2_in1k`", "Backbone: `tf_efficientnet_b3.ns_jft_in1k`"),
    ("Init: ★ Babych BC25 iter3 eca_nfnet_l0 ckpt", "Init: ★ Babych BC25 iter3+ tf_efficientnet_b3 ckpt"),
    # CFG / 3-fold → 1-fold
    ("N_FOLDS = 3", "N_FOLDS = 1   # ★ M4: 1-fold (smaller commit、Babych transfer 確認用)"),
    # Babych ckpt pattern
    ('for f in BABYCH_DIR.glob("eca_nfnet_l0*.pt"):',
     'for f in BABYCH_DIR.glob("tf_efficientnet_b3*.pt"):'),
    # backbone in model
    ('backbone_name="eca_nfnet_l0.ra2_in1k"', 'backbone_name="tf_efficientnet_b3.ns_jft_in1k"'),
    # Output file prefix m3 → m4
    ("m3_fold", "m4_fold"),
    ("m3_summary.json", "m4_summary.json"),
    ('"m3_*"', '"m4_*"'),
    # Batch size adjustment (b3 vs nfnet_l0: b3 lighter, can fit 32)
    ("BATCH_SIZE = 24", "BATCH_SIZE = 32     # ★ M4: b3 lighter than nfnet_l0、batch 32 OK"),
    # Header print
    ('"[Fold {fold_k}] M3 training', '"[Fold {fold_k}] M4 training'),
    ('print(f"M3 model:', 'print(f"M4 model:'),
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
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")
