"""Verify all patches applied to Stage 1 NB."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

CHECKS = {
    "install": [
        ("Drive path fix", "/content/drive/MyDrive/kaggle/birdclef2026/exp052"),
        ("kaggle.json fallback", "KAGGLE_JSON_CANDIDATES"),
        ("KAGGLE_API_TOKEN env var", 'os.environ["KAGGLE_API_TOKEN"]'),
        ("API auth verify", "Kaggle API ready"),
    ],
    "dl_data": [
        ("SDK direct (not CLI subprocess)", "from kaggle.api.kaggle_api_extended import KaggleApi"),
        ("api.dataset_download_files", "api.dataset_download_files"),
        ("Fail-fast on error", "raise FileNotFoundError"),
    ],
    "config": [
        ("LR=5e-4", "LR = 5e-4"),
        ("NUM_WORKERS=4", "NUM_WORKERS = 4"),
        ("MIXUP_ALPHA=1.0", "MIXUP_ALPHA = 1.0"),
        ("EPOCHS=30", "EPOCHS = 30"),
        ("BATCH_SIZE=256", "BATCH_SIZE = 256"),
    ],
    "model_setup": [
        ("GradScaler removed", "# scaler = GradScaler"),
    ],
    "train_loop": [
        ("bf16 (not fp16)", 'autocast("cuda", dtype=torch.bfloat16)'),
        ("No scaler.scale", "loss.backward()"),
        ("rich log taxon", "taxon:"),
        ("NaN safety", "torch.isnan(loss) or torch.isinf(loss)"),
    ],
    "eval_fn": [
        ("rich_evaluate function", "def rich_evaluate"),
        ("per-taxon AUC", "Aves"),
    ],
    "upload_kaggle": [
        ("Kaggle Dataset upload", "exp052-stage1-effv2s"),
    ],
    "disconnect": [
        ("runtime.unassign()", "runtime.unassign()"),
    ],
}

NEGATIVE = {
    "install": ["assert KAGGLE_JSON_PATH.exists()"],  # 旧 strict assert
    "dl_data": ["!kaggle datasets download"],  # 旧 CLI subprocess
    "config": ["LR = 1e-3", "NUM_WORKERS = 8", "MIXUP_ALPHA = 0.5"],
    "train_loop": ['autocast("cuda", dtype=torch.float16)', "scaler.scale("],
}

total_pass = 0
total_check = 0
for cell_id, checks in CHECKS.items():
    cell_src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            cell_src = "".join(c["source"])
            break
    if not cell_src:
        print(f"\n[{cell_id}] CELL NOT FOUND")
        continue
    print(f"\n[{cell_id}]")
    for name, expected in checks:
        ok = expected in cell_src
        print(f"  {'✓' if ok else '✗ MISSING'}  {name}")
        total_check += 1
        if ok: total_pass += 1

print(f"\n=== Negative checks (should NOT exist) ===")
for cell_id, neg_list in NEGATIVE.items():
    cell_src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            cell_src = "".join(c["source"])
            break
    for bad in neg_list:
        present = bad in cell_src
        print(f"  {'✗ LEFTOVER' if present else '✓ removed'}: [{cell_id}] {bad[:60]}")

print(f"\n=== Summary ===")
print(f"  Passed: {total_pass}/{total_check}")
print(f"  Cell count: {len(nb['cells'])}")
