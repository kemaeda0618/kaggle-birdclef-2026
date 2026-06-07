"""Verify exp102 v2 NB: all patches + compile + M7 logging + per-fold upload."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_r3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])}")
print()

checks = [
    # Config
    ('FOLDS = [0, 1, 2]',                       "FOLDS = [0, 1, 2]" in all_src),
    ('N_TOTAL_EPOCHS = 30',                     "N_TOTAL_EPOCHS = 30" in all_src),
    ('WARMUP_EPOCHS = 6',                       "WARMUP_EPOCHS = 6" in all_src),
    ('BATCH = 640',                             "BATCH = 640" in all_src),
    ('NUM_WORKERS = 24',                        "NUM_WORKERS = 24" in all_src),
    ('WRS_WARMUP_EPOCHS = 5',                   "WRS_WARMUP_EPOCHS = 5" in all_src),
    ('USE_EMA = True',                          "USE_EMA = True" in all_src),
    ('EMA_DECAY = 0.999',                       "EMA_DECAY = 0.999" in all_src),
    ('NS22_K = 22',                             "NS22_K = 22" in all_src),
    # WRS
    ('WRS warm-up alpha',                       "_wrs_alpha = min(1.0" in all_src),
    ('focal_effective passed',                  "focal_weights=_focal_effective" in all_src),
    # EMA
    ('class ModelEMA',                          "class ModelEMA:" in all_src),
    ('EMA init',                                "ema = ModelEMA(model, decay=EMA_DECAY)" in all_src),
    ('EMA update',                              "ema.update(model)" in all_src),
    ('EMA val eval',                            "val_preds_ema = _predict_from_waveforms" in all_src),
    # M7 logging helpers
    ('compute_per_species_auc def',             "def compute_per_species_auc" in all_src),
    ('class_stats_str def',                     "def class_stats_str" in all_src),
    ('taxon_str def',                           "def taxon_str" in all_src),
    # M7 logging output
    ('Insecta/Reptilia/Amphibia/Mammalia/Aves', "Insecta" in all_src and "Reptilia" in all_src and "Aves" in all_src),
    ('class stats format',                      "#perfect" in all_src),
    ('BEST tag',                                'best_tag = "BEST "' in all_src),
    ('tax_line printed',                        'print(f"    {tax_line}")' in all_src),
    ('class line printed',                      "print(f\"    class: {cls_line}\")" in all_src),
    # Val eval bug fix
    ('val_ns22 NOT inside if ema (bug fix)',    'val_ns22 = val_macro = float("nan")' in all_src),
    ('full_eval used (not compute_metrics)',    "full_eval(Y_val" in all_src),
    ('compute_metrics gone',                    "compute_metrics(" not in all_src),
    # Per-fold upload
    ('Per-fold upload block',                   "[upload] staged" in all_src),
    ('Per-fold dataset_create_version',         "_api.dataset_create_version" in all_src),
    # Babych spec preserved
    ('MIXUP_ALPHA = 1.0 preserved',             "MIXUP_ALPHA = 1.0" in all_src),
    ('drop_path_rate=0.15 preserved',           "drop_path_rate=0.15" in all_src),
    ('SHARES focal 0.60',                       '"focal": 0.60' in all_src),
    ('SHARES pseudo 0.30',                      '"pseudo_sc": 0.30' in all_src),
    # Backbone
    ('BACKBONE l1',                             '"eca_nfnet_l1"' in all_src),
    # Slug
    ('Slug exp102-l1-3fold-r3p',                "birdclef2026-exp102-l1-3fold-r3p" in all_src),
    ('exp100 GONE',                             "exp100" not in all_src),
]
print("=== Invariants ===")
n_fail = 0
for n, ok in checks:
    if not ok: n_fail += 1
    print(f"  {'[OK]' if ok else '[FAIL]'} {n}")
print(f"\nFailed: {n_fail}/{len(checks)}")

print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}-{c.get('id','?')}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} id={c.get('id','?')}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
