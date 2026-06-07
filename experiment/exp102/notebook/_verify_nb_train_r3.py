"""Verify exp102 NB: all patches + compile."""
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
    # Patch 1-5: Config
    ('FOLDS = [0, 1, 2]',                   "FOLDS = [0, 1, 2]" in all_src),
    ('N_TOTAL_EPOCHS = 30',                 "N_TOTAL_EPOCHS = 30" in all_src),
    ('WARMUP_EPOCHS = 6',                   "WARMUP_EPOCHS = 6" in all_src),
    ('BATCH = 640',                         "BATCH = 640" in all_src),
    ('NUM_WORKERS = 24',                    "NUM_WORKERS = 24" in all_src),
    # WRS warm-up
    ('WRS_WARMUP_EPOCHS = 5',               "WRS_WARMUP_EPOCHS = 5" in all_src),
    ('USE_EMA = True',                      "USE_EMA = True" in all_src),
    ('EMA_DECAY = 0.999',                   "EMA_DECAY = 0.999" in all_src),
    ('WRS warm-up alpha logic',             "_wrs_alpha = min(1.0" in all_src),
    ('_focal_effective passed to MixSamp',  "focal_weights=_focal_effective" in all_src),
    # EMA
    ('class ModelEMA def',                  "class ModelEMA:" in all_src),
    ('EMA init in train_r2',                "ema = ModelEMA(model, decay=EMA_DECAY)" in all_src),
    ('EMA update after opt step',           "ema.update(model)" in all_src),
    ('EMA val evaluation',                  "preds_ema = _predict_from_waveforms" in all_src),
    ('EMA best ckpt overwrite logic',       "[BEST EMA] saved EMA ckpt" in all_src),
    # Babych spec preserved
    ('MIXUP_ALPHA = 1.0 preserved',         "MIXUP_ALPHA = 1.0" in all_src),
    ('drop_path_rate=0.15 preserved',       "drop_path_rate=0.15" in all_src),
    ('SHARES focal 0.60 preserved',         '"focal": 0.60' in all_src),
    ('SHARES pseudo 0.30 preserved',        '"pseudo_sc": 0.30' in all_src),
    # Backbone
    ('BACKBONE eca_nfnet_l1 preserved',     '"eca_nfnet_l1"' in all_src),
    # Slug
    ('Slug: exp102-l1-3fold-r3p',           "birdclef2026-exp102-l1-3fold-r3p" in all_src),
    ('OLD exp100 references GONE',          "exp100" not in all_src),
]
print("=== Invariants ===")
for n, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {n}")

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
