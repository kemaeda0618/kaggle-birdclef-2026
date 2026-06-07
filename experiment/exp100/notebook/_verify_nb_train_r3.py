"""Verify exp100 nb_train_r3.ipynb: 4 patches in place + compile."""
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
    # Patch 1: MIXUP_ALPHA
    ('Patch1: MIXUP_ALPHA = 1.0',            "MIXUP_ALPHA = 1.0" in all_src),
    ('Patch1: OLD MIXUP_ALPHA = 0.4 GONE',   "MIXUP_ALPHA = 0.4" not in all_src),
    # Patch 2: drop_path
    ('Patch2: drop_path_rate=0.15',          "drop_path_rate=0.15" in all_src),
    ('Patch2: OLD drop_path_rate=0.1 GONE',  "drop_path_rate=0.1," not in all_src),
    # Patch 3: SHARES_R2
    ('Patch3: focal 0.60',                   '"focal": 0.60' in all_src),
    ('Patch3: pseudo_sc 0.30',               '"pseudo_sc": 0.30' in all_src),
    ('Patch3: OLD focal 0.65 GONE',          '"focal": 0.65' not in all_src),
    ('Patch3: OLD pseudo_sc 0.25 GONE',      '"pseudo_sc": 0.25' not in all_src),
    # Patch 4: MixSamp WRS
    ('Patch4: focal_weights kwarg',          "focal_weights=None" in all_src),
    ('Patch4: rng.choice w/ p=',             "rng.choice(size, size=n, replace=True, p=self.focal_weights)" in all_src),
    ('Patch4: focal/uniform if-else',        'if name == "focal" and self.focal_weights is not None:' in all_src),
    # Patch 5: train_r2 focal_weights computation
    ('Patch5: _cls_counts',                  "_cls_counts = fds_df" in all_src),
    ('Patch5: _focal_weights computed',      "_focal_weights = np.array(" in all_src),
    ('Patch5: passed to MixSamp',            "focal_weights=_focal_weights" in all_src),
    # Backbone / training spec preserved
    ('Backbone: eca_nfnet_l1 (preserved)',   '"eca_nfnet_l1"' in all_src),
    ('N_TOTAL_EPOCHS = 20 (preserved)',      "N_TOTAL_EPOCHS = 20" in all_src),
    ('WARMUP_EPOCHS = 4 (preserved)',        "WARMUP_EPOCHS = 4" in all_src),
    ('LR = 3e-4 (preserved)',                "LR = 3e-4" in all_src),
    ('USE_PERCH_DISTILL = True (preserved)', "USE_PERCH_DISTILL = True" in all_src),
    # Slug / output renamed
    ('Slug: exp100-l1-babych',               "birdclef2026-exp100-l1-babych" in all_src),
    ('OLD slug exp029 GONE',                 "exp029" not in all_src),
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
