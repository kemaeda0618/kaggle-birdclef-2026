"""Verify exp046 modifications."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp046\notebook\nb_train_r3_l1_gamma12.ipynb", encoding="utf-8"))

CHECKS = [
    ("hdr", "exp046: exp029 R3 + Power Transform γ=1.2"),
    ("hdr", "Aves negative signal"),
    ("setup", 'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp046"'),
    ("config", "POWER_GAMMA = 1.2"),
    ("fold_loop", "if POWER_GAMMA != 1.0:"),
    ("fold_loop", "prob_mat_after = np.power(prob_mat_before, POWER_GAMMA)"),
    ("fold_loop", "all {len(pseudo_df_r1)} rows kept (no chunk drop)"),
    ("upload", 'SLUG  = "birdclef2026-exp046-l1-gamma12"'),
    ("upload", "exp046 l1 power-transform ablation"),
]

NEG_CHECKS = [
    ("setup", '"output" / "exp045"'),  # should be exp046 now
    ("config", "CHUNK_FILTER_MAX_PROB"),  # should be removed
    ("fold_loop", "CHUNK_FILTER_MAX_PROB"),  # should be removed
    ("upload", 'SLUG  = "birdclef2026-exp045-l1-filtered"'),
]

# Verify metrics from exp045 still present
METRIC_CHECKS = [
    ("eval", "rich_eval"),
    ("eval", "per_class_distribution"),
    ("train_r2", "rich_eval"),
    ("train_r2", "per_taxon"),
    ("train_r2", "per_class_dist"),
]

print("=== Positive checks (should EXIST) ===")
for cell_id, expected in CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    found = expected in src
    print(f"  {'✓' if found else '✗'} [{cell_id}] {expected[:80]}")

print("\n=== Negative checks (should NOT exist) ===")
for cell_id, bad in NEG_CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    has_bad = bad in src
    print(f"  {'✗ LEFTOVER' if has_bad else '✓ removed'}: [{cell_id}] {bad[:80]}")

print("\n=== Metrics checks (from exp045 patches, should persist) ===")
for cell_id, expected in METRIC_CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    found = expected in src
    print(f"  {'✓' if found else '✗'} [{cell_id}] {expected[:80]}")
