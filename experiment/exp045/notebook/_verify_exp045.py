"""Verify exp045 modifications applied correctly."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb", encoding="utf-8"))

CHECKS = [
    ("hdr", "exp045: exp029 R3 + Chunk Filter Ablation"),
    ("hdr", "exp029 R3 fold 0 val_ns22 = 0.9177"),
    ("setup", 'DRIVE_EXP_DIR    = DRIVE_INPUT_DIR / "output" / "exp045"'),
    ("config", "CHUNK_FILTER_MAX_PROB = 0.5"),
    ("fold_loop", "if CHUNK_FILTER_MAX_PROB > 0:"),
    ("fold_loop", "keep_mask = max_prob_per_row > CHUNK_FILTER_MAX_PROB"),
    ("upload", 'SLUG  = "birdclef2026-exp045-l1-filtered"'),
    ("upload", 'TITLE = "birdclef2026 exp045 l1 filtered ablation"'),
    ("upload", "exp045 l1 chunk-filter ablation"),
]

# Verify NO leftover exp029 references where shouldn't be
NEG_CHECKS = [
    ("setup", '"output" / "exp029"'),  # should be exp045 now
    ("upload", 'SLUG  = "birdclef2026-exp029-l1-single"'),
]

for cell_id, expected in CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    found = expected in src
    print(f"  {'✓' if found else '✗'} [{cell_id}] {expected[:80]}")
    if not found:
        print(f"      MISSING — checking nearby lines:")
        for line in src.split("\n"):
            if any(w in line for w in expected.split()[:3]):
                print(f"        {line[:120]}")

print("\n--- Negative checks (should NOT exist) ---")
for cell_id, bad in NEG_CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    has_bad = bad in src
    print(f"  {'✗ LEFTOVER' if has_bad else '✓ removed'}: [{cell_id}] {bad[:80]}")
