"""Update R4 NB to use the NEW clean pseudo kernel slug.

OLD: maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029
NEW: maekeso/birdclef2026-exp029-pseudo-r3-train-sc

Both kernels produced the same pseudo_exp029.csv (410MB), but the new one is
properly named/organized under experiment/exp029/.
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_train_r4_l1_single.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

OLD_SLUG = "maekeso/birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029"
NEW_SLUG = "maekeso/birdclef2026-exp029-pseudo-r3-train-sc"

n_patched = 0
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if OLD_SLUG in src:
        new_src = src.replace(OLD_SLUG, NEW_SLUG)
        c["source"] = new_src.splitlines(keepends=True)
        n_patched += 1

print(f"Replaced {OLD_SLUG} → {NEW_SLUG} in {n_patched} cell(s)")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")

# Verify
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print(f"\n=== Verification ===")
print(f"  NEW slug present: {NEW_SLUG in all_src}")
print(f"  OLD slug removed: {OLD_SLUG not in all_src}")

# Compile check
print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
