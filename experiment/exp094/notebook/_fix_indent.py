"""Fix indent bug in cell 20 (Kaggle push section).

Strategy: revert the in-try-block patch + add skip flag at cell start.
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# Revert the bad patch first
BAD = '''    try:
        # ★ exp094: Push skipped (user: Colab で training、Kaggle push 不要)
    raise RuntimeError("exp094: Kaggle push disabled (manual Drive download instead). Comment out this line to enable.")
    api.dataset_view(f"{USER}/{SLUG}")
'''
GOOD = '''    try:
        api.dataset_view(f"{USER}/{SLUG}")
'''

for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if BAD in src:
        new_src = src.replace(BAD, GOOD)
        c["source"] = new_src.splitlines(keepends=True)
        print("[OK] Reverted bad indent patch")
        break

# Now add a clean skip flag at top of cell 20 (upload cell)
# Find cell containing "dataset_view" call
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if "api.dataset_view" in src and "USER" in src and "SLUG" in src:
        # Add skip flag at top of cell
        skip_block = '''# ============================================================
# ★ exp094: Kaggle push DISABLED (user: Colab で training、push 不要)
# To enable: set SKIP_KAGGLE_PUSH = False below
# ============================================================
SKIP_KAGGLE_PUSH = True
if SKIP_KAGGLE_PUSH:
    print("[exp094] Kaggle push skipped. ckpt available at DRIVE_OUTPUT_DIR.")
else:
'''
        # Indent existing cell content by 4 spaces (to be inside the else block)
        existing_lines = src.splitlines(keepends=True)
        indented = "".join("    " + ln if ln.strip() else ln for ln in existing_lines)
        new_src = skip_block + indented
        c["source"] = new_src.splitlines(keepends=True)
        print(f"[OK] Wrapped cell {i} with SKIP_KAGGLE_PUSH flag")
        break

# Save
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB}")

# Verify
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verify ===")
checks = [
    ("SKIP_KAGGLE_PUSH = True", "SKIP_KAGGLE_PUSH = True" in all_src),
    ("Skip message present", "Kaggle push skipped" in all_src),
    ("Bad indent removed", "raise RuntimeError(\"exp094: Kaggle push disabled" not in all_src),
    # Sanity
    ("USE_ALI_KD intact", "USE_ALI_KD     = True" in all_src),
    ("BEST marker still present", '"BEST "' in all_src),
    ("taxon line still present", "_exp094_taxon_str" in all_src),
    ("class line still present", "_exp094_class_stats_str" in all_src),
]
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

# Compile
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
