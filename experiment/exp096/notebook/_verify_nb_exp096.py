"""Verify exp096 NB: structure + compile + key invariants."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_exp096_block_pp.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])}")
print()

checks = [
    ("f_temporal_block_completion def",  "def f_temporal_block_completion(" in all_src),
    ("block_pp called on submission",     "f_temporal_block_completion(submission" in all_src),
    ("BLOCK_THRESH = 0.30",               "BLOCK_THRESH    = 0.30" in all_src),
    ("MIN_BLOCK_LEN = 3",                 "MIN_BLOCK_LEN   = 3" in all_src),
    ("BLOCK_ALPHA = 0.50",                "BLOCK_ALPHA     = 0.50" in all_src),
    ("defensive try/except",              "[block_pp] error:" in all_src),
    ("row_id rsplit",                     'rsplit("_", 1)' in all_src),
    ("block_pp BEFORE to_csv",            all_src.find("f_temporal_block_completion(submission") < all_src.rfind('submission.to_csv("submission.csv"')),
    ("tax_smoothing preserved",           "f_tax_smoothing(submission" in all_src),
    ("lambda_prior=0.65 preserved",       "lambda_prior=0.65" in all_src),
    ("rank_aware power=0.6 preserved",    "power=0.6" in all_src),
]
print("=== Invariants ===")
for n, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {n}")

# Compile each code cell (with !/% lines commented)
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
