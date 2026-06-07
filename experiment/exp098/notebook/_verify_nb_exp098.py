"""Verify exp098 NB: ckpt swap + arch + compile."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_exp098_e095_swap.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])}")
print()

checks = [
    # ckpt swap
    ('NEW dataset: birdclef2026-exp095-l0-r2spec', 'birdclef2026-exp095-l0-r2spec' in all_src),
    ('OLD dataset path GONE',                      'birdclef2026-exp029-l1-single' not in all_src),
    ('NEW BACKBONE: eca_nfnet_l0',                 'E17_BACKBONE = "eca_nfnet_l0"' in all_src),
    ('OLD BACKBONE l1 NOT assigned',               'E17_BACKBONE = "eca_nfnet_l1"' not in all_src),
    ('ckpt name unchanged',                        'r3_fold0_ckpt_best_ns22.pth' in all_src),

    # PP / blend preserved (exp090 invariants)
    ('lambda_prior=0.65 preserved',                'lambda_prior=0.65' in all_src),
    ('rank_aware power=0.6 preserved',             'power=0.6' in all_src),
    ('Tax smoothing 0.15/0.05 preserved',          'genus_alpha=0.15, class_alpha=0.05' in all_src),
    ('BLEND_W_E10=0.35',                           'BLEND_W_E10  = 0.35' in all_src),
    ('BLEND_W_SED=0.40',                           'BLEND_W_SED  = 0.40' in all_src),
    ('BLEND_W_E17=0.25',                           'BLEND_W_E17  = 0.25' in all_src),
    ('Sonotype mirror preserved',                  'MIRROR_PAIRS' in all_src),
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
