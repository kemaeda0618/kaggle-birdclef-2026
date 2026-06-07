"""Verify exp085 NB:
- All cells syntactically valid (Python ast.parse)
- Key contents present (OV install, exp015 OV path, 4-way blend, etc.)
"""
import json, ast, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp085_4model.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# 1. Syntax check
n_ok, n_err = 0, 0
errs = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        errs.append(f"  cell {i} (id={c.get('id','?')}): {e}")
print(f"Syntax: {n_ok} OK, {n_err} ERRORS")
for e in errs:
    print(e)

# 2. Verify key contents
checks = [
    ("openvino wheel install", "cooolz/openvino-package"),
    ("openvino import in Cell 0", "import openvino as ov"),
    ("exp015 OV path",  "birdclef2026-exp015-weights"),
    ("exp015 r2_ov path",  '"r2_ov"'),
    ("e015 cell exists", "exp015 R2 OV"),
    ("_identify_ov_outputs def", "_identify_ov_outputs"),
    ("e015 mel reuse _E17MelTF", "e015_mel_tf = _E17MelTF()"),
    ("4-way blend weight sum check", "BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E015"),
    ("exp015 weight 0.15",  "BLEND_W_E015 = 0.15"),
    ("exp037 weight 0.30",  "BLEND_W_E10  = 0.30"),
    ("Tucker weight 0.35",  "BLEND_W_SED  = 0.35"),
    ("exp029 weight 0.20",  "BLEND_W_E17  = 0.20"),
    ("rank_e015 computed",  "rank_e015 = pd.DataFrame(flat_e015)"),
    ("4-way print msg",  "4-way rank blend"),
    ("Sonotype mirror retained",  "MIRROR_PAIRS"),
    ("submission save", 'submission.to_csv("submission.csv"'),
]
print("\nContent checks:")
nb_str = json.dumps(nb)
for name, needle in checks:
    found = needle in nb_str
    print(f"  {'[OK]' if found else '[MISS]'} {name}")

# 3. Cell e015-infer specific check
e015_cell = next((c for c in nb["cells"] if c.get("id") == "e015-infer"), None)
if e015_cell:
    src = "".join(e015_cell["source"])
    must = [
        "compile_model",
        "_e015_input_port",
        "create_infer_request().infer({_e015_input_port",
        "gaussian_filter1d(_p_mean, sigma=0.65",
        "probs_e015.append",
    ]
    print("\ne015-infer cell internal:")
    for m in must:
        print(f"  {'[OK]' if m in src else '[MISS]'} {m}")
else:
    print("\n[ERROR] e015-infer cell not found")

print("\n=== Done ===")
