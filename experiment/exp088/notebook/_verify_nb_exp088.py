"""Verify exp088 NB syntax + key content."""
import json, ast, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp088_4model.ipynb"
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

# 2. Content checks
all_src = "\n\n".join("".join(c.get("source", [])) for c in nb["cells"])
checks = [
    ("openvino wheel install", "cooolz/openvino-package" in all_src),
    ("openvino import", "import openvino as ov" in all_src),
    ("exp082 OV path", "birdclef2026-exp082-weights" in all_src),
    ("r2_ov path", "r2_ov" in all_src),
    ("e082-infer cell exists", any(c.get("id") == "e082-infer" for c in nb["cells"])),
    ("e015-infer cell removed", not any(c.get("id") == "e015-infer" for c in nb["cells"])),
    ("Babych mel n_fft=4096", "n_fft=4096" in all_src),
    ("Babych mel hop_length=1252", "hop_length=1252" in all_src),
    ("Babych mel n_mels=224", "n_mels=224" in all_src),
    ("Babych mel fmin=0", "f_min=0" in all_src),
    ("_E082MelTF class", "_E082MelTF" in all_src),
    ("_prepare_input_sliding_20s function", "_prepare_input_sliding_20s" in all_src),
    ("20s sliding padded_sec 75", "75" in all_src and "DURATION_SEC = 20" in all_src),
    ("4-way blend weight sum check", "BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E082" in all_src),
    ("exp082 weight 0.15", "BLEND_W_E082 = 0.15" in all_src),
    ("exp037 weight 0.30", "BLEND_W_E10  = 0.30" in all_src),
    ("Tucker weight 0.35", "BLEND_W_SED  = 0.35" in all_src),
    ("exp029 weight 0.20", "BLEND_W_E17  = 0.20" in all_src),
    ("rank_e082 computed", "rank_e082 = pd.DataFrame(flat_e082)" in all_src),
    ("4-way print msg", "4-way rank blend" in all_src),
    ("Sonotype mirror retained", "MIRROR_PAIRS" in all_src),
    ("submission.to_csv", 'submission.to_csv("submission.csv"' in all_src),
    ("exp015 var names removed from blend", "BLEND_W_E015" not in all_src),
    ("probs_e082 not probs_e015 in blend", "probs_e082" in all_src and "probs_e015" not in all_src),
]
print("\nContent checks:")
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[MISS]'} {name}")

# 3. e082-infer cell internal
e082 = next((c for c in nb["cells"] if c.get("id") == "e082-infer"), None)
if e082:
    src = "".join(e082["source"])
    must = [
        "ov.Core()",
        "_e082_input_port",
        "create_infer_request().infer({_e082_input_port",
        "_prepare_input_sliding_20s(_raw_60s)",
        "(12, 1, 640000)",  # comment
        "gaussian_filter1d(_p_mean, sigma=0.65",
        "probs_e082.append",
        "Babych mel",
    ]
    print("\ne082-infer cell internal:")
    for m in must:
        print(f"  {'[OK]' if m in src else '[MISS]'} {m}")
else:
    print("\n[ERROR] e082-infer cell not found")

print("\n=== Done ===")
