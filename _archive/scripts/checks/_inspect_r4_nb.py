"""Inspect R4 train NB structure and verify against R3."""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

R4_NB = Path("experiment/exp029/notebook/nb_train_r4_l1_single.ipynb")
nb = json.loads(R4_NB.read_text(encoding="utf-8"))

print(f"=== R4 NB: {R4_NB.name} ===")
print(f"Cell count: {len(nb['cells'])}\n")

# Show all cell IDs + labels
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    label = ""
    for ln in src.split("\n")[:5]:
        s = ln.strip()
        if s and not s.startswith("# ====") and not s.startswith("```"):
            label = s[:90]; break
    print(f"  [{i:2d}] id={cid:18s} type={c['cell_type'][:4]:5s} | {label}")

print("\n=== R4 specific markers ===")
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
markers = {
    "exp029 R4 in header": "exp029 R4" in all_src,
    "Warm start logic present": "R4 WARM START" in all_src,
    "R3 ckpt path defined": "r3_fold0_ckpt_best_ns22.pth" in all_src,
    "Pseudo source = exp029": "pseudo_exp029.csv" in all_src,
    "Pseudo NOT exp017 (e17)": "pseudo_e17.csv" not in all_src,
    "N_TOTAL_EPOCHS = 12": "N_TOTAL_EPOCHS = 12" in all_src,
    "WARMUP_EPOCHS = 2": "WARMUP_EPOCHS = 2" in all_src,
    "LR = 1.5e-4": "LR = 1.5e-4" in all_src,
    "Drive output to /r4/": '"r4"' in all_src,
    "R4 Kaggle Dataset slug": "birdclef2026-exp029-r4-l1" in all_src,
    "BACKBONE eca_nfnet_l1": 'BACKBONE = "eca_nfnet_l1"' in all_src,
    "USE_PERCH_DISTILL True": "USE_PERCH_DISTILL = True" in all_src,
}
for k, v in markers.items():
    print(f"  {'[OK]' if v else '[FAIL]'} {k}")

print("\n=== Critical: R3 ckpt loading + state_dict apply ===")
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "R4 WARM START" in src:
        # Print 30 lines around warm start
        lines = src.split("\n")
        for i, ln in enumerate(lines):
            if "R4 WARM START" in ln:
                start = max(0, i)
                end = min(len(lines), i + 40)
                print(f"\n--- Warm start cell content (L{start}-L{end}) ---")
                for j in range(start, end):
                    print(f"  L{j:3d}: {lines[j][:140]}")
                break
        break

print("\n=== Critical: pseudo CSV path / loading ===")
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "pseudo_exp029" in src and "csv" in src:
        # Find lines with pseudo
        for ln in src.split("\n"):
            if "pseudo_exp029" in ln or "TEACHER_PSEUDO" in ln:
                if not ln.strip().startswith("#"):  # skip pure comments
                    print(f"  {ln[:140]}")
        break

# Compile-strict syntax check
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
        print(f"  [FAIL] cell {i} (id={c.get('id','?')}): {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
