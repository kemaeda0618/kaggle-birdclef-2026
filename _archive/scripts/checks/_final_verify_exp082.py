"""Final verification of exp082 R2 train NB before execution.

Checks:
1. Mel spec = Babych (N_MELS=224, N_FFT=4096, HOP=1252, FMIN=0)
2. TRAIN_DURATION=20
3. Perch distill 20s 4x averaging exists
4. LabeledSC 4x aggregation
5. PseudoScDS reads 20s windows
6. R1 pseudo path check (DRIVE_R1_PSEUDO)
7. R1 ckpts source path for upload
8. Upload cell uses folder structure
9. No orphan exp017 hardcoded cell
10. No syntax errors
11. Required Drive assets noted
"""
import json
from pathlib import Path
import ast

P = Path("experiment/exp082/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

checks = []

def add(name, ok, detail=""):
    checks.append((name, ok, detail))

# Concat all source
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

# 1. Mel spec (Babych)
ok = all([
    "N_FFT          = 4096" in all_src,
    "HOP_LENGTH     = 1252" in all_src,
    "N_MELS         = 224" in all_src,
    "FMIN           = 0" in all_src,
])
add("Mel spec = Babych (224/4096/1252/0)", ok)

# 2. TRAIN_DURATION=20
add("TRAIN_DURATION = 20", "TRAIN_DURATION = 20" in all_src)

# 3. Perch 20s 4x averaging
add("Perch 20s 4x averaging (N==640000)",
    "elif N == 640000:" in all_src and "emb_d = perch_teacher.embed" in all_src)

# 4. LabeledSC 4x aggregation (corrected comment)
add("LabeledSC 4x aggregation (exp082 comment)",
    "[exp082] Aggregating labeled SS 5s -> 20s windows" in all_src)
add("LabeledSC aggregation var name fixed",
    "start_sec_20s = float(group_sorted" in all_src and "start_sec_10s = float" not in all_src)

# 5. PseudoScDS 20s
# Just check TRAIN_SAMPLES is used in PseudoScDS, and TRAIN_SAMPLES = SR * 20
add("PseudoScDS reads TRAIN_SAMPLES (20s)",
    "TRAIN_SAMPLES = SR * TRAIN_DURATION" in all_src and "n_samples_target=TRAIN_SAMPLES" in all_src)

# 6. R1 pseudo path
add("DRIVE_R1_PSEUDO = exp082/r1-pseudo/",
    'DRIVE_R1_PSEUDO  = DRIVE_INPUT_DIR / "output" / "exp082" / "r1-pseudo" / "pseudo_labels.csv"' in all_src)
add("R1 pseudo assert (early fail if missing)",
    "assert DRIVE_R1_PSEUDO.exists()" in all_src)

# 7. Upload uses Drive R1 + local R2 (folder structure)
add("Upload uses DRIVE_R1_DIR (sibling of DRIVE_OUTPUT_DIR)",
    'DRIVE_R1_DIR = DRIVE_OUTPUT_DIR.parent / "r1"' in all_src)
add("Upload uses r1/ subfolder",
    'r1_dst = td / "r1"' in all_src)
add("Upload uses r2/ subfolder",
    'r2_dst = td / "r2"' in all_src)

# 8. Re-auth at upload start
add("Re-auth before upload",
    'Re-authenticating Kaggle API for upload' in all_src)

# 9. No orphan exp017 cells
exp017_count = sum(1 for c in nb["cells"]
                   if c.get("cell_type") == "code"
                   and 'output" / "exp017"' in "".join(c.get("source", [])))
add(f"No exp017 hardcoded cells (found {exp017_count})", exp017_count == 0)

# 10. Title fixed
add("Cell 18 TITLE = exp082 (not exp017)",
    '"birdclef2026 exp082 weights"' in all_src and '"birdclef2026 exp017 weights"' not in all_src)

# 11. Syntax check
n_ok, n_err = 0, 0
err_details = []
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        err_details.append(f"cell {c.get('id','?')}: {e}")
add(f"Syntax check ({n_ok} OK, {n_err} ERR)", n_err == 0, "\n  ".join(err_details))

# 12. Required SHARES
add("R2 SHARES (focal + labeled_sc + pseudo_sc)",
    "focal" in all_src and "labeled_sc" in all_src and "pseudo_sc" in all_src)

# Print results
print(f"\n=== exp082 R2 final verification ===\n")
all_ok = True
for name, ok, detail in checks:
    mark = "✅" if ok else "❌"
    print(f"{mark} {name}")
    if detail:
        print(f"   {detail}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("=" * 60)
    print("[OK] All checks passed — exp082 R2 NB is safe to run")
    print("=" * 60)
else:
    print("=" * 60)
    print("[FAIL] Some checks failed — DO NOT run yet")
    print("=" * 60)

# Print required Drive assets
print(f"\n=== Required Drive assets (verify on Colab before run) ===")
print("  /content/drive/MyDrive/kaggle/birdclef2026/")
print("    output/exp082/")
print("      r1/                    # R1 ckpts (for upload preserve)")
print("        ckpt_best_ns22.pth")
print("        ckpt_best_macro.pth")
print("        ckpt_latest.pth")
print("        history.json")
print("      r1-pseudo/             # R1 pseudo (for R2 training)")
print("        pseudo_labels.csv    # ★ assert fails if missing")
print("    kaggle.json              # API auth")
print("    perch-onnx/")
print("      perch_v2_no_dft.onnx   # Perch v2 teacher")
