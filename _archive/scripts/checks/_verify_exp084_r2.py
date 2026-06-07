"""Thorough audit of exp084 R2 Colab NB.

Checks:
1. Mel: Hi-time spec (N_FFT=2048, HOP=256, N_MELS=256, FMIN=20)
2. Backbone: convnext_pico.d1_in1k
3. Duration: 5s
4. Drive paths exp084 (not exp083)
5. R1 ckpt + R1 pseudo path
6. BATCH=64, LR=3.0e-4
7. Kaggle Dataset slug birdclef2026-exp084-weights
8. Perch distill 1× 5s (no averaging)
9. LabeledSC no aggregation
10. R1 pseudo assert
11. Syntax OK
"""
import json
from pathlib import Path
import ast

P = Path("experiment/exp084/notebook/nb_train_r2_colab.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

checks = []
def add(name, ok, detail=""):
    checks.append((name, ok, detail))

# 1. Mel: Hi-time
add("Mel N_FFT=2048 (Hi-time)", "N_FFT          = 2048" in all_src)
add("Mel HOP=256 (Hi-time、time res 2x)", "HOP_LENGTH     = 256" in all_src)
add("Mel N_MELS=256", "N_MELS         = 256" in all_src)
add("Mel FMIN=20", "FMIN           = 20" in all_src)
add("Mel FMAX=16000", "FMAX           = 16000" in all_src)
add("WIN_LENGTH = N_FFT (= 2048)", "WIN_LENGTH     = N_FFT" in all_src)

# 2. Backbone
add("Backbone convnext_pico.d1_in1k", 'BACKBONE = "convnext_pico.d1_in1k"' in all_src)

# 3. Duration
add("TRAIN_DURATION = 5", "TRAIN_DURATION = 5" in all_src)
add("VAL_DURATION = 5", "VAL_DURATION   = 5" in all_src)

# 4. Drive paths exp084
add("DRIVE_R1_PSEUDO = exp084/r1-pseudo/", 'DRIVE_R1_PSEUDO  = DRIVE_INPUT_DIR / "output" / "exp084" / "r1-pseudo" / "pseudo_labels.csv"' in all_src)
add("DRIVE_OUTPUT_DIR = exp084/r2", 'DRIVE_OUTPUT_DIR = DRIVE_INPUT_DIR / "output" / "exp084" / "r2"' in all_src)
add("DRIVE_R2_PSEUDO_DIR = exp084/r2-pseudo", 'DRIVE_R2_PSEUDO_DIR = DRIVE_INPUT_DIR / "output" / "exp084" / "r2-pseudo"' in all_src)
add("No exp083 in paths", 'output" / "exp083"' not in all_src and 'birdclef2026-exp083' not in all_src.replace('exp083 R2 (base)', '').replace('exp083 (Hi-freq)', '').replace('exp083 と同様', '').replace('exp083 R2 から', ''))

# 5. R1 pseudo assert
add("R1 pseudo assert exists (early fail if missing)", "assert DRIVE_R1_PSEUDO.exists()" in all_src)

# 6. Batch + LR
add("BATCH = 64", "BATCH = 64" in all_src)
add("LR = 3.0e-4", "LR = 3.0e-4" in all_src)
add("N_TOTAL_EPOCHS = 20", "N_TOTAL_EPOCHS = 20" in all_src)
add("WARMUP_EPOCHS = 3", "WARMUP_EPOCHS = 3" in all_src)

# 7. Kaggle Dataset slug
add("Upload slug birdclef2026-exp084-weights", 'birdclef2026-exp084-weights' in all_src)
add("No exp083 in upload slug", 'birdclef2026-exp083-weights' not in all_src)

# 8. Perch distill 1x 5s
add("Perch 1× 5s direct (no averaging)", "exp083 Perch distill: 5s native = direct embed" in all_src or
    "exp084 Perch distill: 5s native = direct embed" in all_src or
    ("perch_emb = perch_teacher.embed(wav_5s).to(device)" in all_src and
     "emb_a + emb_b + emb_c + emb_d" not in all_src))

# 9. LabeledSC no aggregation (5s native)
add("No 4× LabeledSC aggregation", "Non-overlapping 20s windows" not in all_src)
add("5s native LabeledSC comment", "5s native = no LabeledSC aggregation" in all_src)

# 10. SHARES (R2: 3 sources)
add("R2 SHARES (focal + labeled_sc + pseudo_sc)",
    "focal" in all_src and "labeled_sc" in all_src and "pseudo_sc" in all_src)

# 11. R2: FORCE_FRESH
add("FORCE_FRESH = True (R2 warm start from R1)", "FORCE_FRESH = True" in all_src)

# 12. v3 robust upload (preserve R1 + folder)
add("Upload uses r1/ subfolder", "td / 'r1'" in all_src or 'td / "r1"' in all_src)
add("Upload uses r2/ subfolder", "td / 'r2'" in all_src or 'td / "r2"' in all_src)
add("Re-auth before upload", "Re-authenticating Kaggle API for upload" in all_src)

# 13. M7 logging
add("M7 logging helpers (class_stats_str, taxon_str_m7)",
    "class_stats_str" in all_src and "taxon_str_m7" in all_src)

# 14. Syntax check
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
add(f"Syntax ({n_ok} OK, {n_err} ERR)", n_err == 0, "\n  ".join(err_details))

# Print results
print(f"\n=== exp084 R2 Colab NB verification ===\n")
all_ok = True
for name, ok, detail in checks:
    mark = "✅" if ok else "❌"
    print(f"{mark} {name}")
    if detail and not ok:
        print(f"   {detail}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("=" * 60)
    print("[OK] All checks passed — exp084 R2 NB is safe to run")
    print("=" * 60)
    print()
    print("=== Drive 必須 assets ===")
    print("  /content/drive/MyDrive/kaggle/birdclef2026/")
    print("    output/exp084/")
    print("      r1/")
    print("        ckpt_best_ns22.pth   (118MB convnext_pico)")
    print("        ckpt_best_macro.pth")
    print("        ckpt_latest.pth")
    print("        history.json")
    print("      r1-pseudo/             ★ assert fail if missing")
    print("        pseudo_labels.csv")
    print("    kaggle.json")
    print("    perch-onnx/perch_v2*.onnx")
else:
    print("=" * 60)
    print("[FAIL] Some checks failed — DO NOT run yet")
    print("=" * 60)
