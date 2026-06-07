"""Build exp112 = exp111 base, swap exp102 → exp106 (exp029 R3 recipe).

exp112 = NB4 (0.30) + Tucker OV 5-fold (0.40) + exp106 OV 2-fold [1,2] (0.30)
  - exp106 uses exp029 R3 recipe (mixup 0.4, drop_path 0.1) — same as anchor exp029 R3 fold 0
  - 2-fold ensemble of fold 1, 2 (drop fold 0 = same as exp029 R3 fold 0 already)
  - Variance reduction without recipe shift = expected stack lift
  - Tucker 5-fold OV maintained
  - exp110 weight (0.30/0.40/0.30)

Patches from exp111:
  - EXP102_DIR → EXP106_DIR (new dataset: birdclef2026-e106-3fold-ov)
  - exp102_fold*.xml → exp106_fold*.xml
  - FOLDS_E102 → FOLDS_E106
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EXP111_NB = Path("experiment/exp111/notebook/nb_exp111_e102_fold12_w30_40_30.ipynb")
EXP112_NB = Path("experiment/exp112/notebook/nb_exp112_e106_fold12_w30_40_30.ipynb")
EXP112_NB.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(EXP111_NB.read_text(encoding="utf-8"))

# Patch e29-infer cell: replace exp102 references with exp106
patched = False
for c in nb["cells"]:
    if c.get("id") != "e29-infer":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    # Simple text replacements (order matters: variable names first, then filenames)
    # Variable name patterns
    src = src.replace("FOLDS_E102", "FOLDS_E106")
    src = src.replace("EXP102_DIR", "EXP106_DIR")
    src = src.replace("EXP102_IR", "EXP106_IR")
    src = src.replace("_ov_e102", "_ov_e106")
    src = src.replace("_core_e102", "_core_e106")
    # Dataset slug
    src = src.replace("birdclef2026-exp102-3fold-ov", "birdclef2026-e106-3fold-ov")
    # File name pattern (covers all f-string uses)
    src = src.replace("exp102_fold", "exp106_fold")
    # Print messages (cosmetic)
    src = src.replace("exp102 OV inference", "exp106 OV inference")
    src = src.replace("e102-ov", "e106-ov")
    src = src.replace("exp102 OV IR not found", "exp106 OV IR not found")
    src = src.replace("# === exp102 OV inference", "# === exp106 OV inference")

    # Verify critical replacements
    assert "exp106_fold" in src, "exp106_fold pattern missing"
    assert "EXP106_DIR" in src, "EXP106_DIR missing"
    assert "exp102_fold" not in src.replace("# ", ""), "exp102_fold still present"
    assert "EXP102_DIR" not in src.replace("# ", ""), "EXP102_DIR still present"

    c["source"] = src.splitlines(keepends=True)
    patched = True
    break

assert patched, "e29-infer cell not found"
print("[OK] e29-infer patched: exp102 → exp106")

EXP112_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {EXP112_NB} ({EXP112_NB.stat().st_size} bytes)")

# Final verify
nb_check = json.loads(EXP112_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    if c.get("id") == "e29-infer":
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "FOLDS_E106 = [1, 2]" in src
        assert "EXP106_DIR" in src
        assert "exp106_fold" in src
        assert "exp102" not in src.lower() or src.lower().count("exp102") == 0
        print("[ALL OK] exp112 NB ready")
        break

# Also verify other cells unchanged from exp111
import hashlib
def cell_hash(cell):
    s = cell["source"]
    s = "".join(s) if isinstance(s, list) else s
    return hashlib.md5(s.encode()).hexdigest()[:8]

nb111 = json.loads(EXP111_NB.read_text(encoding="utf-8"))
for c111, c112 in zip(nb111["cells"], nb_check["cells"]):
    if c111.get("id") not in ("e29-infer",):
        if cell_hash(c111) != cell_hash(c112):
            print(f"[WARN] cell {c111.get('id')} differs from exp111")
print("Verified: only e29-infer differs from exp111")
