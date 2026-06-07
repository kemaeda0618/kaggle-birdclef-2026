"""Build exp113: exp106 standalone 3-fold INT8 submission NB.

Uses INT8 IR from OV5c output (exp106_fold{0,1,2}_int8.xml/.bin).
Standalone (no NB4/Tucker) — measures e106 3-fold INT8 stream LB on real test.
Compare to FP32 standalone 3-fold = 0.937.

kernel_sources:
  - maekeso/birdclef2026-ov5c-int8-e106 (OV5c output = INT8 IR)
  - ttahara/birdclef-2026-download-wheels (openvino offline)
enable_internet: False
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Clone from exp106 standalone OV (FP32) and swap IR paths to INT8
SRC_NB = Path("experiment/exp106/notebook/nb_exp106_standalone_3fold.ipynb")
NB_PATH = Path("experiment/exp113/notebook/nb_exp113_e106_int8_standalone.ipynb")
NB_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

# Patch locate cell: point to OV5c INT8 IR + filename exp106_fold{f}_int8.xml
patched = False
for c in nb["cells"]:
    if c.get("id") != "locate":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]

    NEW_LOCATE = '''# === Locate exp106 INT8 OV IR (3-fold) from OV5c output ===
EXP106_DIR = None
for p in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-ov5c-int8-e106"),
    Path("/kaggle/input/birdclef2026-ov5c-int8-e106"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-ov5c-int8-e106"),
]:
    if p.exists():
        EXP106_DIR = p; break
if EXP106_DIR is None:
    hits = glob.glob("/kaggle/input/**/exp106_fold0_int8.xml", recursive=True)
    if hits:
        EXP106_DIR = Path(hits[0]).parent
assert EXP106_DIR is not None, "exp106 INT8 IR not found (attach maekeso/birdclef2026-ov5c-int8-e106)"
print(f"EXP106_DIR (INT8): {EXP106_DIR}")

FOLDS = [0, 1, 2]
IR_PATHS = {}
for f in FOLDS:
    hits = list(EXP106_DIR.rglob(f"exp106_fold{f}_int8.xml"))
    assert hits, f"exp106_fold{f}_int8.xml not found"
    IR_PATHS[f] = hits[0]
    print(f"  fold {f} (INT8): {hits[0]}")
'''
    c["source"] = NEW_LOCATE.splitlines(keepends=True)
    patched = True
    break

assert patched, "locate cell not found"

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes)")

# verify
nb_check = json.loads(NB_PATH.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    if c.get("id") == "locate":
        s = "".join(c["source"])
        assert "exp106_fold{f}_int8.xml" in s
        assert "ov5c-int8-e106" in s
        print("  [OK] locate: INT8 IR from OV5c")
    if c.get("id") == "compile":
        s = "".join(c["source"])
        print(f"  compile cell present: {'compiled' in s}")
