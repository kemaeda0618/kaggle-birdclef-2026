"""Build exp112 V7 = V4 (FP32 batched, path-fixed) with e106 FOLDS [1,2] -> [1] single.

Guaranteed-fit FP32 submission. No INT8. ~71 min sub time.
  - NB4 (exp037) FP32
  - Tucker SED 5-fold OV (batched, path-fixed)
  - exp106 fold 1 single OV (batched, path-fixed)  ← halves e106 time vs 2-fold
  - weight 0.30/0.40/0.30
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V4_NB = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")
V7_NB = Path("experiment/exp112/notebook/nb_exp112_v7_fold1.ipynb")

nb = json.loads(V4_NB.read_text(encoding="utf-8"))

patched = False
for c in nb["cells"]:
    if c.get("id") != "e29-infer":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "FOLDS_E106 = [1, 2]" in src, "FOLDS_E106 = [1, 2] anchor not found"
    src = src.replace("FOLDS_E106 = [1, 2]",
                      "FOLDS_E106 = [1]   # ★ V7: single fold (fit guaranteed ~71 min)")
    c["source"] = src.splitlines(keepends=True)
    patched = True

assert patched, "e29-infer not patched"
V7_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {V7_NB} ({V7_NB.stat().st_size} bytes)")

# verify
nb_check = json.loads(V7_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    if c.get("id") == "e29-infer":
        s = "".join(c["source"])
        assert "FOLDS_E106 = [1]" in s
        assert "FOLDS_E106 = [1, 2]" not in s
        assert "exp106_fold" in s
        assert "start_async" not in s   # no async (FP32 sequential)
        print("  [OK] e29-infer: FOLDS_E106 = [1] single, FP32, batched, no async")
        break
