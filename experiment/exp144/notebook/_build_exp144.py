"""Build exp144: exp116 with Tucker restored to FULL 5-fold (OV).

The clean, never-run test of "does e106 help vs exp110":
  exp110 = single e29 (PyTorch) + Tucker 5-fold (ONNX)            = 0.952
  exp116 = e106 2-fold (OV)    + Tucker 3-fold (OV, cut for time) = 0.950
  exp144 = e106 2-fold (OV)    + Tucker 5-fold (OV, FULL)         = ???

Tucker OV is ~1.38x faster than ONNX, so restoring 5-fold (+2 cheap folds, ~16s/fold local)
fits the budget that exp116 (which ran COMPLETE) already proved viable.
ONLY change from exp116: sed_xml_paths[:3] -> [:5]. e106 folds & weights 0.30/0.40/0.30 unchanged.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp116/notebook/nb_exp116_fold01.ipynb")
DST = Path("experiment/exp144/notebook/nb_exp144_tucker5_e106.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

OLD_LIMIT = "sed_xml_paths = sed_xml_paths[:3]   # ★ Tucker 3-fold (margin for ~82min)"
NEW_LIMIT = "sed_xml_paths = sed_xml_paths[:5]   # ★ exp144: Tucker 5-fold FULL (OV fast, frees time for e106)"
OLD_PRINT = 'print(f"SED OV folds USED (3-fold): {[p.name for p in sed_xml_paths]}")'
NEW_PRINT = 'print(f"SED OV folds USED (5-fold FULL): {[p.name for p in sed_xml_paths]}")'

sed_done = False
for c in nb["cells"]:
    if c.get("id") != "sed-infer":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert OLD_LIMIT in src, "exp116 Tucker 3-fold limit line not found"
    assert OLD_PRINT in src, "exp116 sed print line not found"
    src = src.replace(OLD_LIMIT, NEW_LIMIT).replace(OLD_PRINT, NEW_PRINT)
    c["source"] = src.splitlines(keepends=True)
    sed_done = True
    break
assert sed_done, "sed-infer cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

# verify
chk = json.loads(DST.read_text(encoding="utf-8"))
got = {}
for c in chk["cells"]:
    cid = c.get("id")
    if cid not in ("sed-infer", "e29-infer", "blend"):
        continue
    src = "".join(c["source"])
    got[cid] = src
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
    compile(clean, f"<{cid}>", "exec")

assert "sed_xml_paths[:5]" in got["sed-infer"], "Tucker 5-fold not applied"
assert "sed_xml_paths[:3]" not in got["sed-infer"], "stale 3-fold limit remains"
assert "FOLDS_E106 = [0, 1]" in got["e29-infer"], "e106 2-fold (folds 0,1) changed unexpectedly"
assert "BLEND_W_E10  = 0.30" in got["blend"] and "BLEND_W_SED  = 0.40" in got["blend"] and "BLEND_W_E17  = 0.30" in got["blend"], "weights not 0.30/0.40/0.30"
print("[OK] Tucker 5-fold FULL | e106 folds [0,1] | weights 0.30/0.40/0.30 | all cells compile")
