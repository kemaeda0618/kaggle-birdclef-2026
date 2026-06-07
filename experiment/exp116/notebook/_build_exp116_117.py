"""Build exp116 (fold0+1) + exp117 (fold0+2) FP32 from exp112 V4 base.

exp112 V4 = NB4 + Tucker 5-fold OV + exp106 fold[1,2] OV FP32 (batched, path-fixed), weight 0.30/0.40/0.30.
Changes for each:
  - FOLDS_E106: [1,2] -> [0,1] (exp116) / [0,2] (exp117)  ← restore anchor fold0
  - Tucker: 5-fold -> 3-fold (sed_xml_paths[:3]) for ~82 min margin
  - all FP32 (INT8 abandoned, -0.018 on real test)
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V4_NB = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")


def build(out_path, folds_e106, label):
    nb = json.loads(V4_NB.read_text(encoding="utf-8"))

    # Patch 1: e29-infer FOLDS_E106
    e29_done = False
    for c in nb["cells"]:
        if c.get("id") != "e29-infer":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "FOLDS_E106 = [1, 2]" in src, "FOLDS_E106 anchor not found"
        src = src.replace("FOLDS_E106 = [1, 2]",
                          f"FOLDS_E106 = {folds_e106}   # ★ {label}: anchor fold0 restored, FP32")
        c["source"] = src.splitlines(keepends=True)
        e29_done = True
    assert e29_done

    # Patch 2: sed-infer Tucker 5-fold -> 3-fold (margin)
    sed_done = False
    for c in nb["cells"]:
        if c.get("id") != "sed-infer":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        # V4 has 5-fold full. Add [:3] limit after sed_xml_paths sort.
        anchor = 'print(f"SED OV folds USED (exp111 = 5-fold full): {[p.name for p in sed_xml_paths]}")'
        assert anchor in src, "Tucker 5-fold anchor not found"
        replacement = ('sed_xml_paths = sed_xml_paths[:3]   # ★ Tucker 3-fold (margin for ~82min)\n'
                       'print(f"SED OV folds USED (3-fold): {[p.name for p in sed_xml_paths]}")')
        src = src.replace(anchor, replacement)
        c["source"] = src.splitlines(keepends=True)
        sed_done = True
    assert sed_done

    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    # verify
    nbc = json.loads(out_path.read_text(encoding="utf-8"))
    for c in nbc["cells"]:
        if c.get("id") == "e29-infer":
            s = "".join(c["source"])
            assert f"FOLDS_E106 = {folds_e106}" in s
            assert "exp106_fold" in s and "int8" not in s.lower()  # FP32
            print(f"  [OK] e29-infer: FOLDS_E106 = {folds_e106} FP32")
        if c.get("id") == "sed-infer":
            s = "".join(c["source"])
            assert "sed_xml_paths[:3]" in s
            print(f"  [OK] sed-infer: Tucker 3-fold")


build(Path("experiment/exp116/notebook/nb_exp116_fold01.ipynb"), [0, 1], "exp116")
print()
build(Path("experiment/exp117/notebook/nb_exp117_fold02.ipynb"), [0, 2], "exp117")
