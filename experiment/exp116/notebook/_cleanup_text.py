"""Clean up stale comments/text in exp112-derived NBs (exp112, exp116, exp117).

Stale tokens from base reuse:
  - "exp111" / "exp102" / "e102" / "e102-ov"  -> wrong exp name
  - "exp029 R3" mel comment -> fine (exp106 IS exp029 R3 recipe) but clarify
  - "5-fold full" print after [:3] slice -> misleading
  - V3/V4 version refs

Targets:
  exp116: NB4 + Tucker 3-fold + exp106 fold[0,1]   (FP32)
  exp117: NB4 + Tucker 3-fold + exp106 fold[0,2]   (FP32)
  (exp112 already submitted; clean for record consistency)
"""
import io, json, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGETS = {
    "exp116": ("experiment/exp116/notebook/nb_exp116_fold01.ipynb", "[0, 1]"),
    "exp117": ("experiment/exp117/notebook/nb_exp117_fold02.ipynb", "[0, 2]"),
}


def clean_sed_cell(src, exp_name):
    # Fix Tucker SED stale prints/comments
    src = src.replace(
        "# ★ exp111: restore Tucker SED 5-fold full ensemble (exp102 fold 1+2 で時間節約済、Tucker 5-fold 維持可能)",
        f"# {exp_name}: Tucker SED — limit to 3-fold for ~82min budget margin")
    # the [:3] line print already replaced in build; normalize any leftover "5-fold full"
    src = src.replace("SED OV folds USED (exp111 = 5-fold full)",
                      "SED OV folds USED (3-fold)")
    src = src.replace("# === Stage B: Tucker 5-fold SED OpenVINO ensemble (★ V2: BATCH_FILES=3) ===",
                      f"# === Stage B: Tucker SED 3-fold OpenVINO ensemble ({exp_name}, batched) ===")
    src = src.replace("# === Stage B: Tucker 5-fold SED OpenVINO ensemble ===",
                      f"# === Stage B: Tucker SED 3-fold OpenVINO ensemble ({exp_name}) ===")
    return src


def clean_e29_cell(src, exp_name, folds):
    # Header: was "exp102 OV inference" / "exp106 OV inference (...)"
    src = re.sub(r"# === exp10[26] OV inference.*?===",
                 f"# === exp106 OV inference ({exp_name}: fold {folds}, FP32) — R3 stream (exp029 R3 recipe) ===",
                 src, count=1)
    # idempotent install comment
    src = src.replace("# openvino install (idempotent — sed-infer already imported, just verify)",
                      "# openvino install (idempotent — sed-infer already loaded it)")
    # locate comment
    src = src.replace("# ---- Locate exp102 OV IR files ----",
                      "# ---- Locate exp106 OV IR files (fold0=exp029 R3 anchor, fold1/2=new) ----")
    src = src.replace("# Locate exp106 OV IR files (★ V4: notebooks/ direct first)",
                      "# Locate exp106 OV IR files (notebooks/ mount first)")
    # mel comment (exp106 = exp029 R3 recipe, so 'same config as exp029 R3' is correct — keep but clarify)
    src = src.replace("# ---- Mel transform (same config as exp029 R3 training) ----",
                      "# ---- Mel transform (exp106 = exp029 R3 recipe: 256 mels, n_fft 2048, hop 512) ----")
    # standardize comment
    src = src.replace("# per-instance standardize (CRITICAL: matches exp029 R3 train/infer)",
                      "# per-instance standardize (matches exp106/exp029 R3 train/infer)")
    # FOLDS comment line (build already set value; normalize trailing comment)
    src = re.sub(r"FOLDS_E106 = \[[0-9, ]+\][^\n]*",
                 f"FOLDS_E106 = {folds}   # {exp_name}: fold0 anchor + extra fold (FP32, INT8 abandoned -0.018)",
                 src, count=1)
    # log message e102-ov / e106-ov  -> e106-ov
    src = src.replace("e102-ov", "e106-ov")
    src = src.replace('"  e106-ov-batched', '"  e106-ov-batched')  # noop safety
    # "replaces exp029 R3 PyTorch" lines
    src = src.replace("— replaces exp029 R3 PyTorch (3-fold ensemble)", "")
    src = src.replace("— replaces exp029 R3 PyTorch", "")
    return src


for exp_name, (path, folds) in TARGETS.items():
    p = Path(path)
    nb = json.loads(p.read_text(encoding="utf-8"))
    changed = []
    for c in nb["cells"]:
        cid = c.get("id")
        if cid not in ("sed-infer", "e29-infer"):
            continue
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        orig = s
        if cid == "sed-infer":
            s = clean_sed_cell(s, exp_name)
        elif cid == "e29-infer":
            s = clean_e29_cell(s, exp_name, folds)
        if s != orig:
            c["source"] = s.splitlines(keepends=True)
            changed.append(cid)
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{exp_name}: cleaned cells {changed}")

    # verify: no stale tokens remain in comments, and code intact
    nbc = json.loads(p.read_text(encoding="utf-8"))
    blob = "\n".join("".join(c.get("source", [])) for c in nbc["cells"])
    stale = []
    for tok in ["exp111", "exp102", "e102-ov", "5-fold full"]:
        if tok in blob:
            stale.append(tok)
    print(f"  remaining stale tokens: {stale if stale else 'NONE'}")
    # confirm folds value intact
    assert f"FOLDS_E106 = {folds}" in blob, f"FOLDS_E106 {folds} lost!"
    assert "sed_xml_paths[:3]" in blob, "Tucker 3-fold lost!"
    print(f"  [OK] FOLDS_E106={folds}, Tucker 3-fold intact")

    # compile check on the two cells
    for c in nbc["cells"]:
        if c.get("id") in ("sed-infer", "e29-infer"):
            src = "".join(c["source"])
            clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                               for ln in src.splitlines())
            try:
                compile(clean, f"<{c['id']}>", "exec")
            except SyntaxError as e:
                print(f"  [FAIL compile] {c['id']}: {e}")
