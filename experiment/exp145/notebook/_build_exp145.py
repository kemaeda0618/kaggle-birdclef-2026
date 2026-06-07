"""Build exp145: exp112 V4 (best e106 config = [1,2] + Tucker 5-fold = 0.951) with e29 weight BOOSTED.

Rationale (from fold-pair data: [1,2]=0.951 > [0,2]=0.950 > [0,1]=0.949):
  e106 fold[1,2] is a STRONGER e29 stream than exp110's old single fold0.
  The 30/40/30 weight peak was found on the OLD weak e29 -> optimum likely shifted up for the better stream.
  exp145 boosts e29: NB4 0.30->0.275, Tucker 0.40 (unchanged), e29 0.30->0.325.
ONLY change from exp112 V4: blend weights. fold[1,2] + Tucker 5-fold preserved.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")
DST = Path("experiment/exp145/notebook/nb_exp145_fold12_e29boost.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

# sanity: confirm exp112 V4 has the expected best-config (do NOT modify these)
for c in nb["cells"]:
    if c.get("id") == "e29-infer":
        assert "FOLDS_E106 = [1, 2]" in "".join(c["source"]), "exp112 V4 e106 not [1,2]"
    if c.get("id") == "sed-infer":
        sed = "".join(c["source"])
        assert "sed_xml_paths[:3]" not in sed and "sed_xml_paths[:5]" not in sed, "Tucker fold limited (expected 5-fold full)"

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "BLEND_W_E10  = 0.30" in src and "BLEND_W_SED  = 0.40" in src and "BLEND_W_E17  = 0.30" in src, "exp112 V4 weight anchors not found"
    src = src.replace("BLEND_W_E10  = 0.30", "BLEND_W_E10  = 0.275")
    src = src.replace("BLEND_W_E17  = 0.30", "BLEND_W_E17  = 0.325")
    c["source"] = src.splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

chk = json.loads(DST.read_text(encoding="utf-8"))
got = {}
for c in chk["cells"]:
    cid = c.get("id")
    if cid in ("sed-infer", "e29-infer", "blend"):
        got[cid] = "".join(c["source"])
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in got[cid].splitlines())
        compile(clean, f"<{cid}>", "exec")
assert "FOLDS_E106 = [1, 2]" in got["e29-infer"], "e106 [1,2] lost"
assert "sed_xml_paths[:3]" not in got["sed-infer"] and "sed_xml_paths[:5]" not in got["sed-infer"], "Tucker 5-fold lost"
assert "BLEND_W_E10  = 0.275" in got["blend"] and "BLEND_W_SED  = 0.40" in got["blend"] and "BLEND_W_E17  = 0.325" in got["blend"], "weights not 0.275/0.40/0.325"
assert abs(0.275 + 0.40 + 0.325 - 1.0) < 1e-9
print("[OK] e106[1,2] + Tucker 5-fold preserved | weights 0.275/0.40/0.325 (e29 boosted, sum 1.0) | compiles")
