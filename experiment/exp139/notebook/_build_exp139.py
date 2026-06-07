"""Build exp139: exp110 with blend weights NB4 0.25 / Tucker 0.40 / e29 0.35 (lean on NFNet generalizer).

Only change vs exp110 (0.30/0.40/0.30): NB4 0.30->0.25, e29 0.30->0.35. Tucker stays 0.40.
Rationale: e29(eca_nfnet_l1) has best val->LB generalization (gap -0.004); NB4 is in-NB-CV contaminated.
Pure base reweight — NO amphibian, NO new model. Sum = 1.0. Safe cushion attempt.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp139/notebook/nb_exp139_blend_25_40_35.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "BLEND_W_E10  = 0.30" in src and "BLEND_W_E17  = 0.30" in src, "exp110 weight anchors not found"
    src = src.replace("BLEND_W_E10  = 0.30", "BLEND_W_E10  = 0.25")   # NB4 0.30 -> 0.25
    src = src.replace("BLEND_W_E17  = 0.30", "BLEND_W_E17  = 0.35")   # e29  0.30 -> 0.35
    c["source"] = src.splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

# verify
chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        assert "BLEND_W_E10  = 0.25" in src
        assert "BLEND_W_SED  = 0.40" in src
        assert "BLEND_W_E17  = 0.35" in src
        # confirm sum line still present + compiles
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        compile(clean, "<blend>", "exec")
        print("[OK] weights = NB4 0.25 / Tucker 0.40 / e29 0.35 (sum 1.00), blend cell compiles")
        break
