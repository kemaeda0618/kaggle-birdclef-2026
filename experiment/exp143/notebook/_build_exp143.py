"""Build exp143: exp110 with EQUAL blend weights NB4 0.333 / Tucker 0.334 / e29 0.333.

The one untested weight point: equal split (Tucker pulled down from 0.40 to ~0.333).
exp110=30/40/30=0.952 (peak); this closes the weight axis ("does equal beat the tuned peak?").
Pure base reweight, all 3 weights changed. Sum = 1.000.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp143/notebook/nb_exp143_blend_333_334_333.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "BLEND_W_E10  = 0.30" in src, "exp110 NB4 anchor not found"
    assert "BLEND_W_SED  = 0.40" in src, "exp110 Tucker anchor not found"
    assert "BLEND_W_E17  = 0.30" in src, "exp110 e29 anchor not found"
    src = src.replace("BLEND_W_E10  = 0.30", "BLEND_W_E10  = 0.333")
    src = src.replace("BLEND_W_SED  = 0.40", "BLEND_W_SED  = 0.334")
    src = src.replace("BLEND_W_E17  = 0.30", "BLEND_W_E17  = 0.333")
    c["source"] = src.splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")
chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        assert "BLEND_W_E10  = 0.333" in src, "NB4 0.333 not applied"
        assert "BLEND_W_SED  = 0.334" in src, "Tucker 0.334 not applied"
        assert "BLEND_W_E17  = 0.333" in src, "e29 0.333 not applied"
        # no stale 0.30/0.40 weight lines remain
        assert "BLEND_W_E10  = 0.30\n" not in src and "BLEND_W_SED  = 0.40\n" not in src and "BLEND_W_E17  = 0.30\n" not in src, "stale weight remains"
        s = 0.333 + 0.334 + 0.333
        assert abs(s - 1.0) < 1e-9, f"weights sum {s} != 1.0"
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        compile(clean, "<blend>", "exec")
        print("[OK] weights = NB4 0.333 / Tucker 0.334 / e29 0.333 (sum 1.000), compiles")
        break
