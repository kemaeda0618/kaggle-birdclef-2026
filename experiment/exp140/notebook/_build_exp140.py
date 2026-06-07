"""Build exp140: exp110 with blend weights NB4 0.275 / Tucker 0.40 / e29 0.325 (2.5%-step toward e29).

Brackets the e29-up direction: exp110=30/40/30, exp140=27.5/40/32.5, exp139=25/40/35.
Only NB4 0.30->0.275 and e29 0.30->0.325. Tucker stays 0.40. Sum=1.00. Pure base reweight.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp140/notebook/nb_exp140_blend_275_40_325.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "BLEND_W_E10  = 0.30" in src and "BLEND_W_E17  = 0.30" in src, "exp110 anchors not found"
    src = src.replace("BLEND_W_E10  = 0.30", "BLEND_W_E10  = 0.275")
    src = src.replace("BLEND_W_E17  = 0.30", "BLEND_W_E17  = 0.325")
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
        assert "BLEND_W_E10  = 0.275" in src and "BLEND_W_SED  = 0.40" in src and "BLEND_W_E17  = 0.325" in src
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        compile(clean, "<blend>", "exec")
        print("[OK] weights = NB4 0.275 / Tucker 0.40 / e29 0.325 (sum 1.00), compiles")
        break
