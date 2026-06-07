"""Build exp110: exp090 base with blend weight (NB4 0.30, Tucker 0.40, e29 0.30).

Change: e29 weight 0.25 → 0.30 (boost NFNet stream), NB4 0.35 → 0.30 (reduce in-NB CV contaminated stream).
Tucker maintained at 0.40 (strongest standalone, exp021 already proved up-weight = drag).
Expected stack LB: 0.950-0.953 (untried, e29 weight 0.30 untested).
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EXP090_NB = Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb")
EXP110_NB = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
EXP110_NB.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(EXP090_NB.read_text(encoding="utf-8"))

# Patch blend cell: change BLEND_W constants
patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    OLD = '''BLEND_W_E10  = 0.35   # NB4 v7 weight
BLEND_W_SED  = 0.40   # Tucker public SED weight
BLEND_W_E17  = 0.25   # exp029 R3 (eca_nfnet_l1) weight — note var name kept "E17" for code reuse'''
    NEW = '''BLEND_W_E10  = 0.30   # ★ exp110: NB4 weight 0.35 → 0.30 (in-NB CV contaminated stream を下げる)
BLEND_W_SED  = 0.40   # Tucker public SED weight (維持、exp021 で 0.45 → -0.001 確認済)
BLEND_W_E17  = 0.30   # ★ exp110: exp029 R3 weight 0.25 → 0.30 (NFNet test 域 generalization 強化)'''

    assert OLD in src, "blend weight anchor not found in exp090 blend cell"
    src = src.replace(OLD, NEW)
    c["source"] = src.splitlines(keepends=True)
    patched = True
    break

assert patched, "blend cell not found"

EXP110_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {EXP110_NB} ({EXP110_NB.stat().st_size} bytes)")

# verify
nb_check = json.loads(EXP110_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "BLEND_W_E10  = 0.30" in src
        assert "BLEND_W_SED  = 0.40" in src
        assert "BLEND_W_E17  = 0.30" in src
        print("[OK] verify: blend weights = 0.30 / 0.40 / 0.30")
        break
