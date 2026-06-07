"""Build exp146 (33/33/33) + exp147 (50/0/50, Tucker dropped) on exp112 V4 base ([1,2]+Tucker5).

exp146 = equal weights on the [1,2] config (plateau corner; exp143 was equal on the OLD fold0 stack).
exp147 = NB4 0.5 + e106[1,2] 0.5, Tucker weight 0 (diagnostic: how much does Tucker carry?).
Only the 3 blend weight lines change. USE_SED_PRE_AVG=False so SED=0 is safe (no div-by-zero).
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")

def build(dst_path, w10, wsed, w17, label):
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    # sanity: [1,2] + Tucker 5-fold preserved
    for c in nb["cells"]:
        if c.get("id") == "e29-infer":
            assert "FOLDS_E106 = [1, 2]" in "".join(c["source"]), "e106 not [1,2]"
        if c.get("id") == "sed-infer":
            assert "sed_xml_paths[:3]" not in "".join(c["source"]), "Tucker limited"
    patched = False
    for c in nb["cells"]:
        if c.get("id") != "blend":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "BLEND_W_E10  = 0.30" in src and "BLEND_W_SED  = 0.40" in src and "BLEND_W_E17  = 0.30" in src
        src = src.replace("BLEND_W_E10  = 0.30", f"BLEND_W_E10  = {w10}")
        src = src.replace("BLEND_W_SED  = 0.40", f"BLEND_W_SED  = {wsed}")
        src = src.replace("BLEND_W_E17  = 0.30", f"BLEND_W_E17  = {w17}")
        c["source"] = src.splitlines(keepends=True)
        patched = True
        break
    assert patched
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    # verify
    chk = json.loads(dst_path.read_text(encoding="utf-8"))
    for c in chk["cells"]:
        if c.get("id") == "blend":
            s = "".join(c["source"])
            assert f"BLEND_W_E10  = {w10}" in s and f"BLEND_W_SED  = {wsed}" in s and f"BLEND_W_E17  = {w17}" in s
            assert abs(float(w10) + float(wsed) + float(w17) - 1.0) < 1e-9, "sum != 1.0"
            assert "USE_SED_PRE_AVG = False" in s, "SED-pre-avg must be False for SED=0 safety"
            clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
            compile(clean, "<blend>", "exec")
    print(f"[OK] {label}: weights {w10}/{wsed}/{w17} (sum 1.0) | [1,2]+Tucker5 | compiles -> {dst_path.name}")

build(Path("experiment/exp146/notebook/nb_exp146_fold12_w333.ipynb"), "0.333", "0.334", "0.333", "exp146 equal")
build(Path("experiment/exp147/notebook/nb_exp147_fold12_w50_0_50.ipynb"), "0.50", "0.0", "0.50", "exp147 noTucker")
