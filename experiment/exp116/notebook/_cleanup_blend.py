"""Fix stale blend-cell comments in exp116/exp117."""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGETS = {
    "exp116": ("experiment/exp116/notebook/nb_exp116_fold01.ipynb", "fold0+1"),
    "exp117": ("experiment/exp117/notebook/nb_exp117_fold02.ipynb", "fold0+2"),
}

for exp_name, (path, folds_label) in TARGETS.items():
    p = Path(path)
    nb = json.loads(p.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") != "blend":
            continue
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        s = s.replace(
            "BLEND_W_E10  = 0.30   # ★ exp111: NB4 weight 0.35 → 0.30 (exp110 weight tune adoption)",
            "BLEND_W_E10  = 0.30   # NB4 (exp037) weight (exp110 tune)")
        s = s.replace(
            "BLEND_W_SED  = 0.40   # Tucker SED weight (維持)",
            "BLEND_W_SED  = 0.40   # Tucker SED 3-fold weight")
        s = s.replace(
            "BLEND_W_E17  = 0.30   # ★ exp111: exp102 stream weight 0.25 → 0.30 (exp110 weight tune adoption)",
            f"BLEND_W_E17  = 0.30   # exp106 {folds_label} stream weight (exp110 tune); var name E17 kept for code reuse")
        c["source"] = s.splitlines(keepends=True)
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    # verify no stale tokens anywhere
    blob = "\n".join("".join(c.get("source", [])) for c in json.loads(p.read_text(encoding="utf-8"))["cells"])
    stale = [t for t in ["exp111", "exp102", "e102-ov", "5-fold full"] if t in blob]
    # weights intact
    assert "BLEND_W_E10  = 0.30" in blob and "BLEND_W_SED  = 0.40" in blob and "BLEND_W_E17  = 0.30" in blob
    print(f"{exp_name}: stale tokens = {stale if stale else 'NONE'}, weights intact [OK]")
