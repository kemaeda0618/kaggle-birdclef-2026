"""Build exp151: per-taxon blend weights (diagnostic-aligned).

exp150 diag (labeled SC, noise-ceiling) weak hint: Aves -> e29 best, non-Aves -> Tucker best.
exp151 makes blend weights class-conditional:
  Aves columns    : NB4 0.25 / Tucker 0.35 / e29 0.40   (e29-heavy)
  non-Aves columns: NB4 0.25 / Tucker 0.50 / e29 0.25   (Tucker-heavy)
Low-confidence (diag was noise-ceiling, no clean CV) but downside-bounded (finals = exp110+exp146).
ONLY change: the scalar rank-blend in the blend cell -> per-class weighted (vectors sum to 1 per column).
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp151/notebook/nb_exp151_pertaxon.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

OLD = (
    "    blend_flat = (BLEND_W_E10 * rank_e10\n"
    "                  + BLEND_W_SED * rank_sed\n"
    "                  + BLEND_W_E17 * rank_e17)\n"
    '    print(f"  3-way rank blend: NB4={BLEND_W_E10} / Tucker={BLEND_W_SED} / e29={BLEND_W_E17}")'
)
NEW = (
    "    # === exp151 per-taxon weights (diag-aligned): Aves->e29-heavy, non-Aves->Tucker-heavy ===\n"
    "    _tx151 = pd.read_csv(BASE / 'taxonomy.csv')\n"
    "    _cls151 = dict(zip(_tx151['primary_label'].astype(str), _tx151['class_name'].astype(str)))\n"
    "    _W_AVES_151 = (0.25, 0.35, 0.40)   # NB4, Tucker, e29\n"
    "    _W_NON_151  = (0.25, 0.50, 0.25)\n"
    "    _w10_151 = np.zeros(N_CLASSES, np.float32)\n"
    "    _wsed_151 = np.zeros(N_CLASSES, np.float32)\n"
    "    _w17_151 = np.zeros(N_CLASSES, np.float32)\n"
    "    _n_aves_151 = 0\n"
    "    for _ci, _lbl in enumerate(PRIMARY_LABELS):\n"
    "        _is_aves_151 = _cls151.get(str(_lbl), '') == 'Aves'\n"
    "        _n_aves_151 += int(_is_aves_151)\n"
    "        _w_151 = _W_AVES_151 if _is_aves_151 else _W_NON_151\n"
    "        _w10_151[_ci], _wsed_151[_ci], _w17_151[_ci] = _w_151\n"
    "    assert np.allclose(_w10_151 + _wsed_151 + _w17_151, 1.0), 'per-class weights must sum to 1'\n"
    "    blend_flat = (_w10_151[None, :] * rank_e10\n"
    "                  + _wsed_151[None, :] * rank_sed\n"
    "                  + _w17_151[None, :] * rank_e17)\n"
    "    print(f'  [exp151] per-taxon blend: {_n_aves_151} Aves (e29-heavy) / {N_CLASSES - _n_aves_151} non-Aves (Tucker-heavy)')"
)

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert OLD in src, "scalar blend block not found (exact text mismatch)"
    assert src.count(OLD) == 1, "blend block not unique"
    c["source"] = src.replace(OLD, NEW).splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

# verify
chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") == "blend":
        s = "".join(c["source"])
        assert "_w10_151[None, :] * rank_e10" in s, "per-class blend not applied"
        assert "_W_AVES_151 = (0.25, 0.35, 0.40)" in s and "_W_NON_151  = (0.25, 0.50, 0.25)" in s
        assert "BLEND_W_E10 * rank_e10" not in s, "stale scalar blend remains"
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
        compile(clean, "<blend>", "exec")
        print("[OK] per-taxon blend inserted (Aves 0.25/0.35/0.40, non-Aves 0.25/0.50/0.25), compiles")
        break

# offline sanity: weight-vector logic produces sum-1 vectors
import numpy as np, pandas as pd
_tx = pd.read_csv("taxonomy.csv")
_cls = dict(zip(_tx["primary_label"].astype(str), _tx["class_name"].astype(str)))
labels = _tx["primary_label"].astype(str).tolist()
WA, WN = (0.25, 0.35, 0.40), (0.25, 0.50, 0.25)
w10 = np.array([(WA if _cls.get(l, "") == "Aves" else WN)[0] for l in labels])
wsed = np.array([(WA if _cls.get(l, "") == "Aves" else WN)[1] for l in labels])
w17 = np.array([(WA if _cls.get(l, "") == "Aves" else WN)[2] for l in labels])
n_aves = sum(1 for l in labels if _cls.get(l, "") == "Aves")
assert np.allclose(w10 + wsed + w17, 1.0)
print(f"[offline check] {len(labels)} species, {n_aves} Aves / {len(labels)-n_aves} non-Aves, all weights sum to 1.0")
