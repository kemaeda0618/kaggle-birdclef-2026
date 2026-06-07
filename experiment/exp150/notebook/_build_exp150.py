"""Build exp150: DIAGNOSTIC NB (no submission) — per-stream x per-taxon AUC on labeled train_soundscapes.

Goal: decide whether per-taxon blend has a basis. Clone exp110, redirect test_paths to the 66
labeled SC files, and append a cell computing per-stream (NB4/Tucker/e29) x per-taxon AUC vs ground truth.
If a stream is clearly best for the weak taxa (Amphibia/Insecta/Reptilia), per-taxon blend is worth a sub.

Caveat (noted in output): labeled SC is contaminated for NB4 (trains its MLP probes on it) -> NB4 AUC inflated.
So Tucker/e29 beating NB4 on a taxon despite that = strong signal; Tucker-vs-e29 comparison is the cleaner read.
Reuses exp110's exact CPU stream outputs (probs_exp010/probs_sed/probs_e17/probs_blend) -> faithful.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp150/notebook/nb_exp150_diag_perstream_taxon.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

# 1. redirect test_paths to labeled SC files
ANCHOR = 'test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))'
NEW = (
    '# ★ exp150 diag: use the 66 labeled train_soundscapes files (have ground truth)\n'
    '_lab_df0 = pd.read_csv(BASE / "train_soundscapes_labels.csv")\n'
    '_lab_stems0 = sorted(set(_lab_df0["filename"].apply(lambda x: Path(str(x)).stem)))\n'
    '_sc_map0 = {p.stem: p for p in (BASE / "train_soundscapes").glob("*.ogg")}\n'
    'test_paths = [_sc_map0[s] for s in _lab_stems0 if s in _sc_map0]\n'
    'print(f"[exp150 diag] labeled SC files: {len(test_paths)}")'
)
patched = False
for c in nb["cells"]:
    if c.get("id") != "v313_23":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert ANCHOR in src, "test path anchor not found in v313_23"
    c["source"] = src.replace(ANCHOR, NEW).splitlines(keepends=True)
    patched = True
    break
assert patched, "v313_23 not found"

# 2. append per-stream x per-taxon AUC diagnostic cell (after blend, where stream vars exist)
DIAG = r'''# === exp150 DIAGNOSTIC: per-stream x per-taxon AUC (labeled SC ground truth) ===
import re as _re, collections as _col
from sklearn.metrics import roc_auc_score as _auc

_lab = pd.read_csv(BASE / "train_soundscapes_labels.csv")
_tax = pd.read_csv(BASE / "taxonomy.csv")
_l2t = dict(zip(_tax["primary_label"].astype(str), _tax["class_name"].astype(str)))
_l2i = {l: i for i, l in enumerate(PRIMARY_LABELS)}

def _t2s(v):
    s = str(v).strip()
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        return int(round(p[0]*3600 + p[1]*60 + p[2])) if len(p) == 3 else int(round(p[0]*60 + p[1]))
    return int(round(float(v)))

_seg = {}
for _, r in _lab.iterrows():
    _stem = Path(str(r["filename"])).stem
    _es = _t2s(r["end"])
    _sp = [x.strip() for x in str(r["primary_label"]).replace(",", ";").split(";") if x.strip() and x.strip() != "nan"]
    _seg.setdefault((_stem, _es), set()).update(_sp)

_rids = meta_te["row_id"].astype(str).tolist()
_keep, _Yr = [], []
for _i, _rid in enumerate(_rids):
    _m = _re.match(r"^(.*)_(\d+)$", _rid)
    if not _m:
        continue
    _stem, _es = _m.group(1), int(_m.group(2))
    if (_stem, _es) not in _seg:
        continue
    _y = np.zeros(N_CLASSES, dtype=np.float32)
    for _s in _seg[(_stem, _es)]:
        if _s in _l2i:
            _y[_l2i[_s]] = 1.0
    _keep.append(_i); _Yr.append(_y)
_keep = np.array(_keep); _Y = np.stack(_Yr)
print(f"[diag] matched rows: {len(_keep)} / {len(_rids)}, positives: {int(_Y.sum())}")

_nb4 = probs_exp010 if "probs_exp010" in dir() else probs_exp037
_streams = {"NB4": _nb4, "Tucker": probs_sed, "e29": probs_e17, "blend": probs_blend}

def _per_class(P3):
    P = P3.reshape(-1, N_CLASSES)[_keep]
    out = {}
    for _ci, _lbl in enumerate(PRIMARY_LABELS):
        _pos = int(_Y[:, _ci].sum())
        if 0 < _pos < len(_Y):
            try:
                out[_lbl] = _auc(_Y[:, _ci], P[:, _ci])
            except Exception:
                pass
    return out

_res = _col.defaultdict(dict); _cnt = {}
for _name, _P3 in _streams.items():
    _pc = _per_class(_P3)
    _bt = _col.defaultdict(list)
    for _lbl, _a in _pc.items():
        _bt[_l2t.get(_lbl, "?")].append(_a)
    for _tx, _vals in _bt.items():
        _res[_tx][_name] = float(np.mean(_vals)); _cnt[_tx] = len(_vals)
    _res["ALL"][_name] = float(np.mean(list(_pc.values()))); _cnt["ALL"] = len(_pc)

print("\n[diag] CAVEAT: NB4 trains on labeled SC -> NB4 AUC inflated. Tucker/e29 beating NB4 = strong; Tucker-vs-e29 cleaner.")
print(f"\n{'taxon':>12} {'n':>4} {'NB4':>7} {'Tucker':>7} {'e29':>7} {'blend':>7}  best(non-NB4)")
for _tx in sorted(_res.keys()):
    _row = _res[_tx]
    _alt = {k: v for k, v in _row.items() if k in ("Tucker", "e29")}
    _best = max(_alt, key=_alt.get) if _alt else "?"
    print(f"{_tx:>12} {_cnt.get(_tx,0):>4} "
          f"{_row.get('NB4', float('nan')):>7.3f} {_row.get('Tucker', float('nan')):>7.3f} "
          f"{_row.get('e29', float('nan')):>7.3f} {_row.get('blend', float('nan')):>7.3f}  {_best}")
print("\n[diag] per-taxon blend has a basis IF a stream is clearly best on weak taxa (Amphibia/Insecta/Reptilia).")
'''
nb["cells"].append({"cell_type": "code", "id": "exp150-diag", "metadata": {},
                    "execution_count": None, "outputs": [], "source": DIAG.splitlines(keepends=True)})

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

# verify
chk = json.loads(DST.read_text(encoding="utf-8"))
ids = [c.get("id") for c in chk["cells"]]
assert ids[-1] == "exp150-diag", "diag cell not last"
for c in chk["cells"]:
    if c.get("id") in ("v313_23", "exp150-diag"):
        s = "".join(c["source"])
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
        compile(clean, f"<{c['id']}>", "exec")
        print(f"  [OK] {c['id']} compiles")
assert "train_soundscapes_labels.csv" in "".join(next(c["source"] for c in chk["cells"] if c.get("id")=="v313_23"))
print("[OK] exp150 diagnostic built")
