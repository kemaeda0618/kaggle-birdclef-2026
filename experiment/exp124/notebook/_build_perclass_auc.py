"""Build per-class AUC analysis NB (no submission).

Runs exp090-style stack on labeled train_soundscapes (66 files, 739 segments),
compares to train_soundscapes_labels.csv ground truth, reports:
  - per-class AUC (sorted, find drag species)
  - per-taxon AUC (Aves/Insecta/Amphibia/Mammalia/Reptilia)
  - which species drag the macro average

Caveat: labeled SC is contaminated (NB4 trained on it) -> absolute AUC inflated,
but RELATIVE ranking (which species weak) is informative.

To keep runtime short, this clones exp090 (full stack) but runs ONLY on the 66
labeled SC files (not hidden test). Add a per-class AUC cell at the end.
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EXP090 = Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb")
OUT = Path("experiment/exp124/notebook/nb_exp124_perclass_auc.ipynb")
OUT.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(EXP090.read_text(encoding="utf-8"))

# Find the bridge cell — it builds audio_cache from test files. We override to use labeled SC.
# Strategy: insert a cell BEFORE bridge that redefines test_paths to labeled SC files,
# and add a final cell that computes per-class AUC from the blended `probs` vs ground truth.

# 1. Patch MODE/test discovery: force test_paths = labeled SC files
# The exp090 bridge cell does: test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
# We'll patch the bridge cell to use labeled SC instead, and remember filenames.

bridge_patched = False
for c in nb["cells"]:
    if c.get("id") != "bridge":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    # show original test discovery to patch
    anchor = 'test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))'
    if anchor in src:
        new = (
            '# ★ exp124: use labeled train_soundscapes (have ground truth) instead of hidden test\n'
            '_labels_df = pd.read_csv(BASE / "train_soundscapes_labels.csv")\n'
            '_labeled_stems = sorted(set(_labels_df["filename"].apply(lambda x: Path(str(x)).stem)))\n'
            '_sc_dir = BASE / "train_soundscapes"\n'
            '_sc_map = {p.stem: p for p in _sc_dir.glob("*.ogg")}\n'
            'test_paths = [_sc_map[s] for s in _labeled_stems if s in _sc_map]\n'
            'print(f"[exp124] per-class AUC mode: {len(test_paths)} labeled SC files")'
        )
        src = src.replace(anchor, new)
        c["source"] = src.splitlines(keepends=True)
        bridge_patched = True
        break
# fallback: search any cell with test_soundscapes glob
if not bridge_patched:
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if 'test_soundscapes' in src and 'glob' in src and '.ogg' in src:
            print(f"[info] test path glob found in cell {c.get('id')}, manual check needed")

# 2. Append per-class AUC analysis cell at the very end (after blend produces `probs`)
AUC_CELL = r'''# === exp124: per-class / per-taxon AUC analysis (labeled SC ground truth) ===
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# `probs` from blend cell: shape (n_rows, N_CLASSES) in sample_sub row order?
# exp090 builds `sub` DataFrame. We reconstruct per-window preds aligned to labeled SC.
# Build ground truth Y for the labeled SC windows we ran.

labels_df = pd.read_csv(BASE / "train_soundscapes_labels.csv")

def _t2s(v):
    s = str(v).strip()
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        return p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]
    return float(s)

end_col = "end" if "end" in labels_df.columns else ("end_time" if "end_time" in labels_df.columns else None)
seg = {}
for _, r in labels_df.iterrows():
    stem = Path(str(r["filename"])).stem
    es = int(round(_t2s(r[end_col])))
    sp = [x.strip() for x in str(r["primary_label"]).replace(",",";").split(";") if x.strip() and x.strip()!="nan"]
    seg.setdefault((stem, es), set()).update(sp)

# `sub` has row_id = {stem}_{end_sec}, columns = PRIMARY_LABELS
pred_rows = []
y_rows = []
matched = 0
for _, row in sub.iterrows():
    rid = row["row_id"]
    # row_id format {stem}_{endsec}
    m = re.match(r"^(.*)_(\d+)$", rid)
    if not m: continue
    stem, es = m.group(1), int(m.group(2))
    if (stem, es) not in seg:
        continue
    species = seg[(stem, es)]
    y = np.zeros(len(PRIMARY_LABELS), dtype=np.float32)
    for s in species:
        if s in label_to_idx:
            y[label_to_idx[s]] = 1.0
    pred_rows.append(row[PRIMARY_LABELS].values.astype(np.float32))
    y_rows.append(y)
    matched += 1

P = np.stack(pred_rows); Y = np.stack(y_rows)
print(f"[exp124] matched windows: {matched}, positives: {int(Y.sum())}")

# per-class AUC
taxonomy_df = pd.read_csv(BASE / "taxonomy.csv")
label2taxon = dict(zip(taxonomy_df["primary_label"].astype(str), taxonomy_df["class_name"].astype(str)))

rows = []
for ci, lbl in enumerate(PRIMARY_LABELS):
    pos = int(Y[:, ci].sum())
    if 0 < pos < len(Y):
        try:
            a = roc_auc_score(Y[:, ci], P[:, ci])
        except Exception:
            a = float("nan")
    else:
        a = float("nan")
    rows.append({"label": lbl, "taxon": label2taxon.get(lbl, "?"), "pos": pos, "auc": a})
df = pd.DataFrame(rows)
evald = df[df["auc"].notna()]
print(f"\n=== Overall ===")
print(f"evaluable species (pos>0): {len(evald)} / {len(PRIMARY_LABELS)}")
print(f"macro AUC (this labeled SC): {evald['auc'].mean():.4f}")

print(f"\n=== Per-taxon AUC ===")
print(evald.groupby("taxon")["auc"].agg(["count", "mean", "min"]).round(4))

print(f"\n=== Worst 25 species (drag) ===")
worst = evald.sort_values("auc").head(25)
for _, r in worst.iterrows():
    print(f"  {r['label']:14s} {r['taxon']:10s} pos={int(r['pos']):3d} auc={r['auc']:.4f}")

print(f"\n=== Best 10 (for reference) ===")
for _, r in evald.sort_values("auc", ascending=False).head(10).iterrows():
    print(f"  {r['label']:14s} {r['taxon']:10s} pos={int(r['pos']):3d} auc={r['auc']:.4f}")

df.to_csv("/kaggle/working/perclass_auc.csv", index=False)
print("\nsaved perclass_auc.csv")
'''

nb["cells"].append({
    "cell_type": "code", "id": "exp124-perclass-auc", "metadata": {},
    "execution_count": None, "outputs": [],
    "source": AUC_CELL.splitlines(keepends=True),
})

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
print(f"bridge_patched: {bridge_patched}")

# check label_to_idx variable name exists in exp090
blob = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for vn in ["label_to_idx", "LABEL2IDX", "label2idx"]:
    if vn in blob:
        print(f"  label index var found: {vn}")
