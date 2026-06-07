"""Build Kaggle NB: per-species training file count (train_audio clips + labeled SC segments).

For each of 234 species, count:
  - train_audio clips (primary_label match in train.csv)
  - train_audio clips (secondary_labels match too)
  - labeled SC segments (train_soundscapes_labels.csv, semicolon-split)
  - total trainable signal
  - class_name (taxon)
Sorted ascending -> identify species with LOW/ZERO training signal (the 155 weak + 28 ghost).

Output: species_train_count.csv + console summary by taxon and by signal tier.
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp125/notebook/nb_exp125_species_count.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp125: per-species trainable file count ===
import re
from pathlib import Path
import numpy as np, pandas as pd

def ff(cands, marker):
    for p in cands:
        p = Path(p)
        if p.exists() and (list(p.rglob(marker)) or (p / marker).exists()):
            return p
    return None

COMP = ff(["/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"], "taxonomy.csv")
assert COMP is not None, "competition data not found"
print("COMP:", COMP)

tax = pd.read_csv(COMP / "taxonomy.csv")
train = pd.read_csv(COMP / "train.csv")
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(COMP / "sample_submission.csv")

SPECIES = sample_sub.columns[1:].tolist()
label2taxon = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
print(f"target species: {len(SPECIES)}")
print(f"train.csv rows: {len(train)}, labels rows: {len(labels)}")
"""

C2 = """# === count train_audio clips (primary + secondary) ===
train["primary_label"] = train["primary_label"].astype(str)

# primary count
primary_count = train["primary_label"].value_counts().to_dict()

# secondary count (secondary_labels is a list-like string)
secondary_count = {}
if "secondary_labels" in train.columns:
    for sl in train["secondary_labels"].dropna():
        s = str(sl)
        # parse list-like: "['a','b']" or "a;b" or "a,b"
        toks = re.findall(r"[A-Za-z0-9]+", s)
        for t in toks:
            if t in SPECIES:
                secondary_count[t] = secondary_count.get(t, 0) + 1
print("primary labels with clips:", len(primary_count))
print("secondary mentions:", sum(secondary_count.values()))
"""

C3 = """# === count labeled SC segments (semicolon-split, unique segments) ===
def t2s(v):
    s = str(v).strip()
    if ":" in s:
        p = [float(x) for x in s.split(":")]
        return p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]
    return float(s)

end_col = "end" if "end" in labels.columns else ("end_time" if "end_time" in labels.columns else None)
# unique (filename, end) segment -> species set
seg_species = {}
for _, r in labels.iterrows():
    stem = Path(str(r["filename"])).stem
    es = int(round(t2s(r[end_col]))) if end_col else 0
    sp = [x.strip() for x in str(r["primary_label"]).replace(",",";").split(";")
          if x.strip() and x.strip() != "nan"]
    seg_species.setdefault((stem, es), set()).update(sp)

sc_count = {}
for species_set in seg_species.values():
    for sp in species_set:
        if sp in SPECIES:
            sc_count[sp] = sc_count.get(sp, 0) + 1
print(f"unique SC segments: {len(seg_species)}")
print(f"species appearing in SC: {len(sc_count)}")
"""

C4 = """# === build per-species table ===
rows = []
for sp in SPECIES:
    pc = primary_count.get(sp, 0)
    scc_ = secondary_count.get(sp, 0)
    scl = sc_count.get(sp, 0)
    rows.append({
        "species": sp,
        "taxon": label2taxon.get(sp, "?"),
        "clip_primary": pc,
        "clip_secondary": scc_,
        "sc_segments": scl,
        "total_signal": pc + scl,           # primary clips + SC segments (main trainable signal)
        "total_with_sec": pc + scc_ + scl,
    })
df = pd.DataFrame(rows).sort_values("total_signal")
df.to_csv("/kaggle/working/species_train_count.csv", index=False)
print(f"saved species_train_count.csv ({len(df)} species)")
"""

C5 = """# === summary ===
print("=== Signal tiers (total_signal = clip_primary + sc_segments) ===")
tiers = [(0,0,"GHOST (zero signal)"), (1,5,"very rare 1-5"), (6,20,"rare 6-20"),
         (21,50,"low 21-50"), (51,200,"mid 51-200"), (201,10**9,"common 200+")]
for lo, hi, name in tiers:
    sub = df[(df["total_signal"]>=lo) & (df["total_signal"]<=hi)]
    print(f"  {name:22s}: {len(sub):3d} species")

print("\\n=== by taxon: median/min total_signal ===")
print(df.groupby("taxon")["total_signal"].agg(["count","median","min","max"]).round(1))

print("\\n=== ZERO-signal species (ghost: no clip, no SC) ===")
ghost = df[df["total_signal"]==0]
print(f"  count: {len(ghost)}")
for _, r in ghost.iterrows():
    print(f"  {r['species']:14s} {r['taxon']:10s} clip_sec={int(r['clip_secondary'])}")

print("\\n=== WEAKEST 40 (lowest total_signal) ===")
print(f"{'species':>14} {'taxon':>10} {'clip_p':>7} {'clip_s':>7} {'sc_seg':>7} {'total':>6}")
for _, r in df.head(40).iterrows():
    print(f"{r['species']:>14} {r['taxon']:>10} {int(r['clip_primary']):>7} {int(r['clip_secondary']):>7} {int(r['sc_segments']):>7} {int(r['total_signal']):>6}")

print("\\n=== species with ZERO clip but SOME SC (learnable from SC only) ===")
sc_only = df[(df["clip_primary"]==0) & (df["sc_segments"]>0)]
print(f"  count: {len(sc_only)}")
for _, r in sc_only.iterrows():
    print(f"  {r['species']:14s} {r['taxon']:10s} sc_seg={int(r['sc_segments'])}")
"""

CELLS = [(C1, "load"), (C2, "clips"), (C3, "sc"), (C4, "table"), (C5, "summary")]
nb = {"cells": [cell(s, c) for s, c in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB}")
for s, c in CELLS:
    try:
        compile(s, f"<{c}>", "exec"); print(f"  [OK] {c}")
    except SyntaxError as e:
        print(f"  [FAIL] {c}: {e}")
