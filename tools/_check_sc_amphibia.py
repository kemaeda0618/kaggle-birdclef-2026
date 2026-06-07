"""Does labeled train_soundscapes contain Amphibia positives (for an FP-aware gate)?"""
import re
import pandas as pd
from pathlib import Path

BC = Path("C:/Users/maeke/work/kaggle/birdclef-2026")
tax = pd.read_csv(BC / "taxonomy.csv")
AMP = set(tax[tax["class_name"] == "Amphibia"]["primary_label"].astype(str))
l2c = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))

lab = pd.read_csv(BC / "train_soundscapes_labels.csv")
print("labels.csv columns:", list(lab.columns))
print("rows:", len(lab))

# find the label column (primary_label or similar)
lab_col = next((c for c in ["primary_label", "primary_labels", "label", "labels", "species"] if c in lab.columns), None)
print("label column:", lab_col)
print(lab.head(3).to_string())

# count segments per species (a "segment" = a row; label col may be ;/, separated)
from collections import Counter
seg_species = []  # set of labels per row
amp_seg = 0
for v in lab[lab_col].fillna(""):
    toks = [t.strip() for t in re.split(r"[;,]", str(v)) if t.strip() and t.strip() != "nan"]
    seg_species.append(set(toks))
    if any(t in AMP for t in toks):
        amp_seg += 1

cnt = Counter()
for s in seg_species:
    for t in s:
        cnt[t] += 1

amp_present = {k: v for k, v in cnt.items() if k in AMP}
print("\n=== Amphibia in labeled train_soundscapes ===")
print("total label rows (segments):", len(lab))
print("rows containing >=1 amphibian:", amp_seg)
print("rows with NO amphibian (negatives):", len(lab) - amp_seg)
print("distinct amphibian species present:", len(amp_present), "/ 35")
print("\nper-amphibian segment counts:")
for k, v in sorted(amp_present.items(), key=lambda x: -x[1]):
    print(f"  {k:10s} {l2c.get(k,'?'):10s} {v}")
print("\namphibians NOT in labeled SC:", sorted(AMP - set(amp_present)))
