"""Full per-species accounting for all 35 BC26 Amphibia: train_audio count + AnuraSet added."""
import pandas as pd
from pathlib import Path

D = Path(__file__).with_name("_data")
tax = pd.read_csv("taxonomy.csv")
tr = pd.read_csv("train.csv"); tr["primary_label"] = tr["primary_label"].astype(str)
m = pd.read_csv(D / "bc26_mapping.csv")
lab = pd.read_csv(D / "labels.csv")

amp = tax[tax["class_name"] == "Amphibia"].copy()
amp["primary_label"] = amp["primary_label"].astype(str)
tc = tr["primary_label"].value_counts().to_dict()
sci2id = dict(zip(tax["scientific_name"], tax["primary_label"].astype(str)))

# AnuraSet count per bc26 id (direct + synonym)
anu_count = {}
for _, r in m.iterrows():
    if pd.notna(r["bc26_primary_label"]) and r["match_type"] in ("direct", "synonym"):
        bid = sci2id.get(r["bc26_primary_label"])
        col = "SPECIES_" + r["anuraset_code"]
        if bid and col in lab.columns:
            anu_count[bid] = int((lab[col] > 0).sum())

rows = []
for _, r in amp.iterrows():
    bid = r["primary_label"]
    ta = tc.get(bid, 0)
    an = anu_count.get(bid, 0)
    rows.append({"bc26": bid, "sci": r["scientific_name"], "train_audio": ta,
                 "anuraset_added": an, "total_after": ta + an, "added": "YES" if an > 0 else "-"})
df = pd.DataFrame(rows).sort_values("train_audio")

print(f"{'bc26':9} {'scientific':30} {'train_audio':11} {'anuraset':9} {'total':6} {'added'}")
for _, r in df.iterrows():
    print(f"{r['bc26']:9} {r['sci']:30} {r['train_audio']:11} {r['anuraset_added']:9} {r['total_after']:6} {r['added']}")

print("\n=== summary ===")
print("BC26 Amphibia species:", len(df))
print("train_audio count  -> min/median/max:", int(df['train_audio'].min()), int(df['train_audio'].median()), int(df['train_audio'].max()))
print("species with <=20 train_audio (data-scarce):", int((df['train_audio'] <= 20).sum()))
print("species that GOT AnuraSet data:", int((df['anuraset_added'] > 0).sum()), "/ 35")
print("total AnuraSet recordings added:", int(df['anuraset_added'].sum()))
print("scarce (<=20) species that got AnuraSet data:",
      int(((df['train_audio'] <= 20) & (df['anuraset_added'] > 0)).sum()))
print("scarce species still WITHOUT extra data:",
      sorted(df[(df['train_audio'] <= 20) & (df['anuraset_added'] == 0)]['bc26'].tolist()))
