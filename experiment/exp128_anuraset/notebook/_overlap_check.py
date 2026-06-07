"""Cross-reference AnuraSet (denden12 bc26 mapping) with our 35 Amphibia + exp127 weak species."""
import pandas as pd
from pathlib import Path

D = Path(__file__).with_name("_data")
d = pd.read_csv(D / "labels.csv")
m = pd.read_csv(D / "bc26_mapping.csv")
tax = pd.read_csv("taxonomy.csv")
amp = set(tax[tax["class_name"] == "Amphibia"]["primary_label"].astype(str))
tr = pd.read_csv("train.csv")
tr["primary_label"] = tr["primary_label"].astype(str)
tc = tr["primary_label"].value_counts().to_dict()
sci2id = dict(zip(tax["scientific_name"], tax["primary_label"].astype(str)))

# weak species from exp127 OOF (failed generalization)
weak = {"64898", "67252", "555146", "22961", "25092", "22985", "326272"}

direct = m[m["bc26_primary_label"].notna() & m["match_type"].isin(["direct", "synonym"])]

rows = []
for _, r in direct.iterrows():
    col = "SPECIES_" + r["anuraset_code"]
    pos = int((d[col] > 0).sum()) if col in d.columns else -1
    strong = int((d[col] >= 2).sum()) if col in d.columns else -1   # stronger presence
    bcid = sci2id.get(r["bc26_primary_label"], "?")
    rows.append((r["anuraset_code"], bcid, r["anuraset_binomial"], pos, strong, tc.get(bcid, 0),
                 "WEAK" if bcid in weak else ""))

print(f"{'anura':7} {'bc26':9} {'sci':28} {'pos_rec':7} {'strong':6} {'train':6} {'flag':4}")
for a, bcid, binom, pos, strong, ta, w in sorted(rows, key=lambda x: -x[3]):
    print(f"{a:7} {bcid:9} {binom:28} {pos:7} {strong:6} {ta:6} {w:4}")

covered = {x[1] for x in rows}
print("\n=== summary ===")
print("AnuraSet direct-match species (total):", len(rows))
print("of which are in our Amphibia 35:", len(covered & amp))
print("our weak species COVERED by AnuraSet:", sorted(weak & covered))
print("our weak species NOT covered:", sorted(weak - covered))
print("our 35 Amphibia NOT covered by AnuraSet direct:", sorted(amp - covered))
