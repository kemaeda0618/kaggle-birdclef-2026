"""Match iNatSounds 2024 categories vs BirdCLEF 2026 species."""
import json
import pandas as pd

import os
TRAIN_JSON = os.environ.get("TEMP", "/tmp") + "/train.json"
with open(TRAIN_JSON, encoding="utf-8") as f:
    d = json.load(f)
cats = pd.DataFrame(d["categories"])
print("iNatSounds categories total:", len(cats))
print("Class breakdown:", cats["class"].value_counts().to_dict())

tax = pd.read_csv("C:/Users/maeke/work/kaggle/birdclef-2026/taxonomy.csv")
tax["sci_l"] = tax["scientific_name"].str.strip().str.lower()
cats["sci_l"] = cats["name"].str.strip().str.lower()
inat_sci = set(cats["sci_l"])
tax["in_inat"] = tax["sci_l"].isin(inat_sci)

print()
print("=== Coverage of 2026 by iNatSounds ===")
for cls in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
    sub = tax[tax["class_name"] == cls]
    n = sub["in_inat"].sum()
    print(f"  {cls}: {n}/{len(sub)} ({100*n/max(len(sub),1):.0f}%)")

print("\n=== Mammalia in iNatSounds ===")
print(tax[(tax["class_name"]=="Mammalia") & tax["in_inat"]][["scientific_name","common_name"]].to_string(index=False))
print("\n=== Reptilia in iNatSounds ===")
print(tax[(tax["class_name"]=="Reptilia") & tax["in_inat"]][["scientific_name","common_name"]].to_string(index=False))
print("\n=== Insecta in iNatSounds ===")
print(tax[(tax["class_name"]=="Insecta") & tax["in_inat"]][["scientific_name","common_name"]].to_string(index=False))
print("\n=== Amphibia in iNatSounds ===")
print(tax[(tax["class_name"]=="Amphibia") & tax["in_inat"]][["scientific_name","common_name"]].to_string(index=False))

# Per-species sample counts
ann = pd.DataFrame(d["annotations"])
counts = ann["category_id"].value_counts().to_dict()
cat_id_to_n = {cid: counts.get(cid, 0) for cid in cats["id"]}
cats["n_audio"] = cats["id"].map(cat_id_to_n)

print("\n=== Audio sample counts for matched non-bird species ===")
matched_non_bird = tax[(tax["class_name"]!="Aves") & tax["in_inat"]].merge(
    cats[["sci_l","n_audio"]], on="sci_l", how="left"
)
print(matched_non_bird[["scientific_name","common_name","class_name","n_audio"]].sort_values("n_audio", ascending=False).to_string(index=False))
