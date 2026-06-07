"""train.csv の secondary_labels 列を調査"""
import pandas as pd
import ast
from collections import Counter

df = pd.read_csv("train.csv")
print(f"=== train.csv ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print()

# secondary_labels の format 確認
print(f"=== secondary_labels samples (first 10) ===")
for i, row in df.head(10).iterrows():
    print(f"  primary={row.primary_label:15s}  secondary={row.secondary_labels}")
print()

# empty rate
empty_mask = df["secondary_labels"].isin(["[]", "", None]) | df["secondary_labels"].isna()
print(f"=== secondary_labels stats ===")
print(f"Empty rate: {empty_mask.mean():.2%} ({empty_mask.sum()} / {len(df)})")
print(f"Has secondary: {(~empty_mask).mean():.2%} ({(~empty_mask).sum()} files)")
print()

# parse secondary_labels and count
def parse_secondary(s):
    if pd.isna(s) or s == "[]" or s == "":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

df["secondary_parsed"] = df["secondary_labels"].apply(parse_secondary)
df["n_secondary"] = df["secondary_parsed"].apply(len)

print(f"=== n_secondary distribution ===")
print(df["n_secondary"].value_counts().sort_index())
print(f"Mean: {df['n_secondary'].mean():.2f}, Max: {df['n_secondary'].max()}")
print()

# Most common secondary labels
all_secondary = [s for sub in df["secondary_parsed"] for s in sub]
print(f"=== Top 20 most common secondary labels ===")
for label, count in Counter(all_secondary).most_common(20):
    print(f"  {label:15s}: {count}")
print()
print(f"Total unique secondary species: {len(set(all_secondary))}")
print(f"Total secondary occurrences: {len(all_secondary)}")
print()

# Overlap with BC2026 234 species
taxonomy = pd.read_csv("taxonomy.csv")
bc26_species = set(taxonomy["primary_label"].tolist())
all_secondary_set = set(all_secondary)
in_bc26 = all_secondary_set & bc26_species
out_bc26 = all_secondary_set - bc26_species

print(f"=== BC2026 234 species との overlap ===")
print(f"BC2026 species: {len(bc26_species)}")
print(f"Secondary unique: {len(all_secondary_set)}")
print(f"  ★ Secondary in BC2026: {len(in_bc26)} (rescue 対象)")
print(f"  Secondary NOT in BC2026: {len(out_bc26)} (使えない)")
print()

# rescue 対象 occurrences
rescue_total = sum(1 for s in all_secondary if s in bc26_species)
print(f"Rescue-able secondary 出現数: {rescue_total} (出現中の {rescue_total/len(all_secondary):.1%})")
print()

# どの primary に rescue 機会多い?
df["n_rescue"] = df["secondary_parsed"].apply(
    lambda lst: sum(1 for s in lst if s in bc26_species)
)
print(f"=== rescue 対象を持つ file 数 ===")
has_rescue = df["n_rescue"] > 0
print(f"At least 1 rescue-able secondary: {has_rescue.sum()} files ({has_rescue.mean():.1%})")
print(f"Mean rescue species per file: {df['n_rescue'].mean():.2f}")
print()

# Primary 別、rescue 機会の集中
print(f"=== rescue 機会 / primary species (top 15) ===")
primary_stats = df.groupby("primary_label").agg(
    n_files=("primary_label", "count"),
    n_with_rescue=("n_rescue", lambda x: (x > 0).sum()),
    total_rescue_secondaries=("n_rescue", "sum"),
).sort_values("total_rescue_secondaries", ascending=False)
print(primary_stats.head(15))

print()
print("=== Class diversity ===")
cls_counts = taxonomy["class_name"].value_counts()
print(cls_counts)
