"""Extract non-Aves BC2026 species (with valid scientific names, excluding sonotypes)."""
import pandas as pd
import re
tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
print(f"Total: {len(tax)} species")
print(f"  class breakdown: {tax['class_name'].value_counts().to_dict()}")

# Non-Aves
non_aves = tax[tax["class_name"] != "Aves"].reset_index(drop=True)
print(f"\nNon-Aves total: {len(non_aves)}")
print(f"  by class: {non_aves['class_name'].value_counts().to_dict()}")

# Filter: scientific_name が valid な species (sonotype は除外)
def is_valid_scientific(name):
    """Genus species 形式 (2 単語以上、英字主) は valid。sonotype系は除外。"""
    s = str(name).strip()
    if "sonotype" in s.lower():
        return False
    parts = s.split()
    if len(parts) < 2:
        return False
    # Genus は大文字始まり、species は小文字始まり
    if not parts[0][0].isupper():
        return False
    return True

non_aves["is_valid"] = non_aves["scientific_name"].apply(is_valid_scientific)
queryable = non_aves[non_aves["is_valid"]].reset_index(drop=True)
non_queryable = non_aves[~non_aves["is_valid"]].reset_index(drop=True)

print(f"\nQueryable non-Aves (valid scientific name): {len(queryable)}")
print(f"  by class: {queryable['class_name'].value_counts().to_dict()}")
print(f"\nNon-queryable (sonotype/invalid): {len(non_queryable)}")
print(f"  by class: {non_queryable['class_name'].value_counts().to_dict()}")

print(f"\n=== Queryable list ===")
for _, r in queryable.iterrows():
    print(f"  ({r['scientific_name']!r}, {r['primary_label']!r}, {r['class_name']!r}),")

# Save for embedding
with open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_non_aves_list.py", "w", encoding="utf-8") as f:
    f.write(f"# Non-Aves BC2026 species (queryable scientific names)\n")
    f.write(f"# Total: {len(queryable)} species\n\n")
    f.write("BC2026_NON_AVES = [\n")
    for _, r in queryable.iterrows():
        f.write(f"    ({r['scientific_name']!r}, {r['primary_label']!r}, {r['class_name']!r}),\n")
    f.write("]\n")
print(f"\nSaved: _non_aves_list.py")

print(f"\n=== Excluded (non-queryable) sample ===")
for _, r in non_queryable.head(10).iterrows():
    print(f"  {r['primary_label']}: {r['scientific_name']} ({r['class_name']})")
