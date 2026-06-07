"""Find types unique to 1-3 species."""
import pandas as pd
import ast

df = pd.read_csv("train.csv")

def parse_types(s):
    try:
        return [t.strip() for t in ast.literal_eval(s)]
    except:
        return []

df["type_list"] = df["type"].apply(parse_types)

# Type → species map
type_to_species = {}
for _, row in df.iterrows():
    for t in row["type_list"]:
        if t not in type_to_species:
            type_to_species[t] = set()
        type_to_species[t].add(row["primary_label"])

print(f"Total unique type values: {len(type_to_species)}")
print()

# Sort by # species using this type
sorted_types = sorted(type_to_species.items(), key=lambda x: len(x[1]))

# Types used by ONLY 1 species (unique signature potential)
print("=" * 80)
print("=== Types used by ONLY 1 species (unique acoustic signature potential) ===")
print("=" * 80)
n_shown = 0
for t, sp_set in sorted_types:
    if len(sp_set) != 1:
        continue
    sp = list(sp_set)[0]
    sci = df[df["primary_label"] == sp]["scientific_name"].iloc[0]
    cls = df[df["primary_label"] == sp]["class_name"].iloc[0]
    n_with_type = sum(1 for _, row in df.iterrows() if row["primary_label"]==sp and t in row["type_list"])
    n_total = (df["primary_label"]==sp).sum()
    if n_with_type >= 2:  # at least 2 samples to be useful
        print(f"  type=[{t!r:35}] sp={sp} {cls} {sci[:30]} n={n_with_type}/{n_total}")
        n_shown += 1
        if n_shown >= 20:
            break

print()
print("=" * 80)
print("=== Types used by 2-3 species (small group) ===")
print("=" * 80)
n_shown = 0
for t, sp_set in sorted_types:
    if not (2 <= len(sp_set) <= 3):
        continue
    species_info = []
    for sp in sp_set:
        sci = df[df["primary_label"] == sp]["scientific_name"].iloc[0]
        n_total = (df["primary_label"]==sp).sum()
        species_info.append(f"{sp}({sci[:20]} n={n_total})")
    print(f"  type=[{t!r:30}] -> {len(sp_set)} species: {', '.join(species_info)}")
    n_shown += 1
    if n_shown >= 20:
        break
