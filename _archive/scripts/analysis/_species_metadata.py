"""Find species that can be uniquely identified by metadata.

Sources:
1. train.csv - location, class_name, type, author
2. taxonomy.csv - class_name, scientific_name
3. train_soundscapes_labels.csv - site + time + species (REAL ground truth)
"""
import pandas as pd
import re
from collections import defaultdict

train_df = pd.read_csv("train.csv")
taxo_df = pd.read_csv("taxonomy.csv")
ss_df = pd.read_csv("train_soundscapes_labels.csv")

print(f"train.csv:          {len(train_df)} rows, {train_df['primary_label'].nunique()} species")
print(f"taxonomy.csv:       {len(taxo_df)} species")
print(f"train_sc_labels.csv:{len(ss_df)} rows, {ss_df['primary_label'].nunique()} species labeled in soundscapes")
print()

# === 1. train_soundscapes_labels: site + hour 分析 ===
print("=" * 90)
print("=== 1. train_soundscapes_labels.csv columns ===")
print("=" * 90)
print(ss_df.columns.tolist())
print(ss_df.head(3))
print()

# Parse filename for site + time
# Format example: BC2026_Test_0001_S05_20250227_010002_5.ogg
# S05 = site, 20250227 = date, 010002 = hour/min/sec
def parse_filename(fn):
    m = re.search(r"S(\d{2})_(\d{8})_(\d{6})", fn)
    if m:
        site = "S" + m.group(1)
        date = m.group(2)
        time = m.group(3)
        hour = int(time[:2])
        return site, date, hour
    return None, None, None

ss_df["site"], ss_df["date"], ss_df["hour"] = zip(*ss_df["filename"].apply(parse_filename))
print(f"Unique sites: {ss_df['site'].nunique()}")
print(f"  sites: {sorted(ss_df['site'].dropna().unique())}")
print(f"Unique dates: {ss_df['date'].nunique()}")
print(f"Hour distribution:")
print(ss_df['hour'].value_counts().sort_index().to_string())
print()

# === 2. Species appearing in ONLY 1 site ===
print("=" * 90)
print("=== 2. Species appearing in ONLY 1 site ===")
print("=" * 90)
sp_sites = ss_df.groupby("primary_label")["site"].apply(lambda x: set(x.dropna()))
sp_site_counts = ss_df.groupby("primary_label").size()

# Species in only 1 site (with at least 3 samples to be reliable)
single_site = []
for sp, sites in sp_sites.items():
    if len(sites) == 1 and sp_site_counts[sp] >= 3:
        single_site.append((sp, list(sites)[0], sp_site_counts[sp]))

print(f"Total species in train_sc labels: {len(sp_sites)}")
print(f"Species in only 1 site (n>=3): {len(single_site)}")
for sp, site, n in sorted(single_site, key=lambda x: -x[2])[:25]:
    taxo_row = taxo_df[taxo_df["primary_label"] == sp]
    if len(taxo_row) > 0:
        cls = taxo_row["class_name"].iloc[0]
        sci = taxo_row["scientific_name"].iloc[0]
    else:
        cls = "?"
        sci = "?"
    print(f"  {sp:10} site={site}  n={n}  {cls:9} {sci[:35]}")

# === 3. Species with HIGHLY restricted hour range ===
print()
print("=" * 90)
print("=== 3. Species appearing only in narrow hour range ===")
print("=" * 90)
sp_hours = ss_df.groupby("primary_label")["hour"].apply(lambda x: sorted(set(x.dropna())))
sp_n = ss_df.groupby("primary_label").size()

narrow_hour = []
for sp, hours in sp_hours.items():
    if sp_n[sp] >= 5 and len(hours) <= 3:
        narrow_hour.append((sp, hours, sp_n[sp]))

print(f"Species with hour range <= 3 distinct hours AND n>=5: {len(narrow_hour)}")
for sp, hours, n in sorted(narrow_hour, key=lambda x: -x[2])[:20]:
    taxo_row = taxo_df[taxo_df["primary_label"] == sp]
    if len(taxo_row) > 0:
        cls = taxo_row["class_name"].iloc[0]
        sci = taxo_row["scientific_name"].iloc[0]
    else:
        cls = "?"; sci = "?"
    print(f"  {sp:10} hours={hours} n={n}  {cls:9} {sci[:35]}")

# === 4. Class breakdown of train_sc labeled species ===
print()
print("=" * 90)
print("=== 4. Class breakdown of train_sc labeled species ===")
print("=" * 90)
ss_with_class = ss_df.merge(taxo_df[["primary_label", "class_name"]], on="primary_label", how="left")
print(ss_with_class["class_name"].value_counts())
print()
print(f"Unique Aves species in labels: {ss_with_class[ss_with_class['class_name']=='Aves']['primary_label'].nunique()}")
print(f"Unique non-Aves species in labels: {ss_with_class[ss_with_class['class_name']!='Aves']['primary_label'].nunique()}")
print()

# === 5. Species in train.csv but NOT in train_sc_labels (ghost species) ===
print("=" * 90)
print("=== 5. Ghost species (in taxonomy but not in train_sc labels) ===")
print("=" * 90)
all_sp = set(taxo_df["primary_label"])
labeled_sp = set(ss_df["primary_label"])
ghost = all_sp - labeled_sp
in_train = set(train_df["primary_label"])
ghost_in_train = ghost & in_train
ghost_not_in_train = ghost - in_train

print(f"Total species in taxonomy: {len(all_sp)}")
print(f"Species labeled in train_sc: {len(labeled_sp)}")
print(f"Ghost species (no train_sc labels): {len(ghost)}")
print(f"  - in train.csv: {len(ghost_in_train)}")
print(f"  - NOT in train.csv (extreme ghosts, 28 ghost species per OpPrime): {len(ghost_not_in_train)}")
print()
print("Ghost species NOT in train.csv (extreme ghosts) — first 20:")
for sp in sorted(ghost_not_in_train)[:20]:
    taxo_row = taxo_df[taxo_df["primary_label"] == sp]
    if len(taxo_row) > 0:
        print(f"  {sp:15} class={taxo_row['class_name'].iloc[0]:10} sci={taxo_row['scientific_name'].iloc[0][:35]}")
