"""EDA: per-species median distance from Pantanal center.

Identifies "hard-zero candidates" — species whose training recordings are
geographically far from Pantanal (-19, -57), suggesting they're unlikely
to appear in the test soundscape.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("C:/Users/maeke/work/kaggle/birdclef-2026")
train = pd.read_csv(ROOT / "train.csv")
tax = pd.read_csv(ROOT / "taxonomy.csv")

# Pantanal center (CLAUDE.md: lat -16.5..-21.6, lon -55.9..-57.6)
PANTANAL_LAT = -19.0
PANTANAL_LON = -56.75


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# Filter to records with valid lat/lon
df = train[train["latitude"].notna() & train["longitude"].notna()].copy()
df["dist_km"] = haversine(df["latitude"], df["longitude"], PANTANAL_LAT, PANTANAL_LON)
print(f"Records with lat/lon: {len(df)}/{len(train)}")

# Per-species statistics
agg = df.groupby("primary_label").agg(
    n=("dist_km", "size"),
    median_km=("dist_km", "median"),
    p25_km=("dist_km", lambda x: x.quantile(0.25)),
    p75_km=("dist_km", lambda x: x.quantile(0.75)),
    min_km=("dist_km", "min"),
    max_km=("dist_km", "max"),
).reset_index()

# Join scientific name + class
agg = agg.merge(
    tax[["primary_label", "scientific_name", "common_name", "class_name"]],
    on="primary_label", how="left",
)
agg = agg.sort_values("median_km", ascending=False)

print("\n=== Distance distribution (median per species) ===")
bins = [0, 500, 1000, 2000, 3000, 5000, 10000, 20000]
hist = pd.cut(agg["median_km"], bins=bins).value_counts().sort_index()
for r, c in hist.items():
    print(f"  {r}: {c} species")

print(f"\n=== Hard-zero candidates (median > 5000 km) ===")
hz = agg[agg["median_km"] > 5000].copy()
print(f"Count: {len(hz)}")
print(hz[["primary_label", "scientific_name", "common_name", "class_name", "n", "median_km"]].to_string(index=False))

print(f"\n=== Hard-zero candidates (median > 3000 km) ===")
hz3 = agg[agg["median_km"] > 3000].copy()
print(f"Count: {len(hz3)}")
print(hz3[["primary_label", "scientific_name", "common_name", "class_name", "n", "median_km"]].to_string(index=False))

print(f"\n=== Hard-zero candidates (median > 2000 km) ===")
hz2 = agg[agg["median_km"] > 2000].copy()
print(f"Count: {len(hz2)}")
print(hz2[["primary_label", "scientific_name", "common_name", "class_name", "n", "median_km"]].to_string(index=False))

# Specifically check domestic species (manual list from BACKLOG)
print("\n=== Manually-flagged domestic / unlikely species ===")
domestic_keywords = [
    "Gallus", "Canis familiaris", "Bos taurus", "Equus caballus", "Felis catus",
    "Sus scrofa", "Ovis aries", "Capra hircus",
]
for kw in domestic_keywords:
    hits = agg[agg["scientific_name"].fillna("").str.contains(kw, case=False, na=False)]
    if len(hits):
        for _, r in hits.iterrows():
            print(f"  {r['primary_label']:>15}  {r['scientific_name']:<30}  {r['class_name']:<10}  n={r['n']:<5}  median={r['median_km']:.0f}km")

# Save full table for reference
out_path = Path("scripts/species_distance_pantanal.csv")
agg.to_csv(out_path, index=False)
print(f"\nSaved full table: {out_path}")
