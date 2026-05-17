"""Check whether 'far median distance' species have ANY recordings near Pantanal."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("C:/Users/maeke/work/kaggle/birdclef-2026")
df = pd.read_csv(ROOT / "scripts/species_distance_pantanal.csv")

# Re-compute with finer detail
train = pd.read_csv(ROOT / "train.csv")
PANTANAL_LAT, PANTANAL_LON = -19.0, -56.75


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


train["dist_km"] = haversine(train["latitude"], train["longitude"],
                              PANTANAL_LAT, PANTANAL_LON)
train["near_pantanal"] = train["dist_km"] < 500
train["in_brazil"]     = train["dist_km"] < 2000

# 4 domestic species + a few risky ones
TARGETS = ["redjun", "47144", "74113", "209233",
           "houspa", "osprey", "greyel", "brnowl", "bbwduc", "bkhpar"]

print(f"{'label':<10} {'sci_name':<25} {'n':<5} {'near_500km':<12} {'in_2000km':<10} {'min_km':<10} {'p10_km':<10}")
print("-" * 90)
for lbl in TARGETS:
    sub = train[train["primary_label"].astype(str) == lbl]
    if len(sub) == 0:
        print(f"{lbl}: NO records")
        continue
    sci = sub["scientific_name"].iloc[0] if "scientific_name" in sub else "?"
    n_near = sub["near_pantanal"].sum()
    n_brazil = sub["in_brazil"].sum()
    min_km = sub["dist_km"].min()
    p10 = sub["dist_km"].quantile(0.1)
    print(f"{lbl:<10} {str(sci)[:24]:<25} {len(sub):<5} {n_near:<12} {n_brazil:<10} {min_km:<10.0f} {p10:<10.0f}")

print()
print("=== Interpretation ===")
print("near_500km > 0 ⇒ 実際に Pantanal 近接で録音された個体あり、hard-zero 危険")
print("in_2000km > 0  ⇒ S.America 大陸内で録音されている、hard-zero 中リスク")
print("min_km > 2000  ⇒ Pantanal 以遠のみで録音、hard-zero 比較的安全")
