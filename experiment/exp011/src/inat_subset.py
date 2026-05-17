"""Build iNatSounds 2024 subset mapping for BirdCLEF 2026 234 species.

Reads iNatSounds train.json (locally downloaded) and BirdCLEF 2026 taxonomy.csv,
matches on scientific_name, and writes:

1. inat_subset_mapping.csv — per-species mapping
   cols: inat_category_id, scientific_name, birdclef_primary_label, class_name, n_audio

2. inat_subset_filelist.csv — per-audio file list for matched categories
   cols: inat_category_id, scientific_name, birdclef_primary_label, file_name

Usage (local, after downloading train.json):
    python experiment/exp011/src/inat_subset.py

Output is consumed by the iNat mel_cache notebook on Kaggle to filter 121GB → ~10GB.
"""
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent
TRAIN_JSON = Path(os.environ.get("TEMP", "/tmp")) / "train.json"
TAXONOMY = REPO / "taxonomy.csv"
OUT_MAP = HERE / "inat_subset_mapping.csv"
OUT_FILES = HERE / "inat_subset_filelist.csv"

print(f"Reading {TRAIN_JSON}...")
with open(TRAIN_JSON, encoding="utf-8") as f:
    d = json.load(f)

cats = pd.DataFrame(d["categories"])
anns = pd.DataFrame(d["annotations"])
audio = pd.DataFrame(d["audio"])
print(f"  categories: {len(cats):,}")
print(f"  annotations: {len(anns):,}")
print(f"  audio: {len(audio):,}")

# Normalize scientific_name for join
cats["sci_l"] = cats["name"].str.strip().str.lower()

tax = pd.read_csv(TAXONOMY)
tax["sci_l"] = tax["scientific_name"].str.strip().str.lower()

# Match on scientific_name
matched = tax.merge(
    cats[["id", "sci_l", "class", "audio_dir_name"]].rename(
        columns={"id": "inat_category_id", "class": "inat_class"}
    ),
    on="sci_l", how="inner"
)
print(f"\nMatched species: {len(matched)} / {len(tax)}")

# Per-class breakdown
print("\nClass breakdown:")
print(matched.groupby("class_name").size().to_dict())

# Count audio per matched category
ann_counts = anns["category_id"].value_counts().to_dict()
matched["n_audio"] = matched["inat_category_id"].map(ann_counts).fillna(0).astype(int)

# Save mapping
mapping_df = matched[[
    "inat_category_id", "scientific_name", "primary_label",
    "class_name", "inat_class", "audio_dir_name", "n_audio"
]].rename(columns={"primary_label": "birdclef_primary_label"}).sort_values(
    ["class_name", "scientific_name"]
).reset_index(drop=True)

mapping_df.to_csv(OUT_MAP, index=False)
print(f"\nWritten: {OUT_MAP} ({len(mapping_df)} rows)")
print(f"Total audio files to extract: {mapping_df['n_audio'].sum():,}")

# Build file list: join annotations -> audio -> matched category
matched_ids = set(mapping_df["inat_category_id"])
anns_sub = anns[anns["category_id"].isin(matched_ids)].copy()

# audio: id -> file_name, duration
audio_fname = dict(zip(audio["id"], audio["file_name"]))
audio_dur = dict(zip(audio["id"], audio["duration"]))
anns_sub["file_name"] = anns_sub["audio_id"].map(audio_fname)
anns_sub["duration"] = anns_sub["audio_id"].map(audio_dur)
assert anns_sub["file_name"].notna().all(), "Missing file_name for some annotations"

# Add species info via mapping
cat_to_sci = dict(zip(mapping_df["inat_category_id"], mapping_df["scientific_name"]))
cat_to_pl = dict(zip(mapping_df["inat_category_id"], mapping_df["birdclef_primary_label"]))

files_df = pd.DataFrame({
    "inat_category_id": anns_sub["category_id"],
    "scientific_name": anns_sub["category_id"].map(cat_to_sci),
    "birdclef_primary_label": anns_sub["category_id"].map(cat_to_pl),
    "file_name": anns_sub["file_name"],
    "duration": anns_sub["duration"],
}).sort_values(["birdclef_primary_label", "file_name"]).reset_index(drop=True)

files_df.to_csv(OUT_FILES, index=False)
print(f"Written: {OUT_FILES} ({len(files_df):,} audio files)")

# Sanity prints
print("\n=== Top 20 species by n_audio ===")
print(mapping_df.sort_values("n_audio", ascending=False).head(20)[
    ["scientific_name", "class_name", "n_audio"]
].to_string(index=False))

print("\n=== Per-class total audio ===")
print(mapping_df.groupby("class_name")["n_audio"].agg(["count", "sum", "mean", "min", "max"]).round(1))
