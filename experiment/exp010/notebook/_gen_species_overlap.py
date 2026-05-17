"""Generate species_overlap.ipynb — BirdCLEF past-year species overlap analysis."""
import json
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Notebook cells
# ---------------------------------------------------------------------------

CELLS = []

def code(src: str):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.lstrip("\n"),
    })

# ------------------------------------------------------------------
code("""\
# Cell 1 — Imports & paths
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_2026 = Path("/kaggle/input/birdclef-2026")
WORK = Path("/kaggle/working/past_data")
WORK.mkdir(parents=True, exist_ok=True)

print("Base exists:", BASE_2026.exists())
""")

# ------------------------------------------------------------------
code("""\
# Cell 2 — Load 2026 taxonomy (ground truth)
tax26 = pd.read_csv(BASE_2026 / "taxonomy.csv")
sp26_sci = set(tax26["scientific_name"].str.strip().str.lower())

print(f"BirdCLEF 2026: {len(tax26)} species")
print("Class breakdown:")
print(tax26["class_name"].value_counts().to_string())
display(tax26.head(3))
""")

# ------------------------------------------------------------------
code("""\
# Cell 3 — Download past-year metadata via kaggle CLI
#   Downloads: taxonomy.csv, train.csv, train_metadata.csv (whichever exist)
#   Only metadata CSVs — no audio.

COMPS = [
    "birdclef-2025",
    "birdclef-2024",
    "birdclef-2023",
    "birdclef-2022",
    "birdclef-2021",
]
TARGET_FILES = ["taxonomy.csv", "train.csv", "train_metadata.csv"]


def dl(comp: str, fname: str, dest_dir: Path) -> bool:
    out = dest_dir / fname
    if out.exists():
        return True
    r = subprocess.run(
        ["kaggle", "competitions", "download", "-c", comp, "-f", fname, "-p", str(dest_dir)],
        capture_output=True, text=True, timeout=120,
    )
    # kaggle CLI may return a .zip
    zip_p = dest_dir / (fname + ".zip")
    if zip_p.exists():
        with zipfile.ZipFile(zip_p) as z:
            z.extractall(dest_dir)
        zip_p.unlink()
    return out.exists()


for comp in COMPS:
    d = WORK / comp
    d.mkdir(exist_ok=True)
    found = []
    for f in TARGET_FILES:
        ok = dl(comp, f, d)
        if ok:
            found.append(f)
    print(f"{comp}: downloaded {found if found else 'NONE'}")
""")

# ------------------------------------------------------------------
code("""\
# Cell 4 — Extract species lists from each year
#
# Strategy:
#   1. taxonomy.csv  → scientific_name column  (preferred)
#   2. train.csv     → scientific_name column  (fallback)
#   3. train.csv     → primary_label column    (ebird codes — need cross-ref)
#
# For primary_label (ebird), we need the matching taxonomy to resolve sci names.
# We'll report "ebird codes only" separately.

def load_species(comp_dir: Path):
    \"\"\"Returns (set_of_sci_names_lower, set_of_primary_labels, source_info).\"\"\"
    sci_names = set()
    primary_labels = set()
    sources = []

    for fname in ["taxonomy.csv", "train.csv", "train_metadata.csv"]:
        p = comp_dir / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            continue

        sources.append(fname)

        if "scientific_name" in df.columns:
            s = set(df["scientific_name"].dropna().str.strip().str.lower().unique())
            sci_names |= s

        if "primary_label" in df.columns:
            pl = set(df["primary_label"].dropna().str.strip().unique())
            primary_labels |= pl

    return sci_names, primary_labels, sources


year_data = {}
for comp in COMPS:
    year = int(comp.split("-")[1])
    sci, pl, srcs = load_species(WORK / comp)
    year_data[year] = {"sci": sci, "pl": pl, "srcs": srcs}
    print(f"\\nBirdCLEF {year}: sci_names={len(sci)}, primary_labels={len(pl)}, files={srcs}")
""")

# ------------------------------------------------------------------
code("""\
# Cell 5 — Overlap analysis
rows = []
for year in sorted(year_data.keys(), reverse=True):
    d = year_data[year]
    sci = d["sci"]
    overlap_sci = sci & sp26_sci
    pct = len(overlap_sci) / len(sp26_sci) * 100

    rows.append({
        "year": year,
        "total_sci_species": len(sci),
        "overlap_count": len(overlap_sci),
        "overlap_pct_of_2026": round(pct, 1),
        "source_files": ", ".join(d["srcs"]),
    })
    print(f"\\n=== BirdCLEF {year} ===")
    print(f"  Total species (sci name): {len(sci)}")
    print(f"  Overlap with 2026: {len(overlap_sci)} / {len(sp26_sci)} ({pct:.1f}%)")
    if overlap_sci:
        sample = sorted(overlap_sci)[:10]
        print(f"  Sample overlapping: {sample}")

summary = pd.DataFrame(rows)
print("\\n=== SUMMARY ===")
display(summary)
""")

# ------------------------------------------------------------------
code("""\
# Cell 6 — Per-class breakdown of overlapping species
print("=== Overlap breakdown by class (BirdCLEF 2026 taxonomy) ===\\n")
for year in sorted(year_data.keys(), reverse=True):
    sci = year_data[year]["sci"]
    overlap = tax26[tax26["scientific_name"].str.lower().isin(sci)]
    print(f"BirdCLEF {year}  (n={len(overlap)})")
    if not overlap.empty:
        print(overlap["class_name"].value_counts().to_string())
    print()
""")

# ------------------------------------------------------------------
code("""\
# Cell 7 — Non-overlapping 2026 species (useful for knowing gaps)
print("=== 2026 species NOT covered by ANY past year ===\\n")
all_past_sci = set()
for d in year_data.values():
    all_past_sci |= d["sci"]

not_covered = tax26[~tax26["scientific_name"].str.lower().isin(all_past_sci)]
print(f"2026 species with NO past-year data: {len(not_covered)} / {len(tax26)}")
print()
print(not_covered[["scientific_name", "common_name", "class_name"]].to_string(index=False))
""")

# ------------------------------------------------------------------
code("""\
# Cell 8 — Save overlap species list for future use
rows_out = []
for year in sorted(year_data.keys(), reverse=True):
    sci = year_data[year]["sci"]
    for _, row in tax26.iterrows():
        if row["scientific_name"].lower() in sci:
            rows_out.append({
                "year": year,
                "scientific_name": row["scientific_name"],
                "common_name": row["common_name"],
                "class_name": row["class_name"],
                "inat_taxon_id": row.get("inat_taxon_id", ""),
                "primary_label": row.get("primary_label", ""),
            })

out_df = pd.DataFrame(rows_out)
out_path = Path("/kaggle/working/overlap_species.csv")
out_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}  ({len(out_df)} rows)")
display(out_df.groupby("year")["scientific_name"].count().rename("overlap_count"))
""")

# ------------------------------------------------------------------
code("""\
# Cell 9 — Load 2026 train sample counts
train26 = pd.read_csv(BASE_2026 / "train.csv")
cnt = train26["primary_label"].value_counts().rename("n_train")
tax26 = tax26.join(cnt, on="primary_label")
tax26["n_train"] = tax26["n_train"].fillna(0).astype(int)

print("2026 sample count distribution:")
bins = [0, 1, 5, 10, 20, 50, 100, 9999]
labels = ["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
tax26["bucket"] = pd.cut(tax26["n_train"], bins=bins, labels=labels, right=False)
print(tax26["bucket"].value_counts().sort_index().to_string())
""")

# ------------------------------------------------------------------
code("""\
# Cell 10 — Overlap × 2026 sample counts (2025 & 2021 only)
TARGET_YEARS = [2025, 2021]

for year in TARGET_YEARS:
    if year not in year_data:
        print(f"BirdCLEF {year}: no data loaded")
        continue
    sci = year_data[year]["sci"]
    overlap = tax26[tax26["scientific_name"].str.lower().isin(sci)].copy()
    overlap = overlap.sort_values("n_train")

    print(f"\\n{'='*65}")
    print(f"BirdCLEF {year}  —  {len(overlap)} overlapping species")
    print(f"{'='*65}")
    display(overlap[["scientific_name", "common_name", "class_name", "n_train"]].reset_index(drop=True))

    low = overlap[overlap["n_train"] < 20]
    print(f"  → n_train < 20: {len(low)} species")
    print(f"  → n_train == 0: {(overlap['n_train']==0).sum()} species")
""")

# ------------------------------------------------------------------
code("""\
# Cell 11 — Combined low-sample species across all past years
all_past_sci = set()
for d in year_data.values():
    all_past_sci |= d["sci"]

all_overlap = tax26[tax26["scientific_name"].str.lower().isin(all_past_sci)].copy()
low_sample = all_overlap[all_overlap["n_train"] < 20].sort_values("n_train")

print(f"Past-year overlap AND n_train < 20 in 2026: {len(low_sample)} species")
print()
display(low_sample[["scientific_name", "common_name", "class_name", "n_train"]].reset_index(drop=True))

# Save
out_path = Path("/kaggle/working/overlap_low_sample.csv")
low_sample.to_csv(out_path, index=False)
print(f"\\nSaved: {out_path}")
""")

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": CELLS,
}

out_path = HERE / "species_overlap.ipynb"
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out_path}")
print(f"Cells: {len(CELLS)}")
