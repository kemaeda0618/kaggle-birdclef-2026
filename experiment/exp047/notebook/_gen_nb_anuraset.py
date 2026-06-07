"""Generate AnuraSet extract NB (CPU + Internet for GitHub annotation DL)."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_anuraset_extract.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species_with_inat.txt").read_text(encoding="utf-8")
species_lines = []
for line in species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        species_lines.append("    " + s)
species_block = "\n".join(species_lines)

nb = {
    "cells": [
        md_cell("hdr", """# exp047 AnuraSet Extract (CPU + Internet)

**Purpose**: AnuraSet v2 から BC2026 Amphibia species filter extract

**Input**:
- `bengtlueers/anuraset-v2-raw` (raw audio only)
- Internet ON: GitHub から annotation CSV DL

**Output**: `maekeso/birdclef2026-exp047-anuraset-extracted`

**Structure**:
- Audio: `AnuraSet_v2.0.0_raw/{site}/{site}_{date}_{time}.wav` (multi-species soundscape)
- Annotation: GitHub <https://github.com/soundclim/anuraset> から DL

**Match**: annotation CSV の scientific_name → BC2026 Amphibia
"""),
        code_cell("setup", """import os, sys, json, time, re, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import urllib.request

OUT_DIR = Path("/kaggle/working/extracted")
OUT_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR = OUT_DIR / "audio" / "anuraset"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MAX_PER_SPECIES = 200
STORAGE_CAP_GB = 10.0
START_T = time.time()
"""),
        code_cell("species", f"""__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
SCI_LC_TO_LABEL = {{str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
SCI_LC_SET = set(SCI_LC_TO_LABEL.keys())
# Amphibia subset
amphibia_df = species_df[species_df['class_name'] == 'Amphibia']
print(f"BC2026: {{len(species_df)}} species, Amphibia: {{len(amphibia_df)}}")
"""),
        code_cell("download_annotation", """# AnuraSet annotation from Zenodo (official, doi: 10.5281/zenodo.8342596)
import urllib.request, zipfile

ZENODO_BASE = "https://zenodo.org/records/8342596/files"

# 1. Download species.csv (BC2026 match に使う scientific_name list)
species_csv_path = OUT_DIR / "anuraset_species.csv"
print(f"DL species.csv...")
urllib.request.urlretrieve(f"{ZENODO_BASE}/species.csv?download=1", species_csv_path)
species_meta = pd.read_csv(species_csv_path)
print(f"  ✓ {len(species_meta)} species")
print(f"  Columns: {species_meta.columns.tolist()}")
print(species_meta.head(5).to_string())

# 2. Download weak_labels.csv (1-min level annotations, fine for our purpose)
weak_path = OUT_DIR / "anuraset_weak_labels.csv"
print(f"\\nDL weak_labels.csv...")
urllib.request.urlretrieve(f"{ZENODO_BASE}/weak_labels.csv?download=1", weak_path)
weak_df = pd.read_csv(weak_path)
print(f"  ✓ {len(weak_df)} annotations")
print(f"  Columns: {weak_df.columns.tolist()}")
print(weak_df.head(3).to_string())

# 3. Download strong_labels.zip (precise segment annotations)
strong_zip = OUT_DIR / "strong_labels.zip"
strong_dir = OUT_DIR / "strong_labels"
print(f"\\nDL strong_labels.zip...")
try:
    urllib.request.urlretrieve(f"{ZENODO_BASE}/strong_labels.zip?download=1", strong_zip)
    strong_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(strong_zip) as zf:
        zf.extractall(strong_dir)
    print(f"  Extracted to {strong_dir}")
    # List
    for f in strong_dir.rglob("*"):
        if f.is_file():
            print(f"    {f.relative_to(strong_dir)}  ({f.stat().st_size/1024:.1f} KB)")
except Exception as e:
    print(f"  strong_labels DL err: {str(e)[:100]}")
    strong_dir = None

# Use weak_labels as primary annotation (full coverage, faster)
annotation_df = weak_df

# Map species in annotation to scientific names
if "scientific_name" not in annotation_df.columns and "species" in species_meta.columns:
    # weak_labels may have species_code, join via species_meta
    species_code_col = next((c for c in annotation_df.columns if c.lower() in ["species_code", "code", "label"]), None)
    if species_code_col:
        sci_lookup = dict(zip(species_meta[species_meta.columns[0]], species_meta["species"]))
        annotation_df["scientific_name"] = annotation_df[species_code_col].map(sci_lookup)
print(f"\\nFinal annotation: {len(annotation_df)} rows, cols: {annotation_df.columns.tolist()}")
"""),
        code_cell("extract", """def safe_dir(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))

def check_storage_gb():
    try:
        return sum(f.stat().st_size for f in AUDIO_DIR.rglob("*") if f.is_file()) / 1e9
    except: return 0.0

ANURA_CANDIDATES = [
    Path("/kaggle/input/datasets/bengtlueers/anuraset-v2-raw"),
    Path("/kaggle/input/anuraset-v2-raw"),
]
anura_root = next((p for p in ANURA_CANDIDATES if p.exists()), None)
if anura_root is None:
    raise RuntimeError(f"AnuraSet NOT MOUNTED: {ANURA_CANDIDATES}")

data_dir = None
for cand in [anura_root / "AnuraSet_v2.0.0_raw", anura_root]:
    if cand.exists() and any(c.is_dir() for c in cand.iterdir()):
        data_dir = cand; break
print(f"AnuraSet data_dir: {data_dir}")

# === AnuraSet uses WIDE-format multi-label annotation ===
# species.csv: FAMILY, SPECIES (scientific_name), CODE (6-letter)
# weak_labels.csv: MONITORING_SITE, AUDIO_FILE_ID, SPECIES_XXX (one col per species), 0-3 calling activity

# Build CODE → scientific_name map from species.csv
code_to_sci = {}
code_col = next((c for c in species_meta.columns if c.upper() == "CODE"), None)
sci_col_meta = next((c for c in species_meta.columns if c.upper() == "SPECIES"), None)
print(f"species.csv columns: {species_meta.columns.tolist()}")
print(f"  CODE col: {code_col}, SPECIES col: {sci_col_meta}")
if not code_col or not sci_col_meta:
    raise RuntimeError(f"species.csv missing CODE or SPECIES columns: {species_meta.columns.tolist()}")
for _, r in species_meta.iterrows():
    code_to_sci[str(r[code_col]).upper()] = str(r[sci_col_meta])
print(f"  Built CODE→sci map: {len(code_to_sci)} species")

# weak_labels.csv: identify species columns (SPECIES_*)
species_cols = [c for c in weak_df.columns if c.startswith("SPECIES_")]
print(f"\\nweak_labels.csv: {len(weak_df)} rows, {len(species_cols)} species columns")
print(f"  Sample species cols: {species_cols[:5]}")

# For each species column, get scientific_name + check BC2026 match
bc26_species_cols = []  # list of (species_col, primary_label, sci_lc)
for col in species_cols:
    code = col.replace("SPECIES_", "").upper()
    sci = code_to_sci.get(code, "")
    sci_lc = sci.lower()
    if sci_lc in SCI_LC_SET:
        primary_label = SCI_LC_TO_LABEL[sci_lc]
        bc26_species_cols.append((col, primary_label, sci_lc, code))

print(f"\\nBC2026 match species: {len(bc26_species_cols)}")
for col, lbl, sci, code in bc26_species_cols[:20]:
    n_pos = (weak_df[col] > 0).sum()
    print(f"  {code:8s} → {sci:35s} → {lbl}  ({n_pos} files positive)")
if not bc26_species_cols:
    print(f"\\n  All AnuraSet species: {[code_to_sci.get(c.replace('SPECIES_',''), '?') for c in species_cols[:10]]}")
    print(f"  BC2026 Amphibia sci_name sample: {[s for s in SCI_LC_SET if s.startswith('boa') or s.startswith('lep')][:10]}")
    raise RuntimeError("0 BC2026 species matched in AnuraSet")

# For each BC2026 species, find audio files where this species has positive activity
all_metadata = []
n_copied = 0
audio_id_col = "AUDIO_FILE_ID"
site_col = "MONITORING_SITE"

for col, primary_label, sci_lc, code in tqdm(bc26_species_cols, desc="AnuraSet species"):
    if check_storage_gb() > STORAGE_CAP_GB: break
    # Get audio files with this species positive
    pos_rows = weak_df[weak_df[col] > 0].head(MAX_PER_SPECIES)
    if len(pos_rows) == 0: continue

    dst_dir = AUDIO_DIR / safe_dir(primary_label)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for _, r in pos_rows.iterrows():
        if check_storage_gb() > STORAGE_CAP_GB: break
        audio_id = str(r[audio_id_col])
        site = str(r[site_col])
        # bengtlueers structure: <site_no_zero>/{audio_id}.wav
        # site in weak_labels = "INCT04", file in bengtlueers = "INCT4"
        site_normalized = site.replace("INCT0", "INCT")  # remove leading zero
        # Try multiple path patterns
        candidate_paths = [
            data_dir / site / f"{audio_id}.wav",
            data_dir / site_normalized / f"{audio_id}.wav",
            data_dir / f"{audio_id}.wav",
        ]
        # Fallback: rglob
        src = next((p for p in candidate_paths if p.exists()), None)
        if src is None:
            for p in data_dir.rglob(f"{audio_id}.wav"):
                src = p; break
        if src is None: continue

        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size > 100: continue
        try:
            shutil.copy2(src, dst)
            n_copied += 1
            all_metadata.append({
                "filename": str(dst.relative_to(OUT_DIR)),
                "primary_label": primary_label,
                "scientific_name": sci_lc,
                "source": "anuraset",
                "class_name": "Amphibia",
                "activity": int(r[col]),
                "audio_id": audio_id,
                "site": site,
                "file_size_mb": dst.stat().st_size / 1e6,
            })
        except Exception as e:
            pass

print(f"\\nCopied: {n_copied} files from {len(bc26_species_cols)} species")
print(f"Storage: {check_storage_gb():.2f} GB, time: {(time.time()-START_T)/60:.1f} min")

if all_metadata:
    pd.DataFrame(all_metadata).to_csv(OUT_DIR / "metadata.csv", index=False)
"""),
        code_cell("upload", """import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Clean up temporary annotation files before upload (avoid Kaggle SDK token bug
# from too many file uploads)
print("Cleaning OUT_DIR for upload...")
for f in OUT_DIR.iterdir():
    # Keep only audio/ dir and metadata.csv
    if f.name in ("audio", "metadata.csv"):
        continue
    try:
        if f.is_file():
            f.unlink()
            print(f"  removed file: {f.name}")
        elif f.is_dir():
            import shutil
            shutil.rmtree(f)
            print(f"  removed dir: {f.name}")
    except Exception as e:
        print(f"  rm err {f.name}: {e}")

print(f"\\nOUT_DIR after cleanup:")
for f in OUT_DIR.iterdir():
    print(f"  {f.name}")

USER = "maekeso"; SLUG = "birdclef2026-exp047-anuraset-extracted"; TITLE = "BirdCLEF2026 exp047 AnuraSet Extracted"
DRY_RUN = False
if not DRY_RUN:
    meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        api.dataset_create_new(folder=str(OUT_DIR), public=False, dir_mode="zip", quiet=False)
        print("OK new dataset created")
    except Exception as e:
        print(f"create_new err: {str(e)[:200]}")
        try:
            api.dataset_create_version(folder=str(OUT_DIR), version_notes="AnuraSet extracted v2",
                                        dir_mode="zip", quiet=False)
            print("OK version created")
        except Exception as e2:
            print(f"create_version err: {str(e2)[:200]}")
    print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT} ({len(nb['cells'])} cells)")
