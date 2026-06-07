"""Generate Kaggle NB: extract iNat + AnuraSet to BC2026 species filtered Dataset.

Kaggle CPU, no GPU needed.
Inputs: shadowdude/train-recordings (iNat) + bengtlueers/anuraset-v2-raw
Filter to 234 BC2026 species (mostly non-Aves coverage)
Output: /kaggle/working/extracted/ -> Kaggle Dataset
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp054\notebook\nb1_extract_nonaves.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

# Read all 234 species list
all_species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species.txt").read_text(encoding="utf-8")
all_species_lines = []
for line in all_species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        all_species_lines.append("    " + s)
all_species_block = "\n".join(all_species_lines)

nb = {
    "cells": [
        md_cell("hdr", """# exp054 NB1: Extract iNat + AnuraSet to BC2026-overlap Dataset (Kaggle CPU)

**Purpose**: Phase 2 — non-Aves coverage 補強用データ抽出

**Inputs**:
- `shadowdude/train-recordings` (iNat, 121 GB total)
- `bengtlueers/anuraset-v2-raw` (AnuraSet, 8.5 GB)

**Filter**: BC2026 234 species (主に non-Aves に効く)

**Output**:
- `/kaggle/working/extracted/audio/{source}/{primary_label}/*.ogg|wav`
- `metadata.csv` (filename, primary_label, source, scientific_name)
- Upload → `maekeso/birdclef2026-exp054-nonaves-extracted`

**Expected coverage**:
- iNat: ~24 Amphibia + 4 Mammalia + 3 Insecta + Aves overlap
- AnuraSet: 17 Amphibia
- Union: ~27 Amphibia + 4 Mammalia + 3 Insecta = **34 non-Aves species**

**Expected output size**: ~5-15 GB

**Use case**: Stage 1 v2 input、XC Part 1+2 と合わせて non-Aves backbone 強化
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup + species list
# ============================================================
import os, sys, json, time, re, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

OUT_DIR = Path("/kaggle/working/extracted")
OUT_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR = OUT_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# Safety caps
MAX_PER_SPECIES = 200  # extract 上限/species
STORAGE_CAP_GB = 18.0
print(f"Output: {OUT_DIR}")
"""),

        code_cell("species", """# ============================================================
# Cell 2: BC2026 species list (234)
# ============================================================
__BC2026_SPECIES = [
__ALL_SPECIES_PLACEHOLDER__
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name"])
SCI_LC_TO_LABEL = {str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
LABEL_SET = set(species_df['primary_label'])
SCI_LC_SET = set(SCI_LC_TO_LABEL.keys())
print(f"BC2026: {len(species_df)} species")
print(f"  classes: {species_df['class_name'].value_counts().to_dict()}")
"""),

        code_cell("helpers", """# ============================================================
# Cell 3: Helpers
# ============================================================
def safe_dir(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))

def check_storage_gb():
    try:
        return sum(f.stat().st_size for f in AUDIO_DIR.rglob("*") if f.is_file()) / 1e9
    except Exception:
        return 0.0

all_metadata = []
"""),

        code_cell("anuraset", """# ============================================================
# Cell 4: Extract AnuraSet
# ============================================================
ANURA_CANDIDATES = [
    Path("/kaggle/input/datasets/bengtlueers/anuraset-v2-raw"),
    Path("/kaggle/input/anuraset-v2-raw"),
    Path("/kaggle/input/bengtlueers-anuraset-v2-raw"),
]
anura_root = next((p for p in ANURA_CANDIDATES if p.exists()), None)

if anura_root is None:
    print(f"AnuraSet not mounted - skip. Tried: {ANURA_CANDIDATES}")
else:
    print(f"AnuraSet root: {anura_root}")
    # Find metadata
    meta_candidates = list(anura_root.rglob("metadata*.csv")) + list(anura_root.rglob("*labels*.csv"))
    print(f"  CSV candidates: {[str(c.relative_to(anura_root)) for c in meta_candidates[:5]]}")

    if meta_candidates:
        meta_path = meta_candidates[0]
        df = pd.read_csv(meta_path)
        print(f"  metadata: {len(df)} rows, columns: {df.columns.tolist()[:10]}")

        # Try to find scientific_name column
        sci_col_candidates = ["scientific_name", "species", "species_name", "label", "taxon"]
        sci_col = next((c for c in sci_col_candidates if c in df.columns), None)
        print(f"  sci_col: {sci_col}")

        # Try to find filename column
        name_col_candidates = ["filename", "file", "audio", "path", "wav_path", "audio_path"]
        name_col = next((c for c in name_col_candidates if c in df.columns), None)
        print(f"  name_col: {name_col}")

        if sci_col and name_col:
            # Filter to BC2026 species
            df["_sci_lc"] = df[sci_col].astype(str).str.lower()
            df["_in_bc26"] = df["_sci_lc"].isin(SCI_LC_SET)
            df_match = df[df["_in_bc26"]].reset_index(drop=True)
            print(f"  matched: {len(df_match)} / {len(df)}")
            if len(df_match) > 0:
                print(f"  matched species: {df_match[sci_col].unique().tolist()}")

                # Copy files (with per-species cap)
                df_match = df_match.groupby("_sci_lc", group_keys=False).head(MAX_PER_SPECIES).reset_index(drop=True)
                print(f"  after cap {MAX_PER_SPECIES}: {len(df_match)}")

                n_copied = 0
                for _, r in tqdm(df_match.iterrows(), total=len(df_match), desc="AnuraSet copy"):
                    sci = r[sci_col]
                    label = SCI_LC_TO_LABEL.get(str(sci).lower())
                    if not label: continue

                    rel = str(r[name_col])
                    src = anura_root / rel
                    if not src.exists():
                        # Try anura subdir structure
                        for candidate in anura_root.rglob(Path(rel).name):
                            src = candidate
                            break
                    if not src.exists(): continue

                    dst_dir = AUDIO_DIR / "anuraset" / safe_dir(label)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / src.name
                    if dst.exists(): continue

                    try:
                        shutil.copy2(src, dst)
                        n_copied += 1
                        all_metadata.append({
                            "filename": str(dst.relative_to(OUT_DIR)),
                            "primary_label": label,
                            "scientific_name": str(sci),
                            "source": "anuraset",
                            "class_name": LABEL_TO_CLASS.get(label, ""),
                            "file_size_mb": dst.stat().st_size / 1e6,
                        })
                    except Exception as e:
                        pass

                    if check_storage_gb() > STORAGE_CAP_GB:
                        print(f"⚠ storage cap reached, stopping")
                        break
                print(f"  AnuraSet copied: {n_copied} files")
            else:
                print(f"  no BC2026 species matched in AnuraSet")
        else:
            print(f"  could not identify sci_name / filename columns")
            print(f"  columns full: {df.columns.tolist()}")
"""),

        code_cell("inat", """# ============================================================
# Cell 5: Extract iNat (large 121 GB, careful filter)
# ============================================================
INAT_CANDIDATES = [
    Path("/kaggle/input/datasets/shadowdude/train-recordings"),
    Path("/kaggle/input/train-recordings"),
    Path("/kaggle/input/shadowdude-train-recordings"),
]
inat_root = next((p for p in INAT_CANDIDATES if p.exists()), None)

if inat_root is None:
    print(f"iNat not mounted - skip. Tried: {INAT_CANDIDATES}")
else:
    print(f"iNat root: {inat_root}")
    # iNat structure auto-discovery
    csv_candidates = list(inat_root.rglob("*.csv"))[:10]
    print(f"  CSV candidates: {[str(c.relative_to(inat_root)) for c in csv_candidates]}")

    if csv_candidates:
        meta_path = csv_candidates[0]  # first CSV
        df = pd.read_csv(meta_path)
        print(f"  metadata: {len(df)} rows, columns: {df.columns.tolist()[:10]}")

        sci_col_candidates = ["scientific_name", "species_name", "species", "name", "taxon"]
        sci_col = next((c for c in sci_col_candidates if c in df.columns), None)
        name_col_candidates = ["filename", "file", "audio_id", "path", "filepath", "file_name"]
        name_col = next((c for c in name_col_candidates if c in df.columns), None)
        print(f"  sci_col: {sci_col}, name_col: {name_col}")

        if sci_col and name_col:
            df["_sci_lc"] = df[sci_col].astype(str).str.lower()
            df["_in_bc26"] = df["_sci_lc"].isin(SCI_LC_SET)
            df_match = df[df["_in_bc26"]].reset_index(drop=True)
            print(f"  matched: {len(df_match)} / {len(df)}")
            if len(df_match) > 0:
                print(f"  matched species sample: {df_match[sci_col].unique()[:10].tolist()}")
                print(f"  matched class breakdown:")
                for _, r in df_match.head(50).iterrows():
                    pass

                # per-species cap
                df_match = df_match.groupby("_sci_lc", group_keys=False).head(MAX_PER_SPECIES).reset_index(drop=True)
                print(f"  after cap {MAX_PER_SPECIES}: {len(df_match)}")

                n_copied = 0
                for _, r in tqdm(df_match.iterrows(), total=len(df_match), desc="iNat copy"):
                    sci = r[sci_col]
                    label = SCI_LC_TO_LABEL.get(str(sci).lower())
                    if not label: continue

                    rel = str(r[name_col])
                    src = inat_root / rel
                    if not src.exists():
                        # Try discovery
                        for c in inat_root.rglob(Path(rel).name):
                            src = c
                            break
                    if not src.exists(): continue

                    dst_dir = AUDIO_DIR / "inat" / safe_dir(label)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / src.name
                    if dst.exists(): continue

                    try:
                        shutil.copy2(src, dst)
                        n_copied += 1
                        all_metadata.append({
                            "filename": str(dst.relative_to(OUT_DIR)),
                            "primary_label": label,
                            "scientific_name": str(sci),
                            "source": "inat",
                            "class_name": LABEL_TO_CLASS.get(label, ""),
                            "file_size_mb": dst.stat().st_size / 1e6,
                        })
                    except Exception:
                        pass

                    if check_storage_gb() > STORAGE_CAP_GB:
                        print(f"⚠ storage cap reached, stopping")
                        break
                print(f"  iNat copied: {n_copied} files")
            else:
                print(f"  no BC2026 species matched in iNat")
        else:
            print(f"  could not identify sci_name / filename columns")
            print(f"  columns: {df.columns.tolist()}")
"""),

        code_cell("summary", """# ============================================================
# Cell 6: Summary + save metadata
# ============================================================
if all_metadata:
    final_df = pd.DataFrame(all_metadata)
    print(f"Total extracted: {len(final_df)} files")
    print(f"  by source: {final_df['source'].value_counts().to_dict()}")
    print(f"  by class: {final_df['class_name'].value_counts().to_dict()}")
    print(f"  unique species: {final_df['primary_label'].nunique()} / 234")
    print(f"  total size: {final_df['file_size_mb'].sum()/1024:.2f} GB")

    final_df.to_csv(OUT_DIR / "metadata.csv", index=False)
    print(f"\\nSaved: {OUT_DIR / 'metadata.csv'}")

    # Per-species
    sp = final_df.groupby(["primary_label", "class_name"]).agg(
        n_files=("filename", "count"),
        n_sources=("source", "nunique"),
        total_mb=("file_size_mb", "sum"),
    ).reset_index().sort_values("n_files", ascending=False)
    sp.to_csv(OUT_DIR / "per_species.csv", index=False)
    print(f"\\nTop 10 species:")
    print(sp.head(10).to_string(index=False))
else:
    print("No metadata accumulated")
"""),

        code_cell("upload", """# ============================================================
# Cell 7: Upload as Kaggle Dataset
# ============================================================
import json, shutil
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp054-nonaves-extracted"
TITLE = "BirdCLEF2026 exp054 Non-Aves Extracted (iNat + AnuraSet)"

DRY_RUN = False  # set False to upload

if not DRY_RUN:
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "other"}],
    }
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        api.dataset_create_version(folder=str(OUT_DIR),
                                    version_notes="Phase 2 non-Aves extracted",
                                    dir_mode="zip", quiet=False)
        print("OK new version uploaded")
    except Exception:
        try:
            api.dataset_create_new(folder=str(OUT_DIR), public=False,
                                    dir_mode="zip", quiet=False)
            print("OK new dataset created")
        except Exception as e:
            print(f"upload err: {str(e)[:300]}")
    print(f"URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
else:
    print(f"DRY_RUN={DRY_RUN}, skip upload")
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Inject species
for c in nb["cells"]:
    if c.get("id") == "species":
        src = "".join(c["source"])
        src = src.replace("__ALL_SPECIES_PLACEHOLDER__", all_species_block)
        c["source"] = src.splitlines(keepends=True)
        break

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id'):15s} L={n}")
