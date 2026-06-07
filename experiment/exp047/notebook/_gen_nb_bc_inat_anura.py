"""Generate Kaggle NB: extract BC2021-2025 + iNat + AnuraSet to BC2026-species Dataset.

Kaggle CPU (GPU optional, not strictly needed for file extraction).
Inputs:
  - birdclef-2021, 2022, 2023, 2024, 2025 (competitions)
  - shadowdude/train-recordings (iNat)
  - bengtlueers/anuraset-v2-raw (AnuraSet)
Filter: BC2026 234 species (Aves + non-Aves)
Output: maekeso/birdclef2026-exp047-bc-inat-anura-extracted
"""
import json
from pathlib import Path

OUT_DIR = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook")
OUT = OUT_DIR / "nb_bc_inat_anura_extract.ipynb"

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
        md_cell("hdr", """# exp047 BC + iNat + AnuraSet Extract (Kaggle)

**Purpose**: BC2026 234 species を **過去 BC + iNat + AnuraSet** から抽出、Phase 2 pretrain 用に統合。

**Inputs** (7 sources):
| Source | Type | 期待 BC2026 coverage |
|---|---|---|
| birdclef-2021 | competition | ~34 Aves species |
| birdclef-2022 | competition | ~5 species |
| birdclef-2023 | competition | ~1 species |
| birdclef-2024 | competition | ~1 species |
| birdclef-2025 | competition | ~42 species |
| shadowdude/train-recordings (iNat) | dataset | ~31 species (含 non-Aves) |
| bengtlueers/anuraset-v2-raw | dataset | 17 Amphibia |

**Filter**: BC2026 234 species (scientific_name または primary_label match)

**Output**:
- `/kaggle/working/extracted/audio/{source}/{primary_label}/*.ogg|wav|mp3`
- `metadata.csv` (filename, primary_label, source, scientific_name, class_name)
- Upload → `maekeso/birdclef2026-exp047-bc-inat-anura-extracted`

**Expected**:
- ~80-150 unique BC2026 species covered (union over sources)
- ~15-25 GB after per-species cap

**Safety**:
- Per-species cap (200 file/species)
- Storage cap 18 GB
- Per-source failure に強い (defensive)

**Note**: 過去 BC competition は **rule accept 必要** (各 URL で "Late Submission" 承諾)
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup
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

# Caps
MAX_PER_SPECIES = 200
STORAGE_CAP_GB = 18.0

START_T = time.time()
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
print(f"BC2026 species: {len(species_df)}")
print(f"  by class: {species_df['class_name'].value_counts().to_dict()}")
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

def safety_ok():
    gb = check_storage_gb()
    if gb > STORAGE_CAP_GB:
        print(f"  ⚠ storage cap reached ({gb:.2f} GB), stopping")
        return False
    return True

all_metadata = []
print("OK helpers defined")
"""),

        code_cell("extract_fn", """# ============================================================
# Cell 4: Generic extract function (handles all sources uniformly)
# ============================================================
def extract_source(label, source_id, df_meta, audio_root, name_col, label_col=None, sci_col=None,
                   max_per_species=MAX_PER_SPECIES):
    \"\"\"Generic extract: match df_meta to BC2026 species, copy audio.

    df_meta: source metadata DataFrame
    audio_root: where audio files live
    name_col: column with filename / relative path
    label_col: column with primary_label (e.g., ebird code)
    sci_col: column with scientific_name (fallback)

    Returns: number of files copied
    \"\"\"
    print(f"\\n{'='*60}\\n[{label}] extract from {source_id}\\n{'='*60}")
    df = df_meta.copy()
    print(f"  source rows: {len(df)}")

    # Match
    if label_col and label_col in df.columns:
        df["_match_label"] = df[label_col].astype(str)
        df["_in_bc26"] = df["_match_label"].isin(LABEL_SET)
        matched_via = f"label_col={label_col}"
    elif sci_col and sci_col in df.columns:
        df["_match_sci"] = df[sci_col].astype(str).str.lower()
        df["_in_bc26"] = df["_match_sci"].isin(SCI_LC_SET)
        df["_match_label"] = df["_match_sci"].apply(lambda s: SCI_LC_TO_LABEL.get(s, None))
        matched_via = f"sci_col={sci_col}"
    else:
        print(f"  [SKIP {label}] no matchable column. cols: {df.columns.tolist()[:10]}")
        return 0

    df_match = df[df["_in_bc26"]].reset_index(drop=True)
    print(f"  matched via {matched_via}: {len(df_match)} / {df['_in_bc26'].count()} = {df['_in_bc26'].sum()} match")
    print(f"  unique BC2026 species matched: {df_match['_match_label'].nunique()}")

    if len(df_match) == 0:
        return 0

    # Per-species cap
    before = len(df_match)
    df_match = df_match.groupby("_match_label", group_keys=False).head(max_per_species).reset_index(drop=True)
    if before > len(df_match):
        print(f"  after per-species cap {max_per_species}: {before} -> {len(df_match)}")

    # Copy files
    n_copied = 0
    n_skipped_exists = 0
    n_failed = 0
    for _, r in tqdm(df_match.iterrows(), total=len(df_match), desc=f"  {label} copy"):
        if not safety_ok(): break

        primary_label = r["_match_label"]
        if not primary_label or pd.isna(primary_label): continue
        rel_path = str(r[name_col])
        src = audio_root / rel_path
        if not src.exists():
            # Try basename only
            for c in audio_root.rglob(Path(rel_path).name):
                src = c; break
        if not src.exists():
            n_failed += 1
            continue

        dst_dir = AUDIO_DIR / source_id / safe_dir(primary_label)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size > 100:
            n_skipped_exists += 1
            continue

        try:
            shutil.copy2(src, dst)
            n_copied += 1
            all_metadata.append({
                "filename": str(dst.relative_to(OUT_DIR)),
                "primary_label": primary_label,
                "scientific_name": str(r.get(sci_col, "") if sci_col else ""),
                "source": source_id,
                "class_name": LABEL_TO_CLASS.get(primary_label, ""),
                "file_size_mb": dst.stat().st_size / 1e6,
            })
        except Exception as e:
            n_failed += 1

    print(f"  [{label}] copied: {n_copied}, skip-exist: {n_skipped_exists}, failed: {n_failed}")
    print(f"  current storage: {check_storage_gb():.2f} GB")
    return n_copied
"""),

        code_cell("bc_competitions", """# ============================================================
# Cell 5: Extract BC2021-2025
# ============================================================
# Path candidates per year
def find_bc_root(year):
    candidates = [
        Path(f"/kaggle/input/competitions/birdclef-{year}"),
        Path(f"/kaggle/input/birdclef-{year}"),
    ]
    return next((p for p in candidates if p.exists()), None)

def find_audio_dir(root):
    audio_candidates = [
        root / "train_audio",
        root / "train_short_audio",
    ]
    return next((p for p in audio_candidates if p.exists()), None)

def find_metadata(root):
    meta_candidates = [
        root / "train_metadata.csv",
        root / "train.csv",
        root / "train_short_audio_metadata.csv",
    ]
    return next((p for p in meta_candidates if p.exists()), None)

for year in [2021, 2022, 2023, 2024, 2025]:
    if not safety_ok(): break
    root = find_bc_root(year)
    if root is None:
        print(f"\\nBC{year} not mounted (rule accept か input attach 確認)")
        continue
    audio_root = find_audio_dir(root)
    meta_path = find_metadata(root)
    if meta_path is None or audio_root is None:
        print(f"\\nBC{year}: meta={meta_path}, audio={audio_root} - skip")
        continue

    df = pd.read_csv(meta_path)
    print(f"\\n[BC{year}] meta={meta_path.name} ({len(df)} rows)")
    print(f"  columns: {df.columns.tolist()[:10]}")

    if "filename" not in df.columns:
        print(f"  [skip] no 'filename' column")
        continue

    label_col = "primary_label" if "primary_label" in df.columns else None
    sci_col = "scientific_name" if "scientific_name" in df.columns else None

    extract_source(f"BC{year}", f"bc{year}", df, audio_root,
                   name_col="filename", label_col=label_col, sci_col=sci_col)
"""),

        code_cell("inat", """# ============================================================
# Cell 6: Extract iNat (shadowdude/train-recordings)
# ============================================================
INAT_CANDIDATES = [
    Path("/kaggle/input/datasets/shadowdude/train-recordings"),
    Path("/kaggle/input/train-recordings"),
    Path("/kaggle/input/shadowdude-train-recordings"),
]
inat_root = next((p for p in INAT_CANDIDATES if p.exists()), None)

if inat_root is None:
    print(f"iNat not mounted")
else:
    print(f"iNat root: {inat_root}")
    # Auto-discover metadata CSV
    csv_files = list(inat_root.rglob("*.csv"))
    if not csv_files:
        print(f"  no CSV found")
    else:
        print(f"  CSVs: {[str(c.relative_to(inat_root)) for c in csv_files[:5]]}")
        # Pick the metadata-looking one
        meta_path = None
        for c in csv_files:
            df_peek = pd.read_csv(c, nrows=10)
            cols = df_peek.columns.tolist()
            if any(s in cols for s in ["scientific_name", "species_name", "species", "filename", "file"]):
                meta_path = c
                break
        if meta_path is None:
            meta_path = csv_files[0]
        df = pd.read_csv(meta_path)
        print(f"  metadata: {len(df)} rows, columns: {df.columns.tolist()[:10]}")

        sci_col = next((c for c in ["scientific_name", "species_name", "species", "name", "taxon"]
                        if c in df.columns), None)
        name_col = next((c for c in ["filename", "file", "audio_id", "path", "filepath", "file_name"]
                         if c in df.columns), None)
        print(f"  sci_col: {sci_col}, name_col: {name_col}")

        if sci_col and name_col:
            extract_source("iNat", "inat", df, inat_root,
                           name_col=name_col, sci_col=sci_col)
        else:
            print(f"  [skip iNat] columns: {df.columns.tolist()}")
"""),

        code_cell("anuraset", """# ============================================================
# Cell 7: Extract AnuraSet
# ============================================================
ANURA_CANDIDATES = [
    Path("/kaggle/input/datasets/bengtlueers/anuraset-v2-raw"),
    Path("/kaggle/input/anuraset-v2-raw"),
    Path("/kaggle/input/bengtlueers-anuraset-v2-raw"),
]
anura_root = next((p for p in ANURA_CANDIDATES if p.exists()), None)

if anura_root is None:
    print(f"AnuraSet not mounted")
else:
    print(f"AnuraSet root: {anura_root}")
    meta_candidates = list(anura_root.rglob("metadata*.csv")) + list(anura_root.rglob("*labels*.csv"))
    if not meta_candidates:
        meta_candidates = list(anura_root.rglob("*.csv"))
    print(f"  CSVs: {[str(c.relative_to(anura_root)) for c in meta_candidates[:5]]}")

    if meta_candidates:
        meta_path = meta_candidates[0]
        df = pd.read_csv(meta_path)
        print(f"  metadata: {len(df)} rows, columns: {df.columns.tolist()[:10]}")

        sci_col = next((c for c in ["scientific_name", "species", "species_name", "label", "taxon"]
                        if c in df.columns), None)
        name_col = next((c for c in ["filename", "file", "audio", "path", "wav_path", "audio_path"]
                         if c in df.columns), None)
        print(f"  sci_col: {sci_col}, name_col: {name_col}")

        if sci_col and name_col:
            extract_source("AnuraSet", "anuraset", df, anura_root,
                           name_col=name_col, sci_col=sci_col)
        else:
            print(f"  [skip AnuraSet] columns: {df.columns.tolist()}")
"""),

        code_cell("summary", """# ============================================================
# Cell 8: Summary + per-source / per-species stats
# ============================================================
if all_metadata:
    final_df = pd.DataFrame(all_metadata)
    print(f"Total extracted: {len(final_df)} files")
    print(f"  by source: {final_df['source'].value_counts().to_dict()}")
    print(f"  by class:  {final_df['class_name'].value_counts().to_dict()}")
    print(f"  unique BC2026 species: {final_df['primary_label'].nunique()} / 234")
    print(f"  total size: {final_df['file_size_mb'].sum()/1024:.2f} GB")
    print(f"  total time: {(time.time()-START_T)/60:.1f} min")

    final_df.to_csv(OUT_DIR / "metadata.csv", index=False)
    print(f"\\nSaved: {OUT_DIR / 'metadata.csv'}")

    # Per-source
    src_sum = final_df.groupby("source").agg(
        n_files=("filename", "count"),
        n_species=("primary_label", "nunique"),
        total_mb=("file_size_mb", "sum"),
    ).reset_index()
    src_sum.to_csv(OUT_DIR / "per_source.csv", index=False)
    print(f"\\n=== Per-source ===")
    print(src_sum.to_string(index=False))

    # Per-species
    sp_sum = final_df.groupby(["primary_label", "class_name"]).agg(
        n_files=("filename", "count"),
        n_sources=("source", "nunique"),
        total_mb=("file_size_mb", "sum"),
    ).reset_index().sort_values("n_files", ascending=False)
    sp_sum.to_csv(OUT_DIR / "per_species.csv", index=False)
    print(f"\\n=== Top 10 species ===")
    print(sp_sum.head(10).to_string(index=False))

    # Missing species
    covered = set(final_df["primary_label"].unique())
    all_labels = set(species_df["primary_label"])
    missing = sorted(all_labels - covered)
    print(f"\\n=== Missing species (no data in any source): {len(missing)} ===")
    if missing:
        miss_df = species_df[species_df["primary_label"].isin(missing)]
        print(f"  by class: {miss_df['class_name'].value_counts().to_dict()}")
        miss_df.to_csv(OUT_DIR / "missing_species.csv", index=False)
else:
    print("No metadata accumulated - all sources may have been skipped")
"""),

        code_cell("upload", """# ============================================================
# Cell 9: Upload as Kaggle Dataset
# ============================================================
import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp047-bc-inat-anura-extracted"
TITLE = "BirdCLEF2026 exp047 BC+iNat+AnuraSet Extracted"

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
                                    version_notes="Phase 2 BC+iNat+AnuraSet extracted",
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
