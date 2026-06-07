"""Generate BC2021-2025 extract NB (CPU)."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_bc_extract.ipynb")

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
        md_cell("hdr", """# exp047 BC Extract (BC2021-2025, CPU)

**Purpose**: 過去 BC competition から BC2026 species filter extract

**Inputs**: birdclef-2021/2022/2023/2024/2025 (competitions)
**Output**: `maekeso/birdclef2026-exp047-bc-extracted`

**Rule accept 必要** (各 BC URL で Late Submission accept)
"""),
        code_cell("setup", """import os, sys, json, time, re, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

OUT_DIR = Path("/kaggle/working/extracted")
OUT_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR = OUT_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
MAX_PER_SPECIES = 200
STORAGE_CAP_GB = 18.0
START_T = time.time()
print(f"Output: {OUT_DIR}")
"""),
        code_cell("species", f"""__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
SCI_LC_TO_LABEL = {{str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
LABEL_SET = set(species_df['primary_label'])
SCI_LC_SET = set(SCI_LC_TO_LABEL.keys())
print(f"BC2026: {{len(species_df)}} species")
"""),
        code_cell("helpers", """def safe_dir(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))

def check_storage_gb():
    try:
        return sum(f.stat().st_size for f in AUDIO_DIR.rglob("*") if f.is_file()) / 1e9
    except Exception:
        return 0.0

def safety_ok():
    gb = check_storage_gb()
    if gb > STORAGE_CAP_GB:
        print(f"  ⚠ storage cap reached ({gb:.2f} GB)")
        return False
    return True

all_metadata = []
"""),
        code_cell("extract_fn", """def extract_source(label, source_id, df_meta, audio_root, name_col, label_col=None, sci_col=None,
                   max_per_species=MAX_PER_SPECIES):
    print(f"\\n=== [{label}] extract from {source_id} ===")
    df = df_meta.copy()
    print(f"  source rows: {len(df)}")

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
        print(f"  [SKIP] no matchable column")
        return 0

    df_match = df[df["_in_bc26"]].reset_index(drop=True)
    print(f"  matched via {matched_via}: {len(df_match)} ({df_match['_match_label'].nunique()} species)")
    if len(df_match) == 0: return 0

    df_match = df_match.groupby("_match_label", group_keys=False).head(max_per_species).reset_index(drop=True)
    n_copied = 0
    for _, r in tqdm(df_match.iterrows(), total=len(df_match), desc=label):
        if not safety_ok(): break
        primary_label = r["_match_label"]
        if not primary_label or pd.isna(primary_label): continue
        src = audio_root / str(r[name_col])
        if not src.exists():
            for c in audio_root.rglob(Path(str(r[name_col])).name):
                src = c; break
        if not src.exists(): continue

        dst_dir = AUDIO_DIR / source_id / safe_dir(primary_label)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size > 100: continue
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
        except Exception: pass
    print(f"  [{label}] copied: {n_copied}, storage: {check_storage_gb():.2f} GB")
    return n_copied
"""),
        code_cell("bc", """def find_bc_root(year):
    for c in [Path(f"/kaggle/input/competitions/birdclef-{year}"), Path(f"/kaggle/input/birdclef-{year}")]:
        if c.exists(): return c
    return None
def find_audio_dir(root):
    for c in [root/"train_audio", root/"train_short_audio"]:
        if c.exists(): return c
    return None
def find_metadata(root):
    for c in [root/"train_metadata.csv", root/"train.csv", root/"train_short_audio_metadata.csv"]:
        if c.exists(): return c
    return None

for year in [2021, 2022, 2023, 2024, 2025]:
    if not safety_ok(): break
    root = find_bc_root(year)
    if root is None:
        print(f"BC{year} not mounted (rule accept ?)"); continue
    audio_root = find_audio_dir(root); meta_path = find_metadata(root)
    if meta_path is None or audio_root is None:
        print(f"BC{year}: meta={meta_path} audio={audio_root} - skip"); continue
    df = pd.read_csv(meta_path)
    print(f"\\n[BC{year}] meta={meta_path.name} ({len(df)} rows)")
    if "filename" not in df.columns:
        print(f"  skip: no filename col"); continue
    label_col = "primary_label" if "primary_label" in df.columns else None
    sci_col = "scientific_name" if "scientific_name" in df.columns else None
    extract_source(f"BC{year}", f"bc{year}", df, audio_root,
                   name_col="filename", label_col=label_col, sci_col=sci_col)
"""),
        code_cell("summary", """if all_metadata:
    final_df = pd.DataFrame(all_metadata)
    print(f"\\nTotal: {len(final_df)} files")
    print(f"  by source: {final_df['source'].value_counts().to_dict()}")
    print(f"  unique species: {final_df['primary_label'].nunique()}")
    print(f"  size: {final_df['file_size_mb'].sum()/1024:.2f} GB")
    print(f"  time: {(time.time()-START_T)/60:.1f} min")
    final_df.to_csv(OUT_DIR / "metadata.csv", index=False)
else:
    print("No metadata accumulated")
"""),
        code_cell("upload", """import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = "maekeso"; SLUG = "birdclef2026-exp047-bc-extracted"; TITLE = "BirdCLEF2026 exp047 BC Extracted"
DRY_RUN = False
if not DRY_RUN:
    meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        api.dataset_create_version(folder=str(OUT_DIR), version_notes="BC extracted",
                                    dir_mode="zip", quiet=False)
    except Exception:
        try: api.dataset_create_new(folder=str(OUT_DIR), public=False, dir_mode="zip", quiet=False)
        except Exception as e: print(f"err: {str(e)[:200]}")
    print(f"URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT} ({len(nb['cells'])} cells)")
