"""Generate iNat extract NB (CPU)."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_inat_extract.ipynb")

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
        md_cell("hdr", """# exp047 iNat Extract (CPU)

**Purpose**: iNat (shadowdude/train-recordings) から BC2026 species filter extract

**Input**: shadowdude/train-recordings
**Output**: `maekeso/birdclef2026-exp047-inat-extracted`

**Structure**: `train-recordings/train/{NNNNN}_{kingdom}_{phylum}_{class}_{order}_{family}_{genus}_{species}/audio.ogg`

**Match**: dir 名末尾 (genus_species) → BC2026 scientific_name
"""),
        code_cell("setup", """import os, sys, json, time, re, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

OUT_DIR = Path("/kaggle/working/extracted")
OUT_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR = OUT_DIR / "audio" / "inat"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MAX_PER_SPECIES = 200
STORAGE_CAP_GB = 18.0
START_T = time.time()
"""),
        code_cell("species", f"""__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
SCI_LC_TO_LABEL = {{str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
SCI_LC_SET = set(SCI_LC_TO_LABEL.keys())
print(f"BC2026: {{len(species_df)}} species")
"""),
        code_cell("extract", """def safe_dir(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))

def check_storage_gb():
    try:
        return sum(f.stat().st_size for f in AUDIO_DIR.rglob("*") if f.is_file()) / 1e9
    except: return 0.0

INAT_CANDIDATES = [
    Path("/kaggle/input/datasets/shadowdude/train-recordings"),
    Path("/kaggle/input/train-recordings"),
]
inat_root = next((p for p in INAT_CANDIDATES if p.exists()), None)
if inat_root is None:
    raise RuntimeError(f"iNat NOT MOUNTED: {INAT_CANDIDATES}")
print(f"iNat root: {inat_root}")

train_dir = inat_root / "train"
if not train_dir.exists():
    train_dir = next((c for c in inat_root.iterdir() if c.is_dir()), None)
    if train_dir is None:
        raise RuntimeError("No subdir in iNat root")
print(f"Using train_dir: {train_dir}")

species_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
print(f"Species dirs: {len(species_dirs)}")

def parse_inat_dir(name):
    parts = name.split("_")
    if len(parts) < 3: return None
    return f"{parts[-2]} {parts[-1]}".lower()

matched_dirs = []
for d in species_dirs:
    sci_lc = parse_inat_dir(d.name)
    if sci_lc and sci_lc in SCI_LC_SET:
        matched_dirs.append((d, SCI_LC_TO_LABEL[sci_lc], sci_lc))

print(f"Match to BC2026: {len(matched_dirs)} species")
if not matched_dirs:
    raise RuntimeError("0 BC2026 species matched")

all_metadata = []
n_copied = 0
for d, primary_label, sci_lc in tqdm(matched_dirs, desc="iNat"):
    if check_storage_gb() > STORAGE_CAP_GB: break
    audio_files = []
    for ext in ["*.ogg", "*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(list(d.rglob(ext)))
    audio_files = audio_files[:MAX_PER_SPECIES]
    dst_dir = AUDIO_DIR / safe_dir(primary_label)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in audio_files:
        if check_storage_gb() > STORAGE_CAP_GB: break
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size > 100: continue
        try:
            shutil.copy2(src, dst)
            n_copied += 1
            all_metadata.append({
                "filename": str(dst.relative_to(OUT_DIR)),
                "primary_label": primary_label,
                "scientific_name": sci_lc,
                "source": "inat",
                "class_name": LABEL_TO_CLASS.get(primary_label, ""),
                "file_size_mb": dst.stat().st_size / 1e6,
            })
        except Exception: pass

print(f"\\nCopied: {n_copied} files from {len(matched_dirs)} species")
print(f"Storage: {check_storage_gb():.2f} GB, time: {(time.time()-START_T)/60:.1f} min")

if all_metadata:
    pd.DataFrame(all_metadata).to_csv(OUT_DIR / "metadata.csv", index=False)
"""),
        code_cell("upload", """import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = "maekeso"; SLUG = "birdclef2026-exp047-inat-extracted"; TITLE = "BirdCLEF2026 exp047 iNat Extracted"
DRY_RUN = False
if not DRY_RUN:
    meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    try:
        api.dataset_create_version(folder=str(OUT_DIR), version_notes="iNat extracted",
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
