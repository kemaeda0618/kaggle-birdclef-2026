"""Generate small NB: re-upload AnuraSet output from previous kernel run."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_anuraset_upload.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

nb = {
    "cells": [
        md_cell("hdr", """# exp047 AnuraSet Upload-Only NB

**Purpose**: 前回 anuraset-extract v3 run の output (1,425 files, 17 species) を Kaggle Dataset 化

**Input**:
- kernel_sources: `maekeso/birdclef2026-exp047-anuraset-extract` (前 v3 run の output)

**Output**: `maekeso/birdclef2026-exp047-anuraset-extracted`

**No re-extraction** — 既 extracted output を Dataset 化のみ
"""),
        code_cell("setup", """import os, sys, json, shutil
from pathlib import Path

# Find previous AnuraSet NB output mount path
KERNEL_OUTPUT_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp047-anuraset-extract"),
    Path("/kaggle/input/maekeso/birdclef2026-exp047-anuraset-extract"),
]
src_root = next((p for p in KERNEL_OUTPUT_CANDIDATES if p.exists()), None)
if src_root is None:
    # discover
    inp = Path("/kaggle/input")
    print(f"input dir: {[p.name for p in inp.iterdir() if p.is_dir()]}")
    for p in inp.rglob("*"):
        if p.name == "anuraset" and p.is_dir():
            src_root = p.parent.parent
            break
assert src_root is not None, "Previous AnuraSet output not mounted"
print(f"Source root: {src_root}")

# List structure
print(f"\\nSource contents:")
for f in sorted(src_root.iterdir()):
    if f.is_dir():
        try: n_files = sum(1 for _ in f.rglob('*') if _.is_file())
        except: n_files = '?'
        print(f"  [DIR] {f.name}/ ({n_files} files)")
    else:
        print(f"  [FILE] {f.name} ({f.stat().st_size/1024:.1f} KB)")

# Build new output dir with only audio + metadata.csv
OUT_DIR = Path("/kaggle/working/extracted")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Copy audio dir
audio_src = src_root / "audio"
if audio_src.exists():
    audio_dst = OUT_DIR / "audio"
    print(f"\\nCopying audio dir...")
    shutil.copytree(audio_src, audio_dst, dirs_exist_ok=True)
    n_audio = sum(1 for _ in audio_dst.rglob('*') if _.is_file())
    print(f"  audio files copied: {n_audio}")
else:
    # Maybe audio is unzipped to /audio
    pass

# Copy metadata.csv
meta_src = src_root / "metadata.csv"
if meta_src.exists():
    shutil.copy2(meta_src, OUT_DIR / "metadata.csv")
    print(f"  metadata.csv copied")

print(f"\\nOUT_DIR contents:")
for f in sorted(OUT_DIR.iterdir()):
    if f.is_dir():
        n = sum(1 for _ in f.rglob('*') if _.is_file())
        print(f"  [DIR] {f.name}/ ({n} files)")
    else:
        print(f"  [FILE] {f.name} ({f.stat().st_size/1024:.1f} KB)")
"""),
        code_cell("upload", """import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp047-anuraset-extracted"
TITLE = "BirdCLEF2026 exp047 AnuraSet Extracted"

meta = {"title": TITLE, "id": f"{USER}/{SLUG}", "licenses": [{"name": "other"}]}
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Uploading to {USER}/{SLUG}...")
try:
    api.dataset_create_new(folder=str(OUT_DIR), public=False, dir_mode="zip", quiet=False)
    print("OK new dataset created")
except Exception as e:
    print(f"create_new err: {str(e)[:300]}")
    try:
        api.dataset_create_version(folder=str(OUT_DIR), version_notes="re-upload",
                                    dir_mode="zip", quiet=False)
        print("OK version created")
    except Exception as e2:
        print(f"create_version err: {str(e2)[:300]}")
print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
"""),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
