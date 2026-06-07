"""Generate NB to extract BC2026 target species audio from BC21-25 + iNat + AnuraSet → 1 Kaggle Dataset."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_extract_pretrain.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}
def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}

# Read embedded species list
species_lines = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species.txt").read_text(encoding="utf-8")
# Extract tuple lines
species_block = "\n".join(
    line.rstrip() for line in species_lines.split("\n")
    if line.strip().startswith("(") and line.strip().endswith("),")
)

nb = {
    "cells": [
        md_cell("hdr", """# BC2026 Pretrain Data Extractor

BC2021-2025 + iNat + AnuraSet から **BC2026 234 species 全部** の audio を抽出、1 Kaggle Dataset 化。

## Sources

| Source | Kaggle path | 期待 BC2026 species coverage |
|---|---|---|
| BC2021 | `birdclef-2021` (competition) | ~34 Aves species |
| BC2022 | `birdclef-2022` (competition) | ~5 Aves species |
| BC2023 | `birdclef-2023` (competition) | ~1 species |
| BC2024 | `birdclef-2024` (competition) | ~1 species |
| BC2025 | `birdclef-2025` (competition) | ~42 species (Aves + 一部 Amphibia, Mammalia) |
| iNat | `shadowdude/train-recordings` | ~31 species (Aves 152/162 reported) |
| AnuraSet | `bengtlueers/anuraset-v2-raw` | 17 Amphibia species |

## Output

```
/kaggle/working/bc26_pretrain/
├── audio/<source>/<primary_label>/<filename>
├── metadata.csv (unified: filename, primary_label, source, duration, scientific_name)
└── _per_source_summary.csv
```

## 制限

- 容量上限: /kaggle/working **20 GB** → per-species cap で抑制
- Runtime: **9h Kaggle session**

## Politeness

- ローカル Kaggle Dataset mount のみ、外部 API へのアクセスなし (XC は別 NB)
- T4x2 はオプション、CPU でも完結

## 完了後

`maekeso/birdclef2026-pretrain-overlap-audio` 等として Dataset 化
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup
# ============================================================
import os, time, json, shutil, re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

OUT_DIR = Path("/kaggle/working/bc26_pretrain")
AUDIO_DIR = OUT_DIR / "audio"
OUT_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR.mkdir(exist_ok=True, parents=True)

# Caps for storage control
MAX_PER_SPECIES_PER_SOURCE = 100  # 1 source ごと 1 species 最大 100 録音 (storage 制御)
STORAGE_CAP_GB = 18.0              # /kaggle/working 安全マージン
MAX_RUNTIME_HOURS = 8.5

print(f"Output: {OUT_DIR}")
print(f"Storage cap: {STORAGE_CAP_GB} GB")
print(f"Per-species cap per source: {MAX_PER_SPECIES_PER_SOURCE}")
"""),

        code_cell("species_list", """# ============================================================
# Cell 2: BC2026 species list (embedded, 234 species)
# ============================================================
__BC2026_SPECIES = [
__SPECIES_PLACEHOLDER__
]

species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name"])
print(f"BC2026 species: {len(species_df)}")
print(f"  class breakdown: {species_df['class_name'].value_counts().to_dict()}")

# Build lookup maps
SCI_NAME_LC_TO_LABEL = {str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
SCI_LC_SET = set(SCI_NAME_LC_TO_LABEL.keys())
LABEL_SET = set(species_df['primary_label'].tolist())
print(f"Lookups built: {len(SCI_LC_SET)} sci_name + {len(LABEL_SET)} primary_label")
"""),

        code_cell("safety", """# ============================================================
# Cell 3: Safety + helpers
# ============================================================
START_T = time.time()

def safe_dir(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))

def check_storage_gb():
    \"\"\"Current /kaggle/working size in GB.\"\"\"
    try:
        total = sum(f.stat().st_size for f in AUDIO_DIR.rglob("*") if f.is_file())
        return total / 1e9
    except Exception:
        return 0.0

def check_runtime_h():
    return (time.time() - START_T) / 3600

def safety_ok():
    gb = check_storage_gb()
    h = check_runtime_h()
    if gb > STORAGE_CAP_GB:
        print(f"  ⚠ Storage cap: {gb:.2f} GB > {STORAGE_CAP_GB}, stopping")
        return False
    if h > MAX_RUNTIME_HOURS:
        print(f"  ⚠ Runtime cap: {h:.2f}h > {MAX_RUNTIME_HOURS}, stopping")
        return False
    return True

all_metadata = []  # accumulate across sources

def extract_source(label, source_id, df_meta, audio_root, name_col, label_col=None, sci_col=None):
    \"\"\"Common extraction logic.

    df_meta: source metadata DataFrame
    audio_root: root path where audio files live
    name_col: column for relative filename (e.g., 'filename')
    label_col: column for primary_label or ebird code (optional)
    sci_col: column for scientific_name (optional)

    At least one of label_col / sci_col must be provided for matching.
    \"\"\"
    print(f"\\n{'='*60}\\n[{label}] extracting from {source_id}\\n{'='*60}")
    df = df_meta.copy()
    print(f"  source metadata rows: {len(df)}")

    # Match to BC2026
    matched_label = None
    if label_col and label_col in df.columns:
        df["_match_label"] = df[label_col].astype(str)
        df["_in_bc26"] = df["_match_label"].isin(LABEL_SET)
        matched_label = "label"
    elif sci_col and sci_col in df.columns:
        df["_match_sci"] = df[sci_col].astype(str).str.lower()
        df["_in_bc26"] = df["_match_sci"].isin(SCI_LC_SET)
        df["_match_label"] = df["_match_sci"].apply(lambda s: SCI_NAME_LC_TO_LABEL.get(s, None))
        matched_label = "sci_name"
    else:
        print(f"  [SKIP {label}] no matchable column")
        return 0

    df_match = df[df["_in_bc26"]].reset_index(drop=True)
    print(f"  matched ({matched_label}): {len(df_match)} files / {df_match['_match_label'].nunique()} BC2026 species")

    if len(df_match) == 0:
        return 0

    # Per-species cap
    if MAX_PER_SPECIES_PER_SOURCE:
        before = len(df_match)
        df_match = df_match.groupby("_match_label", group_keys=False).head(MAX_PER_SPECIES_PER_SOURCE).reset_index(drop=True)
        print(f"  after per-species cap {MAX_PER_SPECIES_PER_SOURCE}: {before} -> {len(df_match)}")

    # Copy files
    n_copied = 0
    for _, r in tqdm(df_match.iterrows(), total=len(df_match), desc=f"  copy {label}"):
        if not safety_ok():
            break

        primary_label = r["_match_label"]
        rel_path = str(r[name_col])
        src_audio = audio_root / rel_path
        if not src_audio.exists():
            # try direct filename only
            src_audio = audio_root / Path(rel_path).name
            if not src_audio.exists():
                continue

        # destination
        dst_dir = AUDIO_DIR / source_id / safe_dir(primary_label)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_audio = dst_dir / src_audio.name
        if dst_audio.exists() and dst_audio.stat().st_size > 100:
            continue  # already copied (resume)

        try:
            shutil.copy2(src_audio, dst_audio)
            n_copied += 1

            # Append metadata
            all_metadata.append({
                "filename": str(dst_audio.relative_to(OUT_DIR)),
                "primary_label": primary_label,
                "source": source_id,
                "class_name": LABEL_TO_CLASS.get(primary_label, ""),
                "scientific_name": str(r.get(sci_col, "") if sci_col else ""),
                "file_size_mb": dst_audio.stat().st_size / 1e6,
            })
        except Exception as e:
            print(f"  [copy err] {src_audio}: {str(e)[:80]}")

    print(f"  [{label}] copied {n_copied} files")
    return n_copied

print("OK helpers defined")
"""),

        code_cell("extract_bc2021", """# ============================================================
# Cell 4: Extract BC2021 (train_short_audio)
# ============================================================
BC2021_ROOT = Path("/kaggle/input/birdclef-2021")
if BC2021_ROOT.exists():
    meta_candidates = [
        BC2021_ROOT / "train_metadata.csv",
        BC2021_ROOT / "train_short_audio_metadata.csv",
    ]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    audio_root = BC2021_ROOT / "train_short_audio"

    if meta_path and audio_root.exists():
        df_bc21 = pd.read_csv(meta_path)
        print(f"BC2021 metadata: {len(df_bc21)} rows, columns: {df_bc21.columns.tolist()}")
        # filename column varies: 'filename' or built from primary_label + xc_id
        if 'filename' in df_bc21.columns:
            extract_source("BC2021", "bc2021", df_bc21, audio_root,
                           name_col="filename",
                           label_col="primary_label",
                           sci_col="scientific_name" if "scientific_name" in df_bc21.columns else None)
        else:
            # Build filename from primary_label folder + xc_id
            print(f"  no 'filename' column, columns: {df_bc21.columns.tolist()}")
    else:
        print(f"BC2021: metadata or audio_root missing")
else:
    print("BC2021 not mounted - skip")
"""),

        code_cell("extract_bc2022_2025", """# ============================================================
# Cell 5: Extract BC2022/2023/2024/2025
# ============================================================
for year in [2022, 2023, 2024, 2025]:
    if not safety_ok():
        break
    root = Path(f"/kaggle/input/birdclef-{year}")
    if not root.exists():
        print(f"\\nBC{year} not mounted - skip")
        continue

    # Try common metadata locations
    meta_candidates = [
        root / "train_metadata.csv",
        root / "train.csv",
    ]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    audio_candidates = [
        root / "train_audio",
        root / "train_short_audio",
    ]
    audio_root = next((p for p in audio_candidates if p.exists()), None)

    if meta_path is None or audio_root is None:
        print(f"\\nBC{year}: meta={meta_path}, audio={audio_root} - skip")
        continue

    df = pd.read_csv(meta_path)
    print(f"\\n--- BC{year}: meta={meta_path.name} ({len(df)} rows) ---")
    print(f"  columns: {df.columns.tolist()}")

    if "filename" not in df.columns:
        print(f"  [skip BC{year}] no 'filename' column")
        continue

    label_col = "primary_label" if "primary_label" in df.columns else None
    sci_col = "scientific_name" if "scientific_name" in df.columns else None

    extract_source(f"BC{year}", f"bc{year}", df, audio_root,
                   name_col="filename", label_col=label_col, sci_col=sci_col)
"""),

        code_cell("extract_inat", """# ============================================================
# Cell 6: Extract iNat (shadowdude/train-recordings)
# ============================================================
INAT_ROOT_CANDIDATES = [
    Path("/kaggle/input/train-recordings"),
    Path("/kaggle/input/shadowdude-train-recordings"),
]
inat_root = next((p for p in INAT_ROOT_CANDIDATES if p.exists()), None)

if inat_root:
    print(f"iNat root: {inat_root}")
    # Look for metadata CSV (try common names)
    meta_candidates = list(inat_root.rglob("*.csv"))
    print(f"  candidate CSVs: {[str(c.relative_to(inat_root)) for c in meta_candidates[:5]]}")

    if meta_candidates:
        # Try first CSV
        for meta_path in meta_candidates[:3]:
            try:
                df = pd.read_csv(meta_path, nrows=10)
                print(f"\\n  preview {meta_path.name}: {df.columns.tolist()}")
            except Exception as e:
                print(f"  read err: {str(e)[:80]}")

        # Use the most likely metadata file
        meta_path = meta_candidates[0]
        df = pd.read_csv(meta_path)
        print(f"\\niNat metadata: {len(df)} rows")

        # Common iNat metadata columns to try
        name_col_candidates = ["filename", "file", "audio_id", "path", "filepath"]
        sci_col_candidates = ["scientific_name", "species_name", "species", "name"]
        name_col = next((c for c in name_col_candidates if c in df.columns), None)
        sci_col = next((c for c in sci_col_candidates if c in df.columns), None)

        if name_col and sci_col:
            extract_source("iNat", "inat", df, inat_root,
                           name_col=name_col, sci_col=sci_col)
        else:
            print(f"  [skip iNat] need name+sci cols, columns: {df.columns.tolist()}")
    else:
        print("  no CSV found in iNat dataset")
else:
    print("iNat dataset not mounted - skip")
"""),

        code_cell("extract_anuraset", """# ============================================================
# Cell 7: Extract AnuraSet (bengtlueers/anuraset-v2-raw)
# ============================================================
ANURA_CANDIDATES = [
    Path("/kaggle/input/anuraset-v2-raw"),
    Path("/kaggle/input/bengtlueers-anuraset-v2-raw"),
]
anura_root = next((p for p in ANURA_CANDIDATES if p.exists()), None)

if anura_root:
    print(f"AnuraSet root: {anura_root}")
    meta_candidates = list(anura_root.rglob("metadata*.csv")) + list(anura_root.rglob("*labels*.csv"))
    print(f"  candidate CSVs: {[str(c.relative_to(anura_root)) for c in meta_candidates[:5]]}")

    if meta_candidates:
        meta_path = meta_candidates[0]
        df = pd.read_csv(meta_path)
        print(f"\\nAnuraSet metadata: {len(df)} rows, columns: {df.columns.tolist()[:10]}")

        name_col = next((c for c in ["filename", "file", "audio", "path"] if c in df.columns), None)
        sci_col = next((c for c in ["scientific_name", "species", "species_name", "label"] if c in df.columns), None)

        if name_col and sci_col:
            extract_source("AnuraSet", "anuraset", df, anura_root,
                           name_col=name_col, sci_col=sci_col)
        else:
            print(f"  [skip AnuraSet] need name+sci cols, columns: {df.columns.tolist()}")
    else:
        print("  no metadata CSV found in AnuraSet")
else:
    print("AnuraSet not mounted - skip")
"""),

        code_cell("finalize", """# ============================================================
# Cell 8: Finalize metadata + per-source summary
# ============================================================
if all_metadata:
    final_df = pd.DataFrame(all_metadata)
    print(f"Total files extracted: {len(final_df)}")
    print(f"  by source: {final_df['source'].value_counts().to_dict()}")
    print(f"  by class: {final_df['class_name'].value_counts().to_dict()}")
    print(f"  unique BC2026 species: {final_df['primary_label'].nunique()} / 234")
    print(f"  total size: {final_df['file_size_mb'].sum()/1024:.2f} GB")

    final_df.to_csv(OUT_DIR / "metadata.csv", index=False)
    print(f"\\nSaved: {OUT_DIR / 'metadata.csv'}")

    # Per-source summary
    src_summary = final_df.groupby("source").agg(
        n_files=("filename", "count"),
        n_species=("primary_label", "nunique"),
        total_mb=("file_size_mb", "sum"),
    ).reset_index()
    print(f"\\n=== Per-source summary ===")
    print(src_summary.to_string(index=False))
    src_summary.to_csv(OUT_DIR / "_per_source_summary.csv", index=False)

    # Per-species coverage
    sp_summary = final_df.groupby(["primary_label", "class_name"]).agg(
        n_files=("filename", "count"),
        n_sources=("source", "nunique"),
        total_mb=("file_size_mb", "sum"),
    ).reset_index()
    sp_summary = sp_summary.sort_values("n_files", ascending=False)
    print(f"\\n=== Top 10 species ===")
    print(sp_summary.head(10).to_string(index=False))
    print(f"\\n=== Bottom 10 species (with any data) ===")
    print(sp_summary.tail(10).to_string(index=False))
    sp_summary.to_csv(OUT_DIR / "_per_species_summary.csv", index=False)

    # Missing species (no data from any source)
    covered = set(final_df["primary_label"].unique())
    all_labels = set(species_df["primary_label"])
    missing = sorted(all_labels - covered)
    print(f"\\n=== Missing species (no data extracted): {len(missing)} ===")
    if missing:
        miss_df = species_df[species_df["primary_label"].isin(missing)]
        print(f"  by class: {miss_df['class_name'].value_counts().to_dict()}")
        # Save list
        miss_df.to_csv(OUT_DIR / "_missing_species.csv", index=False)
else:
    print("No metadata accumulated - all extraction steps may have failed")
"""),

        code_cell("save_dataset", """# ============================================================
# Cell 9: (Optional) Save as Kaggle Dataset
# ============================================================
import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "birdclef2026-pretrain-overlap-audio"
TITLE = "BirdCLEF2026 Pretrain Overlap Audio"
USER = "maekeso"

DRY_RUN = True  # ★ Set False to upload

if not DRY_RUN:
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "other"}],  # mixed sources, ascertain individual licenses
    }
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        api.dataset_create_version(folder=str(OUT_DIR),
                                    version_notes="BC21-25 + iNat + AnuraSet overlap",
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
    n_files = len(list(AUDIO_DIR.rglob("*"))) if AUDIO_DIR.exists() else 0
    gb = check_storage_gb()
    print(f"DRY_RUN=True - skip upload")
    print(f"To upload: set DRY_RUN=False and re-run this cell")
    print(f"Will upload {n_files} files ({gb:.2f} GB) to {USER}/{SLUG}")
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Inject species list at source level (before json.dump, to keep proper JSON escaping)
for c in nb["cells"]:
    if c.get("id") == "species_list":
        src = "".join(c["source"])
        src = src.replace("__SPECIES_PLACEHOLDER__", species_block)
        c["source"] = src.splitlines(keepends=True)
        break

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}")

# Re-load + verify cell count
nb2 = json.load(open(OUT, encoding="utf-8"))
print(f"  cells: {len(nb2['cells'])}")
for c in nb2["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id'):20s} L={n}")
