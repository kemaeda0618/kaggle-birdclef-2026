"""Reorder cells (iNat/AnuraSet FIRST) + add dir-based label fallback.

For iNat: match by inat_taxon_id (BC2026 taxonomy.csv has inat_taxon_id column)
For AnuraSet: dir structure check + scientific_name fallback
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_bc_inat_anura_extract.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_INAT_CELL = '''# ============================================================
# Cell: iNat extraction (dir-based label fallback)
# ============================================================
# iNat structure: train-recordings/{taxon_id}/{file}.ogg|wav|mp3
# Labels via directory names (iNat taxon ID)
# Match BC2026 species by `inat_taxon_id` column

INAT_CANDIDATES = [
    Path("/kaggle/input/datasets/shadowdude/train-recordings"),
    Path("/kaggle/input/train-recordings"),
    Path("/kaggle/input/shadowdude-train-recordings"),
]
inat_root = next((p for p in INAT_CANDIDATES if p.exists()), None)

if inat_root is None:
    msg = f"❌ iNat NOT MOUNTED. Tried: {INAT_CANDIDATES}"
    print(msg)
    raise RuntimeError(msg)

print(f"iNat root: {inat_root}")

# Inspect top-level structure
top_entries = sorted(list(inat_root.iterdir()))
n_top = len(top_entries)
print(f"Top-level entries: {n_top}")
print(f"  Sample first 10: {[e.name for e in top_entries[:10]]}")

# Detect structure: directories with names that look like iNat taxon IDs (numeric)
top_dirs = [e for e in top_entries if e.is_dir()]
print(f"  Top-level dirs: {len(top_dirs)}")
if top_dirs:
    # Check if dir names are numeric (iNat taxon ID convention)
    numeric_names = [d.name for d in top_dirs[:50] if d.name.isdigit()]
    print(f"  Numeric dir names (sample): {numeric_names[:10]}")
    print(f"  Numeric fraction: {len(numeric_names)}/{min(50, len(top_dirs))}")

# Try CSV first
all_csvs = list(inat_root.rglob("*.csv")) + list(inat_root.rglob("*.CSV"))
print(f"\\nCSV files found: {len(all_csvs)}")
for c in all_csvs[:5]:
    print(f"    {c.relative_to(inat_root)}")

# Get BC2026 inat_taxon_id mapping
# species_df has primary_label, scientific_name, class_name
# BC2026 taxonomy has inat_taxon_id column too
_BC2026_TAX = [
    Path("/kaggle/input/competitions/birdclef-2026/taxonomy.csv"),
    Path("/kaggle/input/birdclef-2026/taxonomy.csv"),
]
_tax_path = next((p for p in _BC2026_TAX if p.exists()), None)
if _tax_path:
    _tax = pd.read_csv(_tax_path)
    if "inat_taxon_id" in _tax.columns:
        INAT_ID_TO_LABEL = {str(int(r["inat_taxon_id"])): r["primary_label"]
                            for _, r in _tax.iterrows()
                            if pd.notna(r["inat_taxon_id"])}
        print(f"\\n  BC2026 inat_taxon_id mapping: {len(INAT_ID_TO_LABEL)} species")
    else:
        INAT_ID_TO_LABEL = {}
        print(f"\\n  BC2026 taxonomy missing inat_taxon_id column")
else:
    INAT_ID_TO_LABEL = {}
    print(f"\\n  BC2026 taxonomy not found, dir-based fallback limited")

# Approach 1: Match dir names with BC2026 inat_taxon_id
matched_dirs = []
for d in top_dirs:
    if d.name in INAT_ID_TO_LABEL:
        matched_dirs.append((d, INAT_ID_TO_LABEL[d.name]))

print(f"\\nDir-based match (iNat taxon ID): {len(matched_dirs)} dirs matched to BC2026 species")
if matched_dirs:
    print(f"  Sample matches:")
    for d, lbl in matched_dirs[:5]:
        n_files = len(list(d.iterdir()))
        print(f"    {d.name} → {lbl} ({n_files} files)")

if not matched_dirs:
    msg = f"❌ iNat: 0 BC2026 species matched via iNat taxon ID directory names."
    print(msg)
    print(f"  Top dir samples: {[d.name for d in top_dirs[:20]]}")
    print(f"  BC2026 inat_taxon_id sample: {list(INAT_ID_TO_LABEL.keys())[:10]}")
    raise RuntimeError(msg)

# Extract audio files from matched dirs
print(f"\\nExtracting iNat audio (dir-based, cap {MAX_PER_SPECIES}/species)...")
n_copied = 0
n_failed = 0
for d, primary_label in tqdm(matched_dirs, desc="iNat extract"):
    if not safety_ok():
        break

    audio_files = []
    for ext in ["*.ogg", "*.wav", "*.mp3", "*.flac", "*.OGG", "*.WAV", "*.MP3"]:
        audio_files.extend(list(d.rglob(ext)))

    # Per-species cap
    audio_files = audio_files[:MAX_PER_SPECIES]

    dst_dir = AUDIO_DIR / "inat" / safe_dir(primary_label)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in audio_files:
        if not safety_ok(): break
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size > 100:
            continue
        try:
            shutil.copy2(src, dst)
            n_copied += 1
            all_metadata.append({
                "filename": str(dst.relative_to(OUT_DIR)),
                "primary_label": primary_label,
                "scientific_name": _tax[_tax["primary_label"] == primary_label]["scientific_name"].iloc[0] if _tax_path else "",
                "source": "inat",
                "class_name": _tax[_tax["primary_label"] == primary_label]["class_name"].iloc[0] if _tax_path else LABEL_TO_CLASS.get(primary_label, ""),
                "file_size_mb": dst.stat().st_size / 1e6,
            })
        except Exception as e:
            n_failed += 1

print(f"  iNat copied: {n_copied} files from {len(matched_dirs)} matched species")
print(f"  Failed: {n_failed}, storage: {check_storage_gb():.2f} GB")
'''

NEW_ANURA_CELL = '''# ============================================================
# Cell: AnuraSet extraction (dir-based or CSV fallback)
# ============================================================
ANURA_CANDIDATES = [
    Path("/kaggle/input/datasets/bengtlueers/anuraset-v2-raw"),
    Path("/kaggle/input/anuraset-v2-raw"),
    Path("/kaggle/input/bengtlueers-anuraset-v2-raw"),
]
anura_root = next((p for p in ANURA_CANDIDATES if p.exists()), None)

if anura_root is None:
    msg = f"❌ AnuraSet NOT MOUNTED. Tried: {ANURA_CANDIDATES}"
    print(msg)
    raise RuntimeError(msg)

print(f"AnuraSet root: {anura_root}")

# Inspect structure
top_entries = sorted(list(anura_root.iterdir()))
print(f"Top-level entries: {len(top_entries)}")
print(f"  Sample: {[e.name for e in top_entries[:15]]}")
top_dirs = [e for e in top_entries if e.is_dir()]
print(f"  Top-level dirs: {len(top_dirs)}")

# Find CSV
all_csvs = list(anura_root.rglob("*.csv")) + list(anura_root.rglob("*.CSV"))
print(f"\\nCSV files: {len(all_csvs)}")
for c in all_csvs[:5]:
    print(f"    {c.relative_to(anura_root)}  ({c.stat().st_size/1024:.1f} KB)")

# Try CSV-based first
extracted_via_csv = False
if all_csvs:
    for c in all_csvs[:5]:
        try:
            df_peek = pd.read_csv(c, nrows=5)
            cols = df_peek.columns.tolist()
            print(f"  {c.relative_to(anura_root)}: cols={cols[:10]}")
            sci_col = next((cc for cc in ["scientific_name", "species", "species_name", "label", "taxon"]
                            if cc in cols), None)
            name_col = next((cc for cc in ["filename", "file", "audio", "path", "wav_path", "audio_path", "fname"]
                             if cc in cols), None)
            if sci_col and name_col:
                df = pd.read_csv(c)
                print(f"    ★ Using {c.name} as metadata ({len(df)} rows)")
                n_before = len(all_metadata)
                extract_source("AnuraSet", "anuraset", df, anura_root,
                               name_col=name_col, sci_col=sci_col)
                n_added = len(all_metadata) - n_before
                if n_added > 0:
                    extracted_via_csv = True
                    print(f"    OK CSV-based: {n_added} files copied")
                    break
        except Exception as e:
            print(f"  read err: {str(e)[:80]}")

# Fallback: dir-based (scientific_name as dir name?)
if not extracted_via_csv:
    print(f"\\n  CSV-based failed/missing, trying dir-based fallback...")
    # AnuraSet dirs might be species names or site IDs
    # Try matching dir names with BC2026 scientific_name (lowercase)
    matched_dirs = []
    for d in top_dirs:
        dn_lower = d.name.lower()
        # Try exact scientific_name match
        if dn_lower in SCI_LC_SET:
            label = SCI_LC_TO_LABEL[dn_lower]
            matched_dirs.append((d, label))
        # Try with underscore → space
        else:
            dn_space = dn_lower.replace("_", " ")
            if dn_space in SCI_LC_SET:
                label = SCI_LC_TO_LABEL[dn_space]
                matched_dirs.append((d, label))

    print(f"  Dir-based match: {len(matched_dirs)} dirs")

    if not matched_dirs:
        print(f"  Top dir samples (lowercase): {[d.name.lower() for d in top_dirs[:20]]}")
        print(f"  BC2026 sci_name sample: {list(SCI_LC_SET)[:10]}")
        # Don't raise - AnuraSet may legitimately have no match
        print(f"  ⚠ No AnuraSet dirs matched (continuing with 0 files)")
    else:
        for d, primary_label in tqdm(matched_dirs, desc="AnuraSet extract"):
            if not safety_ok(): break
            audio_files = []
            for ext in ["*.wav", "*.WAV", "*.ogg", "*.OGG", "*.mp3", "*.MP3", "*.flac"]:
                audio_files.extend(list(d.rglob(ext)))
            audio_files = audio_files[:MAX_PER_SPECIES]
            dst_dir = AUDIO_DIR / "anuraset" / safe_dir(primary_label)
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in audio_files:
                if not safety_ok(): break
                dst = dst_dir / src.name
                if dst.exists() and dst.stat().st_size > 100:
                    continue
                try:
                    shutil.copy2(src, dst)
                    all_metadata.append({
                        "filename": str(dst.relative_to(OUT_DIR)),
                        "primary_label": primary_label,
                        "scientific_name": "",
                        "source": "anuraset",
                        "class_name": "Amphibia",
                        "file_size_mb": dst.stat().st_size / 1e6,
                    })
                except Exception:
                    pass
'''

# Replace cells
for c in nb["cells"]:
    if c.get("id") == "inat":
        c["source"] = NEW_INAT_CELL.splitlines(keepends=True)
        print("[inat] OK dir-based fallback added")
    elif c.get("id") == "anuraset":
        c["source"] = NEW_ANURA_CELL.splitlines(keepends=True)
        print("[anuraset] OK dir-based fallback added")

# Reorder: inat + anuraset BEFORE bc_competitions
cell_ids = [c.get("id") for c in nb["cells"]]
print(f"\nCurrent cell order: {cell_ids}")

# Move inat and anuraset to right after extract_fn (before bc_competitions)
new_order = []
extract_fn_idx = None
for i, c in enumerate(nb["cells"]):
    if c.get("id") == "extract_fn":
        extract_fn_idx = i
        break

# Build new order: pre-extract_fn cells + extract_fn + inat + anuraset + bc + post
pre = []
extract_fn_cell = None
inat_cell = None
anura_cell = None
bc_cell = None
post = []
seen_extract_fn = False
seen_bc = False
for c in nb["cells"]:
    cid = c.get("id")
    if cid == "extract_fn":
        extract_fn_cell = c
        seen_extract_fn = True
    elif cid == "inat":
        inat_cell = c
    elif cid == "anuraset":
        anura_cell = c
    elif cid == "bc_competitions":
        bc_cell = c
        seen_bc = True
    elif not seen_extract_fn:
        pre.append(c)
    else:
        # After extract_fn, before bc — skip (will reorder)
        if not seen_bc and cid not in ("inat", "anuraset"):
            # might be summary/upload cell that comes after bc
            post.append(c)
        else:
            post.append(c)

# Final order: pre + extract_fn + inat + anura + bc + post (dedupe)
new_cells = list(pre)
if extract_fn_cell: new_cells.append(extract_fn_cell)
if inat_cell: new_cells.append(inat_cell)
if anura_cell: new_cells.append(anura_cell)
if bc_cell: new_cells.append(bc_cell)
# Add summary + upload at end (filter to avoid duplicates)
for c in post:
    cid = c.get("id")
    if cid not in ("extract_fn", "inat", "anuraset", "bc_competitions"):
        if c not in new_cells:  # avoid dup
            new_cells.append(c)

nb["cells"] = new_cells
print(f"\nNew cell order: {[c.get('id') for c in nb['cells']]}")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
