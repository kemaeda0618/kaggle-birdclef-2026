"""Patch bc_inat_anura NB:
1. Print full root directory structure for iNat/AnuraSet (top 20)
2. Try more CSV patterns + recursive (any depth)
3. FAIL-FAST: raise error if metadata not found, or sources not mounted
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_bc_inat_anura_extract.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_INAT_CELL = '''# ============================================================
# Cell 6: Extract iNat (shadowdude/train-recordings) — FAIL-FAST
# ============================================================
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
print(f"\\n=== iNat root structure (top 20 entries) ===")
top_entries = sorted(list(inat_root.iterdir()))[:20]
for e in top_entries:
    if e.is_dir():
        try: n_files = len(list(e.iterdir()))
        except: n_files = "?"
        print(f"  [DIR] {e.name}/  ({n_files} entries)")
    else:
        print(f"  [FILE] {e.name}  ({e.stat().st_size/1024:.1f} KB)")

# Find CSVs anywhere in iNat tree (broad search)
print(f"\\n=== Finding CSVs (rglob, any depth) ===")
all_csvs = list(inat_root.rglob("*.csv")) + list(inat_root.rglob("*.CSV"))
print(f"  Found {len(all_csvs)} CSV files")
for c in all_csvs[:10]:
    print(f"    {c.relative_to(inat_root)}  ({c.stat().st_size/1024:.1f} KB)")

# Also check for parquet, tsv, json
print(f"\\n=== Other metadata candidates ===")
for ext in ["*.parquet", "*.tsv", "*.json", "*.txt"]:
    matches = list(inat_root.rglob(ext))[:5]
    if matches:
        print(f"  {ext}: {len(matches)} found")
        for m in matches[:3]:
            print(f"    {m.relative_to(inat_root)}")

if not all_csvs:
    msg = f"❌ iNat: NO metadata CSV found anywhere in {inat_root}. iNat directory structure unexpected."
    print(msg)
    print(f"   Manual check needed: ls -R {inat_root} | head -100")
    raise RuntimeError(msg)

# Try each CSV to find metadata
print(f"\\n=== Testing CSV candidates for metadata ===")
meta_path = None
for c in all_csvs[:20]:
    try:
        df_peek = pd.read_csv(c, nrows=5)
        cols = df_peek.columns.tolist()
        print(f"  {c.relative_to(inat_root)}: cols={cols[:8]}")
        if any(s in cols for s in ["scientific_name", "species_name", "species", "name", "primary_label"]):
            if any(s in cols for s in ["filename", "file", "audio", "path", "filepath", "audio_id"]):
                meta_path = c
                print(f"    ★ MATCH: this looks like metadata")
                break
    except Exception as e:
        print(f"  {c.relative_to(inat_root)}: read err {str(e)[:80]}")

if meta_path is None:
    msg = f"❌ iNat: no CSV has both species + filename columns. Structure unexpected."
    print(msg)
    raise RuntimeError(msg)

df = pd.read_csv(meta_path)
print(f"\\n  Using {meta_path.relative_to(inat_root)}: {len(df)} rows")
print(f"  All columns: {df.columns.tolist()}")

sci_col = next((c for c in ["scientific_name", "species_name", "species", "name", "taxon"]
                if c in df.columns), None)
name_col = next((c for c in ["filename", "file", "audio_id", "path", "filepath", "file_name"]
                 if c in df.columns), None)
print(f"  sci_col: {sci_col}, name_col: {name_col}")

if not (sci_col and name_col):
    msg = f"❌ iNat: cannot identify sci_name or filename columns. cols={df.columns.tolist()}"
    print(msg)
    raise RuntimeError(msg)

n_before = len(all_metadata)
extract_source("iNat", "inat", df, inat_root, name_col=name_col, sci_col=sci_col)
n_added = len(all_metadata) - n_before
if n_added == 0:
    msg = f"⚠️ iNat: 0 BC2026 species matched in metadata. Possible: species names different convention."
    print(msg)
    print(f"   sample {sci_col} values: {df[sci_col].head(10).tolist()}")
    # Don't raise here (some sources legitimately have 0 BC2026 overlap), but warn
'''

NEW_ANURA_CELL = '''# ============================================================
# Cell 7: Extract AnuraSet — FAIL-FAST
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
print(f"\\n=== AnuraSet root structure (top 20 entries) ===")
top_entries = sorted(list(anura_root.iterdir()))[:20]
for e in top_entries:
    if e.is_dir():
        try: n_files = len(list(e.iterdir()))
        except: n_files = "?"
        print(f"  [DIR] {e.name}/  ({n_files} entries)")
    else:
        print(f"  [FILE] {e.name}  ({e.stat().st_size/1024:.1f} KB)")

# Find CSVs anywhere
print(f"\\n=== Finding CSVs (rglob, any depth) ===")
all_csvs = list(anura_root.rglob("*.csv")) + list(anura_root.rglob("*.CSV"))
print(f"  Found {len(all_csvs)} CSV files")
for c in all_csvs[:10]:
    print(f"    {c.relative_to(anura_root)}  ({c.stat().st_size/1024:.1f} KB)")

# Also check parquet etc
for ext in ["*.parquet", "*.tsv", "*.json"]:
    matches = list(anura_root.rglob(ext))[:3]
    if matches:
        print(f"  {ext}: {len(matches)} found")
        for m in matches[:3]:
            print(f"    {m.relative_to(anura_root)}")

if not all_csvs:
    msg = f"❌ AnuraSet: NO metadata CSV found anywhere in {anura_root}."
    print(msg)
    raise RuntimeError(msg)

# Test each CSV
print(f"\\n=== Testing CSV candidates ===")
meta_path = None
for c in all_csvs[:20]:
    try:
        df_peek = pd.read_csv(c, nrows=5)
        cols = df_peek.columns.tolist()
        print(f"  {c.relative_to(anura_root)}: cols={cols[:8]}")
        if any(s in cols for s in ["scientific_name", "species", "species_name", "label", "taxon", "primary_label"]):
            if any(s in cols for s in ["filename", "file", "audio", "path", "wav_path", "audio_path"]):
                meta_path = c
                print(f"    ★ MATCH")
                break
    except Exception as e:
        print(f"  {c.relative_to(anura_root)}: read err {str(e)[:80]}")

if meta_path is None:
    msg = f"❌ AnuraSet: no CSV has both species + filename. Structure unexpected."
    print(msg)
    raise RuntimeError(msg)

df = pd.read_csv(meta_path)
print(f"\\n  Using {meta_path.relative_to(anura_root)}: {len(df)} rows")
print(f"  All columns: {df.columns.tolist()}")

sci_col = next((c for c in ["scientific_name", "species", "species_name", "label", "taxon"]
                if c in df.columns), None)
name_col = next((c for c in ["filename", "file", "audio", "path", "wav_path", "audio_path"]
                 if c in df.columns), None)
print(f"  sci_col: {sci_col}, name_col: {name_col}")

if not (sci_col and name_col):
    msg = f"❌ AnuraSet: cannot identify cols. {df.columns.tolist()}"
    print(msg)
    raise RuntimeError(msg)

n_before = len(all_metadata)
extract_source("AnuraSet", "anuraset", df, anura_root, name_col=name_col, sci_col=sci_col)
n_added = len(all_metadata) - n_before
if n_added == 0:
    print(f"⚠️ AnuraSet: 0 BC2026 species matched.")
    print(f"   sample {sci_col} values: {df[sci_col].head(10).tolist()}")
'''

n = 0
for c in nb["cells"]:
    if c.get("id") == "inat":
        c["source"] = NEW_INAT_CELL.splitlines(keepends=True)
        n += 1
        print("[inat] OK fail-fast + structure debug")
    elif c.get("id") == "anuraset":
        c["source"] = NEW_ANURA_CELL.splitlines(keepends=True)
        n += 1
        print("[anuraset] OK fail-fast + structure debug")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Total changes: {n}")
print(f"Saved {NB}")
