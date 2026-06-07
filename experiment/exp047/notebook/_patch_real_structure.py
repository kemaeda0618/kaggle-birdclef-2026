"""Apply REAL structure understanding:

iNat: dir name = "00000_Animalia_Phylum_Class_Order_Family_Genus_species"
       → parse genus + species → match BC2026 scientific_name

AnuraSet: dir = site (INCT17, INCT20955, ...), audio = multi-species soundscape
          → no per-species extract possible without external annotation
          → search annotation files deeply, if not found, skip gracefully
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_bc_inat_anura_extract.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_INAT = '''# ============================================================
# Cell: iNat extraction (REAL structure: scientific_name in dir name)
# ============================================================
# Structure: train-recordings/train/{NNNNN}_{kingdom}_{phylum}_{class}_{order}_{family}_{genus}_{species}/
# Example: 00000_Animalia_Arthropoda_Insecta_Blattodea_Hodotermitidae_Hodotermes_mossambicus
# → parse last 2 components → "Hodotermes mossambicus" = scientific_name

INAT_CANDIDATES = [
    Path("/kaggle/input/datasets/shadowdude/train-recordings"),
    Path("/kaggle/input/train-recordings"),
]
inat_root = next((p for p in INAT_CANDIDATES if p.exists()), None)

if inat_root is None:
    msg = f"❌ iNat NOT MOUNTED. Tried: {INAT_CANDIDATES}"
    print(msg); raise RuntimeError(msg)
print(f"iNat root: {inat_root}")

# Walk into train/ subdir
train_dir = inat_root / "train"
if not train_dir.exists():
    # Try alternate
    candidates = list(inat_root.iterdir())
    train_dir = next((c for c in candidates if c.is_dir()), None)
    if train_dir is None:
        msg = f"❌ iNat: no top-level dir under {inat_root}"
        print(msg); raise RuntimeError(msg)
    print(f"  Using {train_dir} as train dir")
else:
    print(f"  Using train_dir: {train_dir}")

# List species dirs (have name pattern with underscores)
species_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
print(f"  Species dirs in {train_dir.name}/: {len(species_dirs)}")
print(f"  Sample names: {[d.name[:80] for d in species_dirs[:3]]}")

# Parse dir name to extract scientific_name
def parse_inat_dir(name):
    """00000_Animalia_Arthropoda_Insecta_Blattodea_Hodotermitidae_Hodotermes_mossambicus
       → ('Hodotermes', 'mossambicus') -> 'Hodotermes mossambicus' """
    parts = name.split("_")
    if len(parts) < 3:
        return None
    # Last 2 parts = genus + species
    genus = parts[-2]
    species = parts[-1]
    return f"{genus} {species}".lower()

# Match dir names to BC2026 species
matched_dirs = []
for d in species_dirs:
    sci_lc = parse_inat_dir(d.name)
    if sci_lc and sci_lc in SCI_LC_SET:
        label = SCI_LC_TO_LABEL[sci_lc]
        matched_dirs.append((d, label, sci_lc))

print(f"\\n  Match to BC2026 scientific_name: {len(matched_dirs)} dirs")
if matched_dirs:
    for d, lbl, sci in matched_dirs[:10]:
        n_files = sum(1 for _ in d.iterdir() if _.is_file())
        print(f"    {sci:35s} → {lbl}  ({n_files} files)")
else:
    msg = f"❌ iNat: 0 BC2026 species matched"
    print(msg)
    sample_parsed = [(d.name, parse_inat_dir(d.name)) for d in species_dirs[:10]]
    print(f"  Sample parsed: {sample_parsed}")
    print(f"  BC2026 sci_lc sample: {list(SCI_LC_SET)[:10]}")
    raise RuntimeError(msg)

# Extract audio
print(f"\\nExtracting iNat (cap {MAX_PER_SPECIES}/species)...")
n_copied = 0
for d, primary_label, sci_lc in tqdm(matched_dirs, desc="iNat"):
    if not safety_ok(): break
    audio_files = []
    for ext in ["*.ogg", "*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(list(d.rglob(ext)))
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
                "scientific_name": sci_lc,
                "source": "inat",
                "class_name": LABEL_TO_CLASS.get(primary_label, ""),
                "file_size_mb": dst.stat().st_size / 1e6,
            })
        except Exception:
            pass

print(f"\\n  iNat copied: {n_copied} files from {len(matched_dirs)} species")
print(f"  storage: {check_storage_gb():.2f} GB")
'''

NEW_ANURA = '''# ============================================================
# Cell: AnuraSet extraction (site-based audio, need external annotation)
# ============================================================
# Structure: anuraset-v2-raw/AnuraSet_v2.0.0_raw/{site_id}/{site_id}_{date}_{time}.wav
# → audio is MULTI-SPECIES soundscape per site
# → species labels require external annotation file

ANURA_CANDIDATES = [
    Path("/kaggle/input/datasets/bengtlueers/anuraset-v2-raw"),
    Path("/kaggle/input/anuraset-v2-raw"),
]
anura_root = next((p for p in ANURA_CANDIDATES if p.exists()), None)

if anura_root is None:
    msg = f"❌ AnuraSet NOT MOUNTED. Tried: {ANURA_CANDIDATES}"
    print(msg); raise RuntimeError(msg)
print(f"AnuraSet root: {anura_root}")

# Find data dir (AnuraSet_v2.0.0_raw subdir or similar)
data_dir = None
for cand in [anura_root / "AnuraSet_v2.0.0_raw", anura_root]:
    if cand.exists() and any(c.is_dir() for c in cand.iterdir()):
        data_dir = cand
        break
if data_dir is None:
    msg = f"❌ AnuraSet: no data dir found"
    print(msg); raise RuntimeError(msg)
print(f"  Data dir: {data_dir}")

# List site dirs
site_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
print(f"  Site dirs: {len(site_dirs)}")
print(f"  Sample: {[d.name for d in site_dirs[:10]]}")

# Search for annotation file (CSV/parquet/json) ANYWHERE
print(f"\\n=== Searching for annotation files ===")
all_annot_files = []
for ext in ["*.csv", "*.CSV", "*.parquet", "*.json", "*.tsv", "*.xlsx"]:
    matches = list(anura_root.rglob(ext))
    if matches:
        all_annot_files.extend(matches)
        print(f"  {ext}: {len(matches)} found")
        for m in matches[:3]:
            print(f"    {m.relative_to(anura_root)}  ({m.stat().st_size/1024:.1f} KB)")

if not all_annot_files:
    # AnuraSet has no annotation in this Dataset
    # Fall back: AnuraSet is unusable for per-species extraction
    print(f"\\n⚠️ AnuraSet: NO annotation files (CSV/parquet/json) found")
    print(f"  AnuraSet structure (site-based soundscape) requires external annotation")
    print(f"  Skipping AnuraSet (no fatal error, continuing without AnuraSet data)")
else:
    # Try to use annotation file
    meta_path = None
    for f in all_annot_files:
        try:
            if f.suffix.lower() == ".csv":
                df_peek = pd.read_csv(f, nrows=5)
            elif f.suffix.lower() == ".parquet":
                df_peek = pd.read_parquet(f).head(5)
            elif f.suffix.lower() == ".json":
                df_peek = pd.read_json(f, lines=True, nrows=5)
            else:
                continue
            cols = df_peek.columns.tolist()
            print(f"  {f.name}: cols={cols[:10]}")

            sci_col = next((c for c in ["scientific_name", "species", "species_name", "label", "taxon"]
                            if c in cols), None)
            name_col = next((c for c in ["filename", "file", "audio", "path", "wav_path", "audio_path", "fname"]
                             if c in cols), None)
            if sci_col and name_col:
                meta_path = f
                print(f"    ★ Found: {sci_col}={sci_col}, name_col={name_col}")
                break
        except Exception as e:
            print(f"  read err: {str(e)[:80]}")

    if meta_path:
        df = pd.read_csv(meta_path) if meta_path.suffix.lower() == ".csv" else pd.read_parquet(meta_path)
        extract_source("AnuraSet", "anuraset", df, anura_root,
                       name_col=name_col, sci_col=sci_col)
    else:
        print(f"\\n⚠️ AnuraSet: annotation files found but no usable species+filename column")
        print(f"  Skipping AnuraSet")
'''

for c in nb["cells"]:
    if c.get("id") == "inat":
        c["source"] = NEW_INAT.splitlines(keepends=True)
        print("[inat] OK real structure (genus+species parse)")
    elif c.get("id") == "anuraset":
        c["source"] = NEW_ANURA.splitlines(keepends=True)
        print("[anuraset] OK site-based structure (search annotations)")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
