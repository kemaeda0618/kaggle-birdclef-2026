"""Fix iNat: handle 'train/' subdir + embed inat_taxon_id in species list.

iNat structure observed:
  train-recordings/
    └── train/
        ├── <taxon_id>/
        │   └── audio.ogg
        ...
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_bc_inat_anura_extract.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Read new species data with inat_taxon_id
species_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species_with_inat.txt").read_text(encoding="utf-8")
species_lines = []
for line in species_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        species_lines.append("    " + s)
species_block = "\n".join(species_lines)
print(f"Embedded species: {len(species_lines)}")

# Replace species cell with inat_taxon_id-enhanced version
NEW_SPECIES_CELL = f"""# ============================================================
# Cell 2: BC2026 species list (234) with inat_taxon_id
# ============================================================
# tuple: (scientific_name, primary_label, class_name, inat_taxon_id)
__BC2026_SPECIES = [
{species_block}
]
species_df = pd.DataFrame(__BC2026_SPECIES, columns=["scientific_name", "primary_label", "class_name", "inat_taxon_id"])
SCI_LC_TO_LABEL = {{str(s).lower(): l for s, l in zip(species_df['scientific_name'], species_df['primary_label'])}}
LABEL_TO_CLASS = dict(zip(species_df['primary_label'], species_df['class_name']))
LABEL_SET = set(species_df['primary_label'])
SCI_LC_SET = set(SCI_LC_TO_LABEL.keys())

# inat_taxon_id mapping (string keys for dir name match)
INAT_ID_TO_LABEL = {{}}
for _, r in species_df.iterrows():
    iid = r["inat_taxon_id"]
    if iid is not None and iid is not pd.NA and not pd.isna(iid):
        INAT_ID_TO_LABEL[str(int(iid))] = r["primary_label"]
print(f"BC2026 species: {{len(species_df)}}")
print(f"  by class: {{species_df['class_name'].value_counts().to_dict()}}")
print(f"  inat_taxon_id mapping: {{len(INAT_ID_TO_LABEL)}} (sample: {{list(INAT_ID_TO_LABEL.keys())[:5]}})")
"""

for c in nb["cells"]:
    if c.get("id") == "species":
        c["source"] = NEW_SPECIES_CELL.splitlines(keepends=True)
        print("[species] OK inat_taxon_id added to embed")
        break

# Update iNat cell to handle train/ subdir + use INAT_ID_TO_LABEL from species cell
NEW_INAT = '''# ============================================================
# Cell: iNat extraction (handles train/ subdir + dir-based)
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
print(f"INAT_ID_TO_LABEL from species cell: {len(INAT_ID_TO_LABEL)} species")

# Inspect top-level
top_entries = sorted(list(inat_root.iterdir()))
print(f"Top entries (1st level): {[e.name for e in top_entries[:10]]} ({len(top_entries)} total)")

# Look for taxon_id dirs. Try multiple depths.
# Depth 0: inat_root directly contains taxon_id dirs
# Depth 1: inat_root/train/ or inat_root/<sub>/ contains taxon_id dirs
# Depth 2: inat_root/train/<sub>/...
taxon_dirs_found = []
search_paths = [inat_root]
for d in inat_root.iterdir():
    if d.is_dir():
        search_paths.append(d)
        # 1 階層深く
        for d2 in d.iterdir() if d.is_dir() else []:
            if d2.is_dir():
                search_paths.append(d2)

print(f"\\nSearching {len(search_paths)} paths for taxon_id dirs...")
for sp in search_paths:
    if not sp.is_dir(): continue
    try:
        dirs_here = [e for e in sp.iterdir() if e.is_dir()]
    except Exception:
        continue
    # Check if dir names are numeric (taxon_id pattern)
    numeric_dirs = [d for d in dirs_here if d.name.isdigit()]
    if len(numeric_dirs) > 10:  # at least 10 numeric dirs = likely taxon_id structure
        print(f"  ★ {sp.relative_to(inat_root) if sp != inat_root else '.'} : {len(numeric_dirs)} numeric dirs (taxon_id pattern)")
        taxon_dirs_found.extend(numeric_dirs)

if not taxon_dirs_found:
    msg = f"❌ iNat: NO taxon_id dirs found at any depth in {inat_root}"
    print(msg)
    print(f"  Tried {len(search_paths)} paths")
    raise RuntimeError(msg)

print(f"\\n  Total numeric (taxon_id) dirs found: {len(taxon_dirs_found)}")
print(f"  Sample: {[d.name for d in taxon_dirs_found[:5]]}")

# Match to BC2026
matched_dirs = []
for d in taxon_dirs_found:
    if d.name in INAT_ID_TO_LABEL:
        matched_dirs.append((d, INAT_ID_TO_LABEL[d.name]))

print(f"\\n  Match to BC2026 inat_taxon_id: {len(matched_dirs)} dirs")
if matched_dirs:
    for d, lbl in matched_dirs[:5]:
        n_files = len(list(d.iterdir())[:100])
        print(f"    {d.name} → {lbl} ({n_files}+ files)")
else:
    msg = f"⚠ iNat: 0 BC2026 species matched (inat_taxon_id mismatch)"
    print(msg)
    print(f"  BC2026 inat_id sample: {list(INAT_ID_TO_LABEL.keys())[:10]}")
    print(f"  iNat dir sample: {[d.name for d in taxon_dirs_found[:10]]}")
    raise RuntimeError(msg)

# Extract audio files
print(f"\\nExtracting iNat audio (cap {MAX_PER_SPECIES}/species)...")
n_copied = 0
for d, primary_label in tqdm(matched_dirs, desc="iNat"):
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
                "scientific_name": species_df[species_df["primary_label"] == primary_label]["scientific_name"].iloc[0],
                "source": "inat",
                "class_name": LABEL_TO_CLASS.get(primary_label, ""),
                "file_size_mb": dst.stat().st_size / 1e6,
            })
        except Exception:
            pass

print(f"\\n  iNat copied: {n_copied} files from {len(matched_dirs)} species")
print(f"  storage: {check_storage_gb():.2f} GB")
'''

for c in nb["cells"]:
    if c.get("id") == "inat":
        c["source"] = NEW_INAT.splitlines(keepends=True)
        print("[inat] OK multi-depth + INAT_ID_TO_LABEL from species cell")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
