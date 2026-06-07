"""Fix Stage 2A NB: BC2026 train_audio download via full competition zip + selective unzip.

Kaggle Competition API does NOT serve directory-level files like 'train_audio.zip'.
Must download entire competition as one zip + extract selectively.
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp053\notebook\nb_stage2a_bc2026_finetune.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_DL_BC2026 = '''# ============================================================
# Cell 2: Download BC2026 competition data (full zip + selective unzip)
# ============================================================
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print("OK Kaggle SDK authenticated")

BC_ROOT = Path("/content/bc2026")
BC_ROOT.mkdir(exist_ok=True)
TRAIN_AUDIO = BC_ROOT / "train_audio"

# Check if already downloaded
n_audio = len(list(TRAIN_AUDIO.rglob("*.ogg"))) if TRAIN_AUDIO.exists() else 0
if n_audio > 30000:
    print(f"train_audio already exists ({n_audio} .ogg files), skip download")
else:
    print(f"Downloading entire BC2026 competition (~30-50 GB, ~15-30 min)...")
    print(f"  Note: Kaggle API doesn't allow directory-level download")
    print(f"        Must download full zip + extract selectively")

    # Download full competition zip
    api.competition_download_files("birdclef-2026", path=str(BC_ROOT), quiet=False)

    # Find downloaded zip
    zip_files = list(BC_ROOT.glob("birdclef-2026.zip"))
    if not zip_files:
        zip_files = list(BC_ROOT.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip found in {BC_ROOT}")
    zip_path = zip_files[0]
    print(f"\\n  Downloaded zip: {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB)")

    # Selective unzip (train_audio + meta only, skip test_soundscapes 20+GB)
    import zipfile
    print(f"\\nExtracting train_audio + meta (skip test_soundscapes to save disk)...")
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        # Filter: train_audio/, meta CSVs
        needed = [n for n in all_names
                  if n.startswith("train_audio/")
                  or n in ("train.csv", "taxonomy.csv", "train_soundscapes_labels.csv",
                           "sample_submission.csv", "recording_location.txt")]
        print(f"  Total files in zip: {len(all_names)}")
        print(f"  Extracting: {len(needed)} (train_audio + meta)")
        for i, name in enumerate(needed):
            if i % 5000 == 0:
                print(f"    [{i}/{len(needed)}]")
            zf.extract(name, path=str(BC_ROOT))

    # Delete zip to free disk
    print(f"\\n  Deleting zip to free disk...")
    zip_path.unlink()

# Verify
import pandas as pd
train_csv = BC_ROOT / "train.csv"
assert train_csv.exists(), f"train.csv missing: {train_csv}"
train_df = pd.read_csv(train_csv)
n_audio = len(list(TRAIN_AUDIO.rglob("*.ogg"))) if TRAIN_AUDIO.exists() else 0
print(f"\\n=== Verify ===")
print(f"  train.csv: {len(train_df)} rows")
print(f"  train_audio: {n_audio} .ogg files")
print(f"  taxonomy.csv: exists={(BC_ROOT/'taxonomy.csv').exists()}")
print(f"  train_soundscapes_labels.csv: exists={(BC_ROOT/'train_soundscapes_labels.csv').exists()}")
assert n_audio > 30000, f"Expected >30k audio files, got {n_audio}"
'''

for c in nb["cells"]:
    if c.get("id") == "dl_bc2026":
        c["source"] = NEW_DL_BC2026.splitlines(keepends=True)
        print("[dl_bc2026] OK fixed to use full competition zip + selective unzip")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
