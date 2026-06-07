"""Check overlap of rohanrao XC datasets vs BC2026 species."""
import json, os, tempfile, zipfile
from pathlib import Path
import pandas as pd

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# BC2026 taxonomy
bc26_tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
print(f"BC2026 species: {len(bc26_tax)}")
print(f"  columns: {bc26_tax.columns.tolist()}")
print(f"  taxon classes: {bc26_tax['class_name'].value_counts().to_dict()}")
bc26_sci_names = set(bc26_tax["scientific_name"].astype(str).str.lower())
bc26_labels = set(bc26_tax["primary_label"].astype(str))

# Download XC dataset CSV
print("\n--- Downloading rohanrao XC CSV ---")
with tempfile.TemporaryDirectory() as td:
    api.dataset_download_file("rohanrao/xeno-canto-bird-recordings-extended-a-m", "train_extended.csv", path=td)
    csv_path = next(Path(td).glob("*.csv*"), None)
    if csv_path.suffix == ".zip":
        with zipfile.ZipFile(csv_path) as zf: zf.extractall(td)
        csv_path = next(Path(td).glob("*.csv"), None)
    xc_df = pd.read_csv(csv_path)

print(f"\nXC dataset (A-M, but full CSV): {len(xc_df)} rows")
print(f"  unique ebird_code: {xc_df['ebird_code'].nunique()}")
print(f"  unique sci_name: {xc_df['sci_name'].nunique()}")

# Overlap analysis
xc_sci = set(xc_df["sci_name"].astype(str).str.lower())
xc_ebird = set(xc_df["ebird_code"].astype(str))

ovl_sci = bc26_sci_names & xc_sci
ovl_lab = bc26_labels & xc_ebird

print(f"\n=== Overlap ===")
print(f"  scientific_name overlap: {len(ovl_sci)} / 234 BC2026 species")
print(f"  ebird_code overlap: {len(ovl_lab)} / 234 BC2026 species")

# Show overlap species
ovl_df = bc26_tax[bc26_tax["scientific_name"].astype(str).str.lower().isin(ovl_sci)]
print(f"\n  Overlapping species ({len(ovl_df)}):")
for _, r in ovl_df.iterrows():
    print(f"    {r['primary_label']:20s} {r['scientific_name']:40s} ({r['class_name']})")

# Recordings count for overlap
ovl_records = xc_df[xc_df["sci_name"].astype(str).str.lower().isin(bc26_sci_names)]
print(f"\n  recordings for overlap species: {len(ovl_records)}")
print(f"  total duration: {ovl_records['duration'].sum()/3600:.1f} hours")
