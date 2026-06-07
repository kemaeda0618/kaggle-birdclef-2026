"""Re-extract BC2026 species with inat_taxon_id column."""
import pandas as pd
tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
print(f"Total: {len(tax)} species, columns: {tax.columns.tolist()}")

with open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species_with_inat.txt", "w", encoding="utf-8") as f:
    f.write("# (scientific_name, primary_label, class_name, inat_taxon_id)\n")
    for _, r in tax.iterrows():
        sci = str(r["scientific_name"])
        label = str(r["primary_label"])
        cls = str(r["class_name"])
        inat_id = r["inat_taxon_id"]
        if pd.isna(inat_id):
            inat_id_str = "None"
        else:
            inat_id_str = str(int(inat_id))
        f.write(f"    ({sci!r}, {label!r}, {cls!r}, {inat_id_str}),\n")
print(f"Saved _all_species_with_inat.txt")
