"""Extract Aves species list from BC2026 taxonomy.csv for embedding in NB."""
import pandas as pd
tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
aves = tax[tax["class_name"] == "Aves"].reset_index(drop=True)
print(f"Aves count: {len(aves)}")

with open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_aves_list.py", "w", encoding="utf-8") as f:
    f.write("# Auto-generated from BC2026 taxonomy.csv (Aves class_name)\n")
    f.write(f"# Total: {len(aves)} Aves species\n\n")
    f.write("BC2026_AVES = [\n")
    for sci, label in zip(aves["scientific_name"], aves["primary_label"]):
        f.write(f"    ({sci!r}, {label!r}),\n")
    f.write("]\n")

print("Saved _aves_list.py")
print(f"Sample first 3:")
for sci, label in zip(aves["scientific_name"].head(3), aves["primary_label"].head(3)):
    print(f"  {label}: {sci}")
