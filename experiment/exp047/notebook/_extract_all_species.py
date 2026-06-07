"""Extract all 234 BC2026 species for embedding (scientific_name + primary_label + class_name)."""
import pandas as pd
tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
print(f"Total: {len(tax)} species")
print(f"  class breakdown: {tax['class_name'].value_counts().to_dict()}")

with open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_all_species.txt", "w", encoding="utf-8") as f:
    f.write("# Auto-generated from BC2026 taxonomy.csv\n")
    f.write(f"# Total: {len(tax)} species\n\n")
    for _, r in tax.iterrows():
        sci = str(r["scientific_name"])
        label = str(r["primary_label"])
        cls = str(r["class_name"])
        f.write(f"    ({sci!r}, {label!r}, {cls!r}),\n")

print("Saved _all_species.txt")
