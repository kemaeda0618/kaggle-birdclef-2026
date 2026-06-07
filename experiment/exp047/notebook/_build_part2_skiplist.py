"""Extract list of completed species from existing run → embed in part2 NB."""
import pandas as pd
from pathlib import Path

csv_path = Path(r"C:\tmp\birdclef2026-exp047-xc-api-dl_out\xc_api\_species_summary.csv")
df = pd.read_csv(csv_path)
print(f"Total species in summary: {len(df)}")

completed = df[df["completion_pct"] >= 100].reset_index(drop=True)
remaining = df[df["completion_pct"] < 100].reset_index(drop=True)
print(f"  Completed (100%): {len(completed)}")
print(f"  Remaining (<100%): {len(remaining)}")
print(f"  Total in summary: {len(completed) + len(remaining)}")

# Print completed list for embedding
print("\nCOMPLETED_SCI_NAMES = [")
for sci in completed["scientific_name"].tolist():
    print(f"    {sci!r},")
print("]")

# Save remaining names to estimate work
print("\nREMAINING species sample:")
for sci in remaining["scientific_name"].head(10).tolist():
    print(f"  {sci}")
print(f"  ... ({len(remaining)} total)")

# Per-species avg URL count (estimate Part 2 work)
total_urls_remaining = remaining["total_urls"].sum()
print(f"\nRemaining URL count (if all fully DL'd): {total_urls_remaining}")
print(f"Estimated size (avg 1.3 MB/file): {total_urls_remaining * 1.3 / 1024:.2f} GB")

# Also output to a file
with open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_completed_list.py", "w", encoding="utf-8") as f:
    f.write("# Auto-generated: 76 species already fully downloaded in Part 1\n")
    f.write(f"# Total: {len(completed)} species\n\n")
    f.write("COMPLETED_SCI_NAMES = {\n")
    for sci in completed["scientific_name"].tolist():
        f.write(f"    {sci!r},\n")
    f.write("}\n")
print(f"\nSaved skip list to _completed_list.py")
