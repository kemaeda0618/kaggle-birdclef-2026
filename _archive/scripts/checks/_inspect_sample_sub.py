"""Inspect sample_submission.csv to understand test format."""
import pandas as pd
import re

sub = pd.read_csv("sample_submission.csv")
print(f"sample_submission.csv: {len(sub)} rows x {sub.shape[1]} cols")
print()
print("First 5 row_ids:")
for r in sub["row_id"].head(5):
    print(f"  {r}")
print("...")
print("Last 5 row_ids:")
for r in sub["row_id"].tail(5):
    print(f"  {r}")
print()

# Parse for site / time
print("=== Parse test soundscape filenames from row_ids ===")
def parse_rid(rid):
    m = re.search(r"_S(\d{2})_(\d{8})_(\d{6})_(\d+)$", rid)
    if m:
        return f"S{m.group(1)}", m.group(2), int(m.group(3)[:2]), int(m.group(4))
    return None, None, None, None

sub["site"], sub["date"], sub["hour"], sub["end_sec"] = zip(*sub["row_id"].apply(parse_rid))

n_sites = sub["site"].nunique()
print(f"Unique sites in sample_submission: {n_sites}")
print(f"  sites: {sorted(sub['site'].dropna().unique())}")
print()
print(f"Unique dates: {sub['date'].nunique()}")
print()
print(f"Hour distribution:")
print(sub["hour"].value_counts().sort_index().to_string())
print()
print(f"end_sec distribution (per file):")
print(sub["end_sec"].value_counts().sort_index().to_string())
print()
unique_files = sub["row_id"].str.rsplit("_", n=1).str[0].nunique()
print(f"Unique test soundscape files: {unique_files}")
print()
# Per site count
print("=== Test files per site ===")
file_sites = sub[["row_id", "site"]].copy()
file_sites["file_stem"] = file_sites["row_id"].str.rsplit("_", n=1).str[0]
sites_per_file = file_sites.drop_duplicates("file_stem")["site"].value_counts()
print(sites_per_file)
