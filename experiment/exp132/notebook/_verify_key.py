"""Pre-flight: verify the audio-key construction {fname}_{min_t}_{max_t} matches actual AnuraSet filenames.
Catches the v3 matching bug BEFORE running training."""
import pandas as pd
from pathlib import Path

D = Path(__file__).with_name("_data")
m = pd.read_csv(D / "metadata.csv")
print("rows:", len(m), "cols:", list(m.columns)[:8])
print("min_t dtype:", m["min_t"].dtype, "max_t dtype:", m["max_t"].dtype)
print("min_t sample:", m["min_t"].head(5).tolist(), "max_t:", m["max_t"].head(5).tolist())
print("min_t range:", m["min_t"].min(), m["min_t"].max(), "| max_t range:", m["max_t"].min(), m["max_t"].max())
print("fname sample:", m["fname"].head(3).tolist())

# construct keys
keys = [f"{r['fname']}_{int(r['min_t'])}_{int(r['max_t'])}" for _, r in m.head(10).iterrows()]
print("\nconstructed keys (first 10):")
for k in keys:
    print("  ", k)
# Known actual filename stem from dataset tree: INCT17_20191113_040000_0_3
print("\nExpected actual stem format: <site>_<date>_<hour>_<min_t>_<max_t>  e.g. INCT17_20191113_040000_0_3")
# sanity: does fname already include site_date_hour?
print("fname[0] parts:", str(m['fname'].iloc[0]).split('_'))
# check for any non-integer min/max
bad = m[(m['min_t'] != m['min_t'].astype(int)) | (m['max_t'] != m['max_t'].astype(int))]
print("rows with non-integer min_t/max_t:", len(bad))
# unique key count == rows? (no collisions)
allkeys = m["fname"].astype(str) + "_" + m["min_t"].astype(int).astype(str) + "_" + m["max_t"].astype(int).astype(str)
print("unique keys:", allkeys.nunique(), "/ rows", len(m), "(should be equal, no collisions)")
