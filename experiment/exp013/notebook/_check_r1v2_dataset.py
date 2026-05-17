"""Check if R1 v2 ONNX dataset exists on Kaggle."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try several possible slug names
candidates = [
    "maekeso/birdclef2026-exp013-r1-v2-student-sed",
    "maekeso/birdclef2026-exp013-r1-v2-hgnet",
    "maekeso/birdclef2026-exp013-r1v2-hgnet",
    "maekeso/birdclef2026-exp013-r1v2",
    "maekeso/birdclef2026-exp013-r1-v2",
    "maekeso/birdclef2026-exp013-student-sed-v2",
    "maekeso/exp013-r1-v2",
    "maekeso/r1-v2-hgnet",
]
# Also list all of maekeso's datasets
print("=== ALL maekeso datasets (recent) ===")
try:
    # public
    print("Public:")
    ds = api.dataset_list(user="maekeso", sort_by="updated", max_size=50)
    for d in ds:
        print(f"  {getattr(d, 'ref', '?')}")
    # private (using mine flag)
    print("\nPrivate / Mine:")
    ds = api.dataset_list(mine=True, sort_by="updated", max_size=50)
    for d in ds:
        print(f"  {getattr(d, 'ref', '?')}")
    for d in ds:
        ref = getattr(d, "ref", "?")
        print(f"  {ref}")
except Exception as e:
    print(f"  failed: {e}")
print()
for slug in candidates:
    print(f"\nTrying: {slug}")
    try:
        files = api.dataset_list_files(slug).files
        print(f"  ✓ EXISTS")
        for f in files:
            print(f"    {f.name:40s} {f.total_bytes/1024/1024:.2f} MB")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {str(e)[:80]}")
