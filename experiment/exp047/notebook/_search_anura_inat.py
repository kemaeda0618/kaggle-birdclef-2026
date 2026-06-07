"""Search Kaggle for AnuraSet and iNat datasets."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

queries = [
    "anuraset",
    "anura set",
    "frog audio",
    "inat sounds",
    "inaturalist sounds",
    "inaturalist audio",
    "inat recordings",
    "amphibian audio bioacoustic",
]

for q in queries:
    print(f"\n=== {q} ===")
    try:
        results = api.dataset_list(search=q, sort_by="hottest", max_size=None)
        for r in results[:5]:
            sz = getattr(r, 'totalBytes', None)
            sz_str = f"{sz/1e9:.1f}GB" if sz else "n/a"
            print(f"  {r.ref}  size={sz_str}")
    except Exception as e:
        print(f"  err: {e}")

# Also check original slugs
print("\n=== Check original slugs ===")
for slug in ["bengtlueers/anuraset-v2-raw", "shadowdude/train-recordings"]:
    print(f"\n--- {slug} ---")
    try:
        meta = api.dataset_metadata(slug, "/tmp")
        print(f"  EXISTS")
    except Exception as e:
        print(f"  NOT FOUND: {str(e)[:100]}")
