"""Search Kaggle datasets for OpenVINO wheel."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

queries = ["openvino", "openvino wheel", "openvino runtime", "openvino offline"]
seen = set()
for q in queries:
    print(f"\n=== query: {q} ===")
    try:
        results = api.dataset_list(search=q, max_size=15)
        for ds in results[:15]:
            ref = ds.ref
            if ref in seen: continue
            seen.add(ref)
            size = getattr(ds, "size", "?")
            updated = str(getattr(ds, "last_updated", "?"))[:10]
            title = getattr(ds, "title", "")[:60]
            print(f"  {ref:55s} {title} (sz {size}, upd {updated})")
    except Exception as e:
        print(f"  ERR: {e}")
