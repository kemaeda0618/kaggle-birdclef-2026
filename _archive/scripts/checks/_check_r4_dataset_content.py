"""Check contents of birdclef2026-exp029-r4-l1 Kaggle Dataset."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp029-r4-l1"
print(f"=== {slug} ===")

# list files
try:
    files = api.dataset_list_files(slug)
    print("Files in Dataset:")
    if hasattr(files, "files"):
        for f in files.files:
            # Try multiple attribute names for size
            size = None
            for attr in ["totalBytes", "total_bytes", "size", "fileSize", "file_size"]:
                if hasattr(f, attr):
                    size = getattr(f, attr)
                    break
            size_str = f"{size/1e6:.1f}MB" if size else "?"
            # Also list all attributes
            print(f"  {f.name}  ({size_str})")
            attrs = [a for a in dir(f) if not a.startswith("_")]
            print(f"    attrs: {attrs[:8]}")
    else:
        print(f"  {files}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")

# Try to view metadata
print("\n=== Metadata download ===")
with tempfile.TemporaryDirectory() as td:
    try:
        api.dataset_metadata(slug, path=td)
        meta_path = Path(td) / "dataset-metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"  title: {meta.get('title', '?')}")
            print(f"  subtitle: {meta.get('subtitle', '?')}")
            print(f"  licenses: {meta.get('licenses', '?')}")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")
