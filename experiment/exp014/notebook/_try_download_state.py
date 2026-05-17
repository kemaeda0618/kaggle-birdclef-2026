"""Try downloading exp014-state dataset directly."""
import json, os, tempfile
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== Try download maekeso/exp014-state ===")
with tempfile.TemporaryDirectory() as td:
    try:
        api.dataset_download_files("maekeso/exp014-state", path=td, unzip=True, quiet=False)
        for p in sorted(Path(td).rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(td)!s:50s} {p.stat().st_size/1024/1024:.2f} MB")
        print(f"\n✓ Dataset exists and downloadable")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
