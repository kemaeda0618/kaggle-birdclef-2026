"""Check BirdNET-related Kaggle assets (NB output, datasets)."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# 1. NB1g status
print("=== NB1g BirdNET embed status ===")
try:
    info = api.kernels_status("maekeso/birdclef2026-exp010-nb1g-birdnet-embed")
    print(info)
except Exception as e:
    print(f"  NB1g not found: {e}")

# 2. NB1g output files
print("\n=== NB1g output files ===")
try:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        api.kernels_output("maekeso/birdclef2026-exp010-nb1g-birdnet-embed", path=td)
        for p in sorted(Path(td).rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(td)!s:50s} {p.stat().st_size/1024/1024:.2f} MB")
except Exception as e:
    print(f"  failed: {e}")

# 3. Search any BirdNET-related dataset by maekeso
print("\n=== maekeso BirdNET datasets ===")
candidates = [
    "maekeso/birdclef2026-birdnet-embed",
    "maekeso/birdclef2026-birdnet-embed-sc",
    "maekeso/birdnet-embeddings-bc2026",
    "maekeso/birdclef2026-exp010-birdnet-embed",
]
for s in candidates:
    try:
        files = api.dataset_list_files(s).files
        print(f"  ✓ {s}")
        for f in files[:5]:
            print(f"    {f.name:40s} {f.total_bytes/1024/1024:.2f} MB")
    except Exception:
        print(f"  ✗ {s}")
