"""Try kernels_pull on user's NB4 NB (latest version)."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp010-nb4-blend-protossm-mlp"
target = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\my_nb4_latest")
target.mkdir(parents=True, exist_ok=True)

try:
    api.kernels_pull(slug, path=str(target), metadata=True)
    print("OK pulled")
    nb_files = list(target.glob("*.ipynb"))
    for nb_file in nb_files:
        print(f"\n=== {nb_file.name} ===")
        nb = json.loads(nb_file.read_text(encoding="utf-8"))
        print(f"  cells: {len(nb['cells'])}")
        for i, c in enumerate(nb["cells"]):
            src = "".join(c.get("source", []))
            first = src.split("\n")[0][:130] if src else "(empty)"
            print(f"  [{i:02d}] {c['cell_type']:8s} {len(src):5d}c | {first}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
