"""Push exp010 species_overlap.ipynb to Kaggle (CPU, internet on)."""
import json, os, tempfile, shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

if not os.environ.get("KAGGLE_API_TOKEN"):
    raise RuntimeError("Set KAGGLE_API_TOKEN env var (KGAT_... bearer token)")

api = KaggleApi()
api.authenticate()

NB = Path(__file__).with_name("species_overlap.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp010-species-overlap"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "BirdCLEF2026 exp010 species overlap",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "")
    if err:
        print("Error:", err)
