"""Push exp034 day-of-year prior NB to Kaggle (CPU, internet OFF)."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB    = Path(__file__).with_name("nb_doy.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp034-day-of-year-prior"  # V1 で title 由来 slug 確定済
TITLE = "birdclef2026 exp034 day-of-year prior"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-perch-embed-anura",
            f"{USER}/birdclef2026-perch-embed-inat-nonbird",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": [
            f"{USER}/birdclef2026-exp010-nb1-embedding",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print(f"URL: {getattr(r, 'url', None) or getattr(r, 'ref', None)}")
    print(f"Version: {getattr(r, 'version_number', None)}")
    err = getattr(r, "error", "") or ""
    if err: print(f"Error: {err}")
