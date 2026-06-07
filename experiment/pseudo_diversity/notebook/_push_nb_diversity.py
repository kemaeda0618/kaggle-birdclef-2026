"""Push pseudo diversity analysis NB to Kaggle (CPU, internet OFF)."""
import json, os, io, sys, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB    = Path(__file__).with_name("nb_diversity.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-pseudo-diversity-analysis"
TITLE = "birdclef2026 pseudo diversity analysis"

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
            "maekeso/exp028-pseudo-eca-nfnet-l0-e17",
            "maekeso/exp028-pseudo-eca-nfnet-l0-r2",
            "maekeso/exp028-pseudo-convnext-pico",
            "maekeso/exp028-pseudo-regnety-008",
            "maekeso/exp028-pseudo-hgnetv2-tucker",
            "maekeso/exp028-pseudo-hgnetv2-r1",
            "maekeso/exp028-pseudo-eca-nfnet-l1-exp029",  # NEW 7th teacher
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    url = getattr(r, "url", None) or getattr(r, "ref", None)
    ver = getattr(r, "version_number", None)
    err = getattr(r, "error", "") or ""
    print(f"URL: {url}")
    print(f"Version: {ver}")
    if err:
        print(f"Error: {err}")
