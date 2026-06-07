"""Push nb_ov_exp102.ipynb to Kaggle.

slug: birdclef2026-exp102-3fold-ov
title: birdclef2026 exp102 3fold ov  (32 chars, under 50)
"""
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

NB = Path(__file__).with_name("nb_ov_exp102.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp102-3fold-ov"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 exp102 3fold ov",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "maekeso/birdclef2026-exp102-l1-3fold-r3p",
        ],
        "kernel_sources": ["ttahara/birdclef-2026-download-wheels"],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", r.url, "Version:", r.version_number)
