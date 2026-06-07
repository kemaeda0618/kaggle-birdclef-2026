"""Push exp069c Hybrid pseudo merge NB.

kernel_sources:
  - maekeso/birdclef2026-exp069a-babych-pseudo-gen
  - maekeso/birdclef2026-exp069b-nb4-pseudo
  - maekeso/birdclef2026-exp069b-tucker-pseudo
  - maekeso/birdclef2026-exp069b-exp029-pseudo
"""
import json, os, sys, io, tempfile, shutil
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

NB = Path(__file__).with_name("nb_pseudo_merge_c.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp069c-hybrid-merge"
TITLE = "birdclef2026 exp069c hybrid merge"

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
        "dataset_sources": [],
        "kernel_sources": [
            "maekeso/birdclef2026-exp069a-babych-pseudo-gen",
            "maekeso/birdclef2026-exp069b-nb4-pseudo",
            "maekeso/birdclef2026-exp069b-tucker-pseudo",
            "maekeso/birdclef2026-exp069b-exp029-pseudo",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "")
    if err: print("Error:", err)
