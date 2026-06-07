"""Push exp030 nb_blend_4way to Kaggle (CPU, internet OFF, submission NB).

4-way post-process blend, lightweight (just averages CSVs).
"""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB    = Path(__file__).with_name("nb_blend_4way.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp030-blend-4way"
TITLE = "birdclef2026 exp030 blend 4way"

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
        # CPU only (lightweight CSV averaging)
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],
        "kernel_sources": [
            f"{USER}/birdclef2026-exp019-blend-w35-40-25",
            f"{USER}/birdclef2026-exp028-protossm-mirror",
            f"{USER}/birdclef2026-exp029-r3-single-l1-infer",
            f"{USER}/birdclef2026-exp020-r2-5fold-infer",
        ],
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
