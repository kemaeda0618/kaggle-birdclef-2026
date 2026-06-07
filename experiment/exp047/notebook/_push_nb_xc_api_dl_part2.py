"""Push Part 2 NB to Kaggle."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_xc_api_dl_part2.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp047-xc-api-dl-part2"
TITLE = "birdclef2026 exp047 xc api dl part2"

with tempfile.TemporaryDirectory() as td:
    td = Path(td); shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python", "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        "enable_gpu": False,  # CPU only
        "competition_sources": ["birdclef-2026"],  # for taxonomy.csv
        "dataset_sources": [], "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print(f"URL: {getattr(r, 'url', None)}")
    print(f"Version: {getattr(r, 'version_number', None)}")
    print(f"Error: {getattr(r, 'error', None)}")
