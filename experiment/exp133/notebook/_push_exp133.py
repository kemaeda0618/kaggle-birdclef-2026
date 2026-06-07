"""Push exp133 v2 verify NB."""
import json, os, io, sys, tempfile, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
NB = Path(__file__).with_name("nb_exp133_verify_v2.ipynb")
USER = "maekeso"; SLUG = "birdclef2026-exp133-verify-v2"
with tempfile.TemporaryDirectory() as td:
    td = Path(td); shutil.copy(NB, td/NB.name)
    meta = {
        "id": f"{USER}/{SLUG}", "title": "birdclef2026 exp133 verify v2",
        "code_file": NB.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "maekeso/birdclef2026-amphib-b0-v2",
            "maekeso/birdclef2026-xc-api-dl-part1",
            "maekeso/birdclef2026-xc-api-dl-part2",
            "maekeso/birdclef2026-xc-api-dl-part3",
        ],
        "kernel_sources": [
            "maekeso/birdclef2026-tucker-sed-ov",
            "maekeso/birdclef2026-e106-3fold-ov",
            "ttahara/birdclef-2026-download-wheels",
        ],
    }
    (td/"kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td)); print("URL:", r.url, "V", r.version_number)
