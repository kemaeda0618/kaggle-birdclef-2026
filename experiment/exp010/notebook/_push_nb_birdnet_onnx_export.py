"""Push BirdNET head ONNX export NB."""
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
NB = Path(__file__).with_name("nb_birdnet_onnx_export.ipynb")
SLUG = "birdclef2026-exp010-birdnet-onnx-export"
with tempfile.TemporaryDirectory() as td:
    td = Path(td); shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"maekeso/{SLUG}", "title": SLUG, "code_file": NB.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_internet": False,
        "competition_sources": [],
        "dataset_sources": [
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": [
            "maekeso/birdclef2026-exp010-nb4-birdnet",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
