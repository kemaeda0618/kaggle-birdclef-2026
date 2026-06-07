"""Push exp104 (e29 slot -> exp102 3-fold ensemble OV) submission NB."""
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

NB = Path(__file__).with_name("nb_exp104_e29_swap_3fold.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp104-e102-3fold"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 exp104 e102 3fold",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,    # ★ submit NB MUST be False (CLAUDE.md)
        "competition_sources": ["birdclef-2026"],
        # ★ exp090 set MINUS: exp029-l1-single (replaced by exp102 OV) & tucker ONNX (replaced by tucker OV)
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": [
            "ashok205/tf-wheels",                       # same as exp090 (tensorflow wheels)
            "maekeso/birdclef2026-tucker-sed-ov",       # Tucker SED OV IR (5-fold)
            "maekeso/birdclef2026-exp102-3fold-ov",     # exp102 OV IR (fold 0,1,2)
            "ttahara/birdclef-2026-download-wheels",    # openvino offline wheels
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", r.url, "Version:", r.version_number)
