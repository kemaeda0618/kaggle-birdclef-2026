"""Push exp116 (fold0+1) + exp117 (fold0+2) FP32 submission NBs."""
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

USER = "maekeso"
# kernel_sources: e106 FP32 OV (OV4 output) + tucker OV + ttahara wheels + tf-wheels
KS = [
    "ashok205/tf-wheels",
    "maekeso/birdclef2026-tucker-sed-ov",
    "maekeso/birdclef2026-e106-3fold-ov",       # FP32 e106 IR (OV4)
    "ttahara/birdclef-2026-download-wheels",
]
DS = [
    "jaejohn/perch-meta",
    "rishikeshjani/perch-onnx-for-birdclef-2026",
]
MS = ["google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1"]

CONFIGS = [
    ("nb_exp116_fold01.ipynb", "birdclef2026-exp116-fold01", "birdclef2026 exp116 fold01", Path(__file__).parent),
    ("nb_exp117_fold02.ipynb", "birdclef2026-exp117-fold02", "birdclef2026 exp117 fold02",
     Path(__file__).parent.parent.parent / "exp117" / "notebook"),
]

for nb_name, slug, title, nbdir in CONFIGS:
    nb_path = nbdir / nb_name
    assert nb_path.exists(), f"missing {nb_path}"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(nb_path, td / nb_name)
        meta = {
            "id": f"{USER}/{slug}",
            "title": title,
            "code_file": nb_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_internet": False,
            "competition_sources": ["birdclef-2026"],
            "dataset_sources": DS,
            "kernel_sources": KS,
            "model_sources": MS,
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        r = api.kernels_push(str(td))
        print(f"{slug}: URL={r.url} Version={r.version_number}")
