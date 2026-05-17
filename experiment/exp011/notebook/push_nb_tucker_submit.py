"""Push Tucker SED ONNX submission notebook to Kaggle (CPU only).

Slug: birdclef2026-tucker-sed-submit
dataset_sources:
  - tuckerarrants/bc2026-distilled-sed-public  (5-fold ONNX models)
  - tuckerarrants/perch-v2-no-dft-onnx         (onnxruntime wheel)
"""
import json, os, tempfile, shutil
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kaggle_json) as f:
        creds = json.load(f)
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB   = Path(__file__).with_name("nb_tucker_submit.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-tucker-sed-submit"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 tucker sed submit",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        # CPU 推論: machine_shape も enable_gpu も書かない
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "tuckerarrants/perch-v2-no-dft-onnx",
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
