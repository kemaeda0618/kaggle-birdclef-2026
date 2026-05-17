"""Push ulyanovantonamaranta/birdclef-2026-gate-fake008 to our account.

Slug: birdclef2026-gate-fake008-repro
Sources (from original kernel-metadata.json):
  dataset_sources:
    - tuckerarrants/bc2026-distilled-sed-public
    - jaejohn/perch-meta
    - rishikeshjani/perch-onnx-for-birdclef-2026
  kernel_sources:
    - ashok205/tf-wheels
  model_sources:
    - google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1
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

NB_SRC = Path("C:/Users/maeke/work/kaggle/birdclef-2026/nb_pull/ulyanovantonamaranta/birdclef-2026-gate-fake008.ipynb")
USER   = "maekeso"
SLUG   = "birdclef2026-gate-fake008-repro"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    nb_dst = td / "birdclef-2026-gate-fake008.ipynb"
    shutil.copy(NB_SRC, nb_dst)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 gate fake008 repro",
        "code_file": nb_dst.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        # CPU only (no machine_shape, no enable_gpu)
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": [
            "ashok205/tf-wheels",
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
