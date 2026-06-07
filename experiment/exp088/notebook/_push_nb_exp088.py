"""Push exp088 4-model ensemble NB to Kaggle.

exp088 = exp078 + exp082 R2 OV (20s sliding window + Babych mel)
- Internet=False
- CPU mode (~75-85 min)
"""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_exp088_4model.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp088-4model-window-babych"
TITLE = "birdclef2026 exp088 4model window babych"

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
        # ★ Mirror exp085 sources + replace exp015-weights with exp082-weights
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            f"{USER}/birdclef2026-exp029-l1-single",
            "cooolz/openvino-package",
            f"{USER}/birdclef2026-exp082-weights",   # ★ NEW (exp082 R2 OV、20s + Babych)
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
