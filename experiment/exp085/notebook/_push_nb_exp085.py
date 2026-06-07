"""Push exp085 4-model ensemble NB to Kaggle.

This is the inference NB (CPU sub, ~70-80 min total).
- Internet=False (offline wheels for openvino/onnxruntime/tf)
- CPU mode (no GPU needed for inference)
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

NB = Path(__file__).with_name("nb_exp085_4model.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp085-4model-blend"
TITLE = "birdclef2026 exp085 4model blend"

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
        # ★ Mirror exp078 sources + add 2 new (openvino + exp015)
        "dataset_sources": [
            # exp078 と同じ 4 dataset
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            f"{USER}/birdclef2026-exp029-l1-single",
            # ★ exp085 新規 (openvino wheel + exp015 OV)
            "cooolz/openvino-package",
            f"{USER}/birdclef2026-exp015-weights",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],   # ★ TF wheels (exp078 と同じ)
        "model_sources": [
            "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1",  # exp078 と同じ
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
