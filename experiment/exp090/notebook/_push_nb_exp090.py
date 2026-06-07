"""Push exp090 NB to Kaggle.

exp090 = exp078 base
  + lambda_prior 0.4 → 0.65 (Hour/Site prior)
  + rank_aware_scaling power 0.4 → 0.6
  + TAX_SMOOTHING (genus 0.15 + class 0.05)
  (FCS retained at 0.4 to avoid overcalibration)
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

NB = Path(__file__).with_name("nb_exp090_tuned_tax.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp090-tuned-tax"
TITLE = "birdclef2026 exp090 tuned tax"

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
        # ★ exp078 と完全同一 (新規 dependency なし)
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            f"{USER}/birdclef2026-exp029-l1-single",
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
