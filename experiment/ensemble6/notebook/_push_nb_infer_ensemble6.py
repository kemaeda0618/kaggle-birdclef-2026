"""Push 6-model ensemble inference NB (CPU sub).

Note: openvino is pre-installed on Kaggle competition env? May need pip install at runtime.
If internet=off + no preinstall, attach openvino wheel dataset.

Currently uses Kaggle\'s pre-installed openvino (try first; if fails, attach wheel dataset).
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

NB = Path(__file__).with_name("nb_infer_ensemble6.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-ensemble6-infer"
TITLE = "birdclef2026 ensemble6 infer"

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
        "enable_internet": False,   # ★ wheel install で submit-ready
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "cooolz/openvino-package",   # ★ openvino wheel for offline install
            f"{USER}/birdclef2026-exp015-weights",
            f"{USER}/birdclef2026-exp017-weights",
            f"{USER}/birdclef2026-exp081-weights",
            f"{USER}/birdclef2026-exp082-weights",
            # exp083/084 R1 dropped (PyTorch CPU too slow, 4-model OV only)
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
