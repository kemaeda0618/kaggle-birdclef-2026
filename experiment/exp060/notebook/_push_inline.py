"""Push exp060 v2 (inline 4-way) - exp048 + exp020 R2 5-fold."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_blend_inline.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp060-blend-exp048-exp020"  # same slug as v1, version up
TITLE = "birdclef2026 exp060 blend exp048 exp020"

with tempfile.TemporaryDirectory() as td:
    td = Path(td); shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            # exp048 base datasets (inherit)
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            "maekeso/birdclef2026-perch-embed-anura",
            "maekeso/birdclef2026-perch-embed-inat-nonbird",
            "konbu17/bird26-train-audio-head-v1",
            "maekeso/birdclef2026-exp029-l1-single",
            # NEW: exp020 R2 ckpts + ONNX runtime wheel
            "maekeso/birdclef2026-exp020-weights-5fold",
            "romantamrazov/onnxruntime-1-24-4",
        ],
        "kernel_sources": [
            "maekeso/birdclef2026-exp010-nb1-embedding",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print(f"URL: {getattr(r, 'url', None)}")
    print(f"Version: {getattr(r, 'version_number', None)}")
    err = getattr(r, "error", None)
    if err: print(f"Error: {err}")
