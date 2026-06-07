"""Push exp039 single-NB blend (exp037 LightProtoSSM + Tucker SED + exp029).

Dependencies (NB4 廃止により減量):
- competition: birdclef-2026
- dataset: jaejohn/perch-meta (Perch v2 meta)
- dataset: rishikeshjani/perch-onnx-for-birdclef-2026 (Perch ONNX)
- dataset: tuckerarrants/bc2026-distilled-sed-public (Tucker SED 5-fold ONNX)
- dataset: maekeso/birdclef2026-exp029-l1-single (exp029 ckpt)
- kernel_source: ashok205/tf-wheels (TF 2.20)
- model: google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1

Removed deps (NB4 で使ってたもの):
- maekeso/birdclef2026-perch-embed-anura
- maekeso/birdclef2026-perch-embed-inat-nonbird
- maekeso/birdclef2026-exp010-nb1-embedding (kernel_source)
- konbu17/bird26-train-audio-head-v1
"""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB    = Path(__file__).with_name("nb_blend.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp039-blend-lightssm-tucker-e29"
TITLE = "birdclef2026 exp039 blend lightssm tucker e29"

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
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            "maekeso/birdclef2026-exp029-l1-single",
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
    url = getattr(r, "url", None) or getattr(r, "ref", None)
    ver = getattr(r, "version_number", None)
    err = getattr(r, "error", "") or ""
    print(f"URL: {url}")
    print(f"Version: {ver}")
    if err:
        print(f"Error: {err}")
