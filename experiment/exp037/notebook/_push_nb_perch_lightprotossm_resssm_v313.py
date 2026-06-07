"""Push exp037 v313 (REAL 0.932 path、mtoshidesu v313904345 base).

Same dependencies as previous port (mtoshidesu/0-932-test-mod-version-4):
- competition: birdclef-2026
- dataset: jaejohn/perch-meta
- dataset: rishikeshjani/perch-onnx-for-birdclef-2026
- kernel_source: ashok205/tf-wheels
- model: google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1

Slug: 既存 `birdclef2026-exp037-perch-lightprotossm-mlp-resssm` の V2 として上書き。
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

NB    = Path(__file__).with_name("nb_perch_lightprotossm_resssm_v313.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp037-perch-lightprotossm-mlp-resssm"   # 既存 slug の V2
TITLE = "birdclef2026 exp037 perch lightprotossm mlp resssm"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # copy with the expected NB filename
    shutil.copy(NB, td / "nb_perch_lightprotossm_resssm.ipynb")
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": "nb_perch_lightprotossm_resssm.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
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
    url = getattr(r, "url", None) or getattr(r, "ref", None)
    ver = getattr(r, "version_number", None)
    err = getattr(r, "error", "") or ""
    print(f"URL: {url}")
    print(f"Version: {ver}")
    if err:
        print(f"Error: {err}")
