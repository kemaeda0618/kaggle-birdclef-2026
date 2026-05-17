"""Push exp018 v1: NB4 + Tucker + exp017 R2 (eca_nfnet_l0) 3-way blend NB.

slug: birdclef2026-exp018-blend-3way-e17
- v5 (exp010) NB4 + Tucker + exp014 R2 LB 0.945 を base に、3rd 軸を exp017 R2 に swap
- kernel_sources: exp010-nb1-embedding (NB4 Perch embed)
- dataset_sources: tucker SED, exp017 weights, perch onnx, anura/inat embeds, konbu17 head
- ratio: NB4 0.40 / Tucker 0.45 / e17 0.15 (v5 と同比率、e14→e17 swap 効果 isolate)
"""
import json, os, sys, io, tempfile, shutil
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

NB = Path(__file__).with_name("nb_blend.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp018-blend-3way-e17"
TITLE = "birdclef2026-exp018-blend-3way-e17"  # title = slug 形式で slug 確定

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
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            "maekeso/birdclef2026-perch-embed-anura",
            "maekeso/birdclef2026-perch-embed-inat-nonbird",
            "konbu17/bird26-train-audio-head-v1",
            "maekeso/birdclef2026-exp017-weights",
        ],
        "kernel_sources": [
            "maekeso/birdclef2026-exp010-nb1-embedding",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
