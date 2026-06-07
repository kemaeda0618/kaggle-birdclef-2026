"""Push exp031: NB4 0.35 / Tucker 0.40 / exp029 (eca_nfnet_l1 R3) 0.25 — 3-way blend.

slug: birdclef2026-exp031-blend-with-exp029-l1
- exp019 (w35-40-25 with e17, LB 0.947) からの e17 → exp029 swap
- 期待 LB 0.947-0.949 (中央 0.948)
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
SLUG = "birdclef2026-exp031-blend-with-exp029-l1"
TITLE = "birdclef2026-exp031-blend-with-exp029-l1"

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
            "maekeso/birdclef2026-exp029-l1-single",   # ★ exp029 ckpt (l1 R3 fold0)
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
