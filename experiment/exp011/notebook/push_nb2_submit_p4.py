"""Push exp011 Phase 4 submission notebook to Kaggle (CPU only).

Slug: birdclef2026-exp011-submit-phase4-sed (title 由来)
kernel_sources: Phase 4 学習 NB の出力 (best.pth = Val-B primary)
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

NB = Path(__file__).with_name("nb2_submit_p4.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp011-submit-phase4-sed"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "BirdCLEF2026 exp011 submit phase4 sed",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        # CPU 推論: machine_shape も enable_gpu も書かない
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],
        "kernel_sources": [
            "maekeso/birdclef2026-exp011-train-phase4-sed",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
