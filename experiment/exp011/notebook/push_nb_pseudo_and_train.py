"""Push nb_pseudo_and_train to Kaggle.

First push:    python push_nb_pseudo_and_train.py
Resume push:   python push_nb_pseudo_and_train.py --resume
                  (adds self in kernel_sources to inherit previous /kaggle/working/)

Slug: birdclef2026-pseudo-and-train  (T4x2, internet enabled)
"""
import json, os, sys, tempfile, shutil
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

NB   = Path(__file__).with_name("nb_pseudo_and_train.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-pseudo-and-train"

INCLUDE_SELF = ("--resume" in sys.argv) or (os.environ.get("RESUME", "0") == "1")
print(f"INCLUDE_SELF (resume mode): {INCLUDE_SELF}")

kernel_sources = [
    "maekeso/birdclef2026-exp010-nb1-embedding",
    # mel caches are kernel outputs, not datasets → kernel_sources
    "maekeso/birdclef2026-mel-cache-train-audio-256",
    "maekeso/birdclef2026-mel-cache-train-sc-256",
]
if INCLUDE_SELF:
    kernel_sources.append(f"{USER}/{SLUG}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 pseudo and train",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "machine_shape": "NvidiaTeslaT4",   # T4x2 GPU
        "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
        ],
        "kernel_sources": kernel_sources,
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
