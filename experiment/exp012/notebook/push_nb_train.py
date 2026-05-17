"""Push exp012 nb_train.ipynb to Kaggle.

First push:    python push_nb_train.py
Resume push:   python push_nb_train.py --resume
                  (adds self in kernel_sources to inherit prev /kaggle/working/)

Slug: exp012-train-fold0  (T4x2, internet enabled)

Title rule: title -> slug-ifies. Keep title in slug-friendly form.
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

NB   = Path(__file__).with_name("nb_train.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp012-train-fold0"
TITLE = "birdclef2026 exp012 train fold0"  # title slugifies to birdclef2026-exp012-train-fold0

INCLUDE_SELF = ("--resume" in sys.argv) or (os.environ.get("RESUME", "0") == "1")
print(f"INCLUDE_SELF (resume mode): {INCLUDE_SELF}")

kernel_sources = []
if INCLUDE_SELF:
    kernel_sources.append(f"{USER}/{SLUG}")

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
        "machine_shape": "NvidiaTeslaT4",   # T4x2 GPU
        "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "tuckerarrants/birdclef-2026-waveform-cache",
            "tuckerarrants/perch-v2-no-dft-onnx",
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
