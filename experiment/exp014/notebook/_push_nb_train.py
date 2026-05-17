"""Push exp014 nb_train.ipynb to Kaggle as T4x2 GPU NB with self-resume support.

First push: maekeso/exp014-state dataset may not exist yet — try with it
attached; if Kaggle returns 404 for that dataset, retry without it.

Usage:  python _push_nb_train.py

Slug rule (title slugifies to slug):
  title = "birdclef2026 exp014 train"
  slug  = "birdclef2026-exp014-train"
"""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

# Windows cp932 fix
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Auto-load KGAT_ token from kaggle.json
if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB    = Path(__file__).with_name("nb_train.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp014-train"
TITLE = "birdclef2026 exp014 train"   # slugifies to birdclef2026-exp014-train


def push(dataset_sources):
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
            "enable_internet": True,             # need internet for Kaggle Dataset upload
            "competition_sources": ["birdclef-2026"],
            "dataset_sources": dataset_sources,
            "kernel_sources": [],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return api.kernels_push(str(td))


# Try with exp014-state attached (resume mode). If it doesn't exist yet, fall back.
WITH_STATE = [
    "tuckerarrants/perch-v2-no-dft-onnx",
    f"{USER}/exp014-state",
]
WITHOUT_STATE = [
    "tuckerarrants/perch-v2-no-dft-onnx",
]

try:
    print(f"Pushing with dataset_sources={WITH_STATE}")
    r = push(WITH_STATE)
    err = getattr(r, "error", "") or ""
    if err and ("not found" in err.lower() or "404" in str(err)):
        raise RuntimeError(err)
    url = getattr(r, "url", None) or getattr(r, "ref", None)
    ver = getattr(r, "version_number", None)
    print(f"URL: {url}")
    print(f"Version: {ver}")
    if err:
        print(f"Error: {err}")
except Exception as e:
    msg = str(e)
    print(f"\nPush with state failed ({msg[:200]})")
    if "not found" in msg.lower() or "404" in msg or "exp014-state" in msg:
        print(f"\nRetrying without {USER}/exp014-state (first run, dataset doesn't exist yet)")
        r = push(WITHOUT_STATE)
        url = getattr(r, "url", None) or getattr(r, "ref", None)
        ver = getattr(r, "version_number", None)
        err = getattr(r, "error", "") or ""
        print(f"URL: {url}")
        print(f"Version: {ver}")
        if err:
            print(f"Error: {err}")
    else:
        raise
