"""Push exp014 R2 nb_infer_r2.ipynb to Kaggle (CPU, internet OFF, submission NB).

Inputs:
- birdclef-2026 (competition)
- tuckerarrants/perch-v2-no-dft-onnx (env parity, NOT used at inference)
- kernel_sources: maekeso/birdclef2026-exp014-train-r2 (R2 train output → ckpts)
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

NB    = Path(__file__).with_name("nb_infer_r2.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp014-infer-r2"
TITLE = "birdclef2026 exp014 infer r2"


def push(kernel_sources):
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
            # CPU only for submission
            "enable_internet": False,
            "competition_sources": ["birdclef-2026"],
            "dataset_sources": [
                "tuckerarrants/perch-v2-no-dft-onnx",
            ],
            "kernel_sources": kernel_sources,
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return api.kernels_push(str(td))


KSRC_FULL = [f"{USER}/birdclef2026-exp014-train-r2"]

try:
    print(f"Pushing with kernel_sources={KSRC_FULL}")
    r = push(KSRC_FULL)
    url = getattr(r, "url", None) or getattr(r, "ref", None)
    ver = getattr(r, "version_number", None)
    err = getattr(r, "error", "") or ""
    print(f"URL: {url}")
    print(f"Version: {ver}")
    if err:
        print(f"Error: {err}")
        if "not found" in err.lower() or "404" in err:
            print("\nR2 train kernel not yet pushed/found — retrying with no kernel_sources")
            r = push([])
            url = getattr(r, "url", None) or getattr(r, "ref", None)
            ver = getattr(r, "version_number", None)
            print(f"URL: {url}")
            print(f"Version: {ver}")
except Exception as e:
    msg = str(e)
    print(f"Push failed ({msg[:200]})")
    if "not found" in msg.lower() or "404" in msg:
        print("\nR2 train kernel not yet pushed — retrying with no kernel_sources")
        r = push([])
        url = getattr(r, "url", None) or getattr(r, "ref", None)
        ver = getattr(r, "version_number", None)
        print(f"URL: {url}")
        print(f"Version: {ver}")
    else:
        raise
