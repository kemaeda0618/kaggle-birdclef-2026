"""Push exp015 nb_pseudo.ipynb to Kaggle (T4x2 GPU, internet ON, pseudo CSV gen).

Generates R1 pseudo on train_soundscapes and uploads as maekeso/exp015-r1-pseudo.

Inputs:
- birdclef-2026 (competition)
- tuckerarrants/perch-v2-no-dft-onnx (Perch v2 ONNX, env parity)
- kernel_sources: maekeso/birdclef2026-exp015-train (R1 ckpts)
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

NB    = Path(__file__).with_name("nb_pseudo.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp015-pseudo"
TITLE = "birdclef2026 exp015 pseudo"   # slugifies to birdclef2026-exp015-pseudo

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
        "enable_internet": True,             # Kaggle Dataset upload
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "tuckerarrants/perch-v2-no-dft-onnx",
        ],
        "kernel_sources": [
            f"{USER}/birdclef2026-exp015-train",
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
