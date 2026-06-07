"""Push exp029 nb_infer_r3_single.ipynb to Kaggle (CPU, internet OFF, submission NB).

R3 single fold (l1 student, exp017 R2 teacher) inference。

Inputs:
- birdclef-2026 (competition)
- maekeso/birdclef2026-exp029-l1-single (R3 ckpt: r3_fold0_ckpt_best_ns22.pth)
- romantamrazov/onnxruntime-1-24-4 (onnxruntime wheel for offline install)
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

NB    = Path(__file__).with_name("nb_infer_r3_single.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp029-r3-single-l1-infer"
TITLE = "birdclef2026 exp029 r3 single l1 infer"

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
        # CPU only for submission (no machine_shape, no enable_gpu)
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp029-l1-single",       # exp029 R3 ckpt (r3_fold0_*.pth)
            "romantamrazov/onnxruntime-1-24-4",            # onnxruntime wheel (cp312、offline install)
        ],
        "kernel_sources": [],
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
