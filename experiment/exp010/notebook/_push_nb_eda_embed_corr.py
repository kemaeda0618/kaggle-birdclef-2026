"""Push EDA NB for cross-model embedding correlation."""
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

NB = Path(__file__).with_name("nb_eda_embed_corr.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp010-eda-embed-corr"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026-exp010-eda-embed-corr",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],
        "kernel_sources": ["maekeso/birdclef2026-exp010-nb1-embedding"],
        "model_sources": ["dhruvpaidukle/birdnet-onnx/tensorFlow2/default/2"],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
