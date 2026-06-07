"""Push exp132 amphibian specialist v3 (strong labels, T4x2)."""
import json, os, io, sys, tempfile, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
NB = Path(__file__).with_name("nb_exp132_train_v3.ipynb")
USER = "maekeso"; SLUG = "birdclef2026-exp132-train-v3"
with tempfile.TemporaryDirectory() as td:
    td = Path(td); shutil.copy(NB, td/NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 exp132 train v3",
        "code_file": NB.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": ["mismaresenka/anuraset-preprocessed"],
        "kernel_sources": [],
    }
    (td/"kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td)); print("URL:", r.url, "V", r.version_number)
