"""Push exp011 Phase 2 Step 1 training notebook to Kaggle (T4x2)."""
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

NB = Path(__file__).with_name("nb1_train_p2.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp011-train-phase2-step1"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "BirdCLEF2026 exp011 train Phase2 Step1",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        # T4x2: machine_shape only, no enable_gpu
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
