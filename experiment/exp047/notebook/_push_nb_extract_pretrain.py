"""Push nb_extract_pretrain.ipynb to Kaggle with all needed data sources attached."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_extract_pretrain.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp047-extract-pretrain"
TITLE = "BirdCLEF2026 exp047 Extract Pretrain"

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
        "enable_internet": True,  # save_dataset cell uses api.kaggle.com
        "machine_shape": "NvidiaTeslaT4",  # T4x2 as user requested
        "competition_sources": [
            "birdclef-2021",
            "birdclef-2022",
            "birdclef-2023",
            "birdclef-2024",
            "birdclef-2025",
        ],
        "dataset_sources": [
            "shadowdude/train-recordings",        # iNat
            "bengtlueers/anuraset-v2-raw",        # AnuraSet
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    url = getattr(r, "url", None)
    ver = getattr(r, "version_number", None)
    print(f"URL: {url}")
    print(f"Version: {ver}")
    print(f"\nKaggle UI で input dataset を確認 (SDK attach 不確実):")
    print(f"  - birdclef-2021 / 2022 / 2023 / 2024 / 2025 (competition)")
    print(f"  - shadowdude/train-recordings (iNat)")
    print(f"  - bengtlueers/anuraset-v2-raw (AnuraSet)")
