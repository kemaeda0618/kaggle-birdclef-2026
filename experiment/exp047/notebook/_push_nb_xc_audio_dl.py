"""Push nb_xc_audio_dl.ipynb to Kaggle."""
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

NB = Path(__file__).with_name("nb_xc_audio_dl.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp047-xc-audio-dl"
TITLE = "BirdCLEF2026 exp047 XC Audio DL"

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
        "enable_internet": True,  # Required for XC downloads
        "enable_gpu": False,  # CPU only (just downloading)
        "competition_sources": [],  # not needed
        "dataset_sources": [
            "yasunorim/xc-birdclef-2026-target-urls",  # ★ Input URL list
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    url = getattr(r, "url", None)
    ver = getattr(r, "version_number", None)
    print(f"URL: {url}")
    print(f"Version: {ver}")
    print(f"\nNext steps:")
    print(f"  1. Open {url}")
    print(f"  2. Save & Run All (this will trigger DL)")
    print(f"  3. Monitor session, may need restart if hit 9h cap (resume supported)")
    print(f"  4. After complete: Set DRY_RUN=False in Cell 6 and Run Cell 6 to upload Dataset")
