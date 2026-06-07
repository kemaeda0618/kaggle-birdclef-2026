"""Push 3 separate NBs: BC, iNat, AnuraSet."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB_DIR = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook")
USER = "maekeso"

NBS = [
    {
        "nb_file": "nb_bc_extract.ipynb",
        "slug": "birdclef2026-exp047-bc-extract",
        "title": "birdclef2026 exp047 bc extract",
        "competitions": ["birdclef-2021", "birdclef-2022", "birdclef-2023", "birdclef-2024", "birdclef-2025"],
        "datasets": [],
        "internet": True,
    },
    {
        "nb_file": "nb_inat_extract.ipynb",
        "slug": "birdclef2026-exp047-inat-extract",
        "title": "birdclef2026 exp047 inat extract",
        "competitions": [],
        "datasets": ["shadowdude/train-recordings"],
        "internet": True,
    },
    {
        "nb_file": "nb_anuraset_extract.ipynb",
        "slug": "birdclef2026-exp047-anuraset-extract",
        "title": "birdclef2026 exp047 anuraset extract",
        "competitions": [],
        "datasets": ["bengtlueers/anuraset-v2-raw"],
        "internet": True,  # GitHub annotation DL
    },
]

for spec in NBS:
    NB = NB_DIR / spec["nb_file"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); shutil.copy(NB, td / NB.name)
        meta = {
            "id": f"{USER}/{spec['slug']}",
            "title": spec["title"],
            "code_file": NB.name,
            "language": "python", "kernel_type": "notebook",
            "is_private": True,
            "enable_internet": spec["internet"],
            "competition_sources": spec["competitions"],
            "dataset_sources": spec["datasets"],
            "kernel_sources": [],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            r = api.kernels_push(str(td))
            print(f"\n[{spec['slug']}]")
            print(f"  URL: {getattr(r, 'url', None)}")
            print(f"  Version: {getattr(r, 'version_number', None)}")
            print(f"  Error: {getattr(r, 'error', None)}")
        except Exception as e:
            print(f"\n[{spec['slug']}] ERR: {str(e)[:200]}")
