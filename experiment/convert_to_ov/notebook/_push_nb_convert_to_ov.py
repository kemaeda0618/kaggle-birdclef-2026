"""Push convert-to-OV NB to Kaggle."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("convert_to_ov.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-convert-to-ov"
TITLE = "birdclef2026 convert to ov"

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
        "enable_internet": True,       # for openvino install + upload
        "competition_sources": [],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp015-weights",
            f"{USER}/birdclef2026-exp017-weights",
            f"{USER}/birdclef2026-exp016-weights",   # ★ NEW (R2 ckpt)
            f"{USER}/birdclef2026-exp081-weights",
            f"{USER}/birdclef2026-exp082-weights",
            f"{USER}/birdclef2026-exp029-l1-single", # ★ NEW (R3 fold0 ckpt)
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
