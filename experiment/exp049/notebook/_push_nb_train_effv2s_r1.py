"""Push exp049 EffNetV2-S R1 train NB to Kaggle T4x2."""
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

NB = Path(__file__).with_name("nb_train_effv2s_r1.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp049-effv2s-r1"
TITLE = "birdclef2026 exp049 effv2s r1"

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
        "enable_internet": True,  # for timm pretrained DL + dataset upload
        "machine_shape": "NvidiaTeslaT4",  # ★ T4x2
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    url = getattr(r, "url", None)
    ver = getattr(r, "version_number", None)
    err = getattr(r, "error", None)
    print(f"URL: {url}")
    print(f"Version: {ver}")
    print(f"Error: {err}")
    print(f"\nNext steps:")
    print(f"  1. Open {url}")
    print(f"  2. UI で input=birdclef-2026 attach 確認、GPU=T4x2 確認、Internet ON 確認")
    print(f"  3. Save & Run All (~6-8h)")
