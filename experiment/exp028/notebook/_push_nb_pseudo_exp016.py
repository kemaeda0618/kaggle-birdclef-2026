"""Push exp028 nb_pseudo_exp016.ipynb to Kaggle (GPU T4, internet OFF).

e17 (exp017 R2 single fold) の train_soundscapes 予測を生成、pseudo_exp016.csv として保存。
Multi-teacher R3 pseudo の teacher 3。

Inputs:
- birdclef-2026 (competition、train_soundscapes ~10,658 files)
- maekeso/birdclef2026-exp016-weights (r2_ckpt_best_ns22.pth)
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

NB    = Path(__file__).with_name("nb_pseudo_exp016.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp028-pseudo-exp016"
TITLE = "birdclef2026 exp028 pseudo exp016"

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
        # ★ CPU push (GPU max 2 制限回避)、後で UI で T4 切替
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp016-weights",   # exp017 R2 single fold ckpt
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
